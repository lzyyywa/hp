import os
import random

from torch.utils.data.dataloader import DataLoader
import tqdm
import test as test
from loss import *
from loss import KLLoss
# [HYPERBOLIC] 引入双曲工具和新损失计算函数
from loss import hyperbolic_loss_calu
import utils.hyperbolic_utils as L

import torch.multiprocessing
import numpy as np
import json
import math
from utils.ade_utils import emd_inference_opencv_test
from collections import Counter
from clip import clip  # 需要引入clip进行tokenize

from utils.hsic import hsic_normalized_cca


def cal_conditional(attr2idx, obj2idx, set_name, daset):
    def load_split(path):
        with open(path, 'r') as f:
            loaded_data = json.load(f)
        return loaded_data

    train_data = daset.train_data
    val_data = daset.val_data
    test_data = daset.test_data
    all_data = train_data + val_data + test_data
    if set_name == 'test':
        used_data = test_data
    elif set_name == 'all':
        used_data = all_data
    elif set_name == 'train':
        used_data = train_data

    v_o = torch.zeros(size=(len(attr2idx), len(obj2idx)))
    for item in used_data:
        verb_idx = attr2idx[item[1]]
        obj_idx = obj2idx[item[2]]

        v_o[verb_idx, obj_idx] += 1

    v_o_on_v = v_o / (torch.sum(v_o, dim=1, keepdim=True) + 1.0e-6)
    v_o_on_o = v_o / (torch.sum(v_o, dim=0, keepdim=True) + 1.0e-6)

    return v_o_on_v, v_o_on_o


def evaluate(model, dataset, config):
    model.eval()
    evaluator = test.Evaluator(dataset, model=None)
    all_logits, all_attr_gt, all_obj_gt, all_pair_gt, loss_avg = test.predict_logits(
        model, dataset, config)
    test_stats = test.test(
        dataset,
        evaluator,
        all_logits,
        all_attr_gt,
        all_obj_gt,
        all_pair_gt,
        config
    )
    result = ""
    key_set = ["attr_acc", "obj_acc", "ub_seen", "ub_unseen", "ub_all", "best_seen", "best_unseen", "best_hm", "AUC"]

    for key in key_set:
        # if key in key_set:
        result = result + key + "  " + str(round(test_stats[key], 4)) + "| "
    print(result)
    model.train()
    return loss_avg, test_stats


def save_checkpoint(state, save_path, epoch, best=False):
    filename = os.path.join(save_path, f"epoch_resume.pt")
    torch.save(state, filename)


# ========conditional train=
def rand_bbox(size, lam):
    W = size[-2]
    H = size[-1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int_(W * cut_rat)
    cut_h = np.int_(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


# [HYPERBOLIC] Helper to encode coarse text labels
def encode_coarse_text(model, labels, device, mode='verb'):
    """
    Encodes list of text strings into Euclidean features using the model's text encoder.
    """
    # 1. Template & Tokenize
    # Simple template: "a video of [ACTION]"
    prompts = [f"a video of {label}" for label in labels]
    tokenized = clip.tokenize(prompts).to(device)
    
    # 2. Embedding (Access internal token_embedding from learner)
    # Note: Using torch.no_grad() because we don't update text encoder for coarse labels typically
    # unless we want gradients to flow back through coarse labels to the text encoder.
    # Given memory constraints, usually no_grad is safer for this auxiliary branch.
    with torch.no_grad():
        if mode == 'verb':
            learner = model.verb_prompt_learner
            projector = model.c2c_text_v
        else:
            learner = model.obj_prompt_learner
            projector = model.c2c_text_o

        # We need to use the learner's embedding logic. 
        # Standard CoOp implementation:
        embedding = learner.token_embedding(tokenized).type(model.dtype)
        
        # Add positional embedding (borrowed from learner's logic or clip_model)
        # Try to access positional_embedding from learner (custom_clip_c2c logic)
        # If not available in learner, we might need to look at how get_text_learner implements it.
        # For safety/robustness:
        pos_emb = getattr(learner, 'positional_embedding', None)
        if pos_emb is None:
             # Fallback to model.text_encoder context if needed, but learner usually has it.
             # Assuming CustomCLIP structure provided previously.
             pass 
        else:
             embedding = embedding + pos_emb.type(model.dtype)
        
        # 3. Encoder
        features = model.text_encoder(embedding, tokenized)
        
        # 4. Projection (C2C specific linear layer)
        features = projector(features)

        # [CRITICAL FIX] 必须移除归一化，防止双曲 NaN
        # features = torch.nn.functional.normalize(features, dim=-1) 
        
    return features


def c2c_vanilla(model, optimizer, lr_scheduler, config, train_dataset, val_dataset, test_dataset,
                scaler):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )

    model.train()
    best_loss = 1e5
    best_metric = 0
    # 这里的 Loss_fn 仅用于日志打印 (logging)
    Loss_fn = CrossEntropyLoss()
    log_training = open(os.path.join(config.save_path, 'log.txt'), 'w')

    attr2idx = train_dataset.attr2idx
    obj2idx = train_dataset.obj2idx

    # [IMPORTANT] 获取训练对的索引，用于将 3D Logits 过滤为 2D Class Scores
    train_pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                                for attr, obj in train_dataset.train_pairs]).cuda()

    # =========================================================================
    # [PURE HYPERBOLIC] 强制预计算粗粒度特征
    # =========================================================================
    print(">>> [Train] Pre-computing coarse text features for Hyperbolic Entailment...")
    device = torch.device("cuda")
    
    # Handle DataParallel wrapping
    model_ref = model.module if hasattr(model, 'module') else model
    
    if hasattr(train_dataset, 'coarse_attrs'):
        coarse_v_euc = encode_coarse_text(model_ref, train_dataset.coarse_attrs, device, mode='verb')
        coarse_o_euc = encode_coarse_text(model_ref, train_dataset.coarse_objs, device, mode='object')
        print(f"    Coarse Verbs shape: {coarse_v_euc.shape}, Coarse Objects shape: {coarse_o_euc.shape}")
    else:
        raise ValueError("[Error] Dataset missing coarse labels. Cannot run Hierarchical Hyperbolic Training.")

    train_losses = []

    for i in range(config.epoch_start, config.epochs):
        progress_bar = tqdm.tqdm(
            total=len(train_dataloader), desc="epoch % 3d" % (i + 1)
        )

        epoch_train_losses = []
        epoch_com_losses = []
        epoch_oo_losses = []
        epoch_vv_losses = []

        temp_lr = optimizer.param_groups[-1]['lr']
        print(f'Current_lr:{temp_lr}')
        
        for bid, batch in enumerate(train_dataloader):
            batch_img = batch[0].cuda()
            batch_verb = batch[1].cuda()
            batch_obj = batch[2].cuda()
            batch_target = batch[3].cuda()
            
            # 强制解包粗粒度索引
            c_attr_idx = batch[6].cuda()
            c_obj_idx = batch[7].cuda()

            with torch.cuda.amp.autocast(enabled=True):
                # Forward Pass (Pure Hyperbolic)
                # outputs: (verb_logits, obj_logits, pred_com, hyp_features)
                p_v, p_o, f, hyp_features = model(batch_img)
                
                # [CORRECTION] 关键修复：利用 train_pairs 过滤组合 Logits
                # f (pred_com) shape: [Batch, N_Verb, N_Obj] (3D)
                # f_filtered shape: [Batch, N_Valid_Train_Pairs] (2D)
                # batch_target contains indices in [0, N_Valid_Train_Pairs-1]
                train_v_inds, train_o_inds = train_pairs[:, 0], train_pairs[:, 1]
                f_filtered = f[:, train_v_inds, train_o_inds]

                # Dynamic Mapping of Coarse Features (using updated curv/alpha)
                _curv = hyp_features['curv']
                model_ref = model.module if hasattr(model, 'module') else model
                
                all_coarse_v_hyp = L.exp_map0(coarse_v_euc * model_ref.text_alpha_v.exp(), _curv)
                all_coarse_o_hyp = L.exp_map0(coarse_o_euc * model_ref.text_alpha_o.exp(), _curv)
                
                batch_coarse_v = all_coarse_v_hyp[c_attr_idx]
                batch_coarse_o = all_coarse_o_hyp[c_obj_idx]
                
                # Loss Calculation
                target_pack = (batch_img, batch_verb, batch_obj, batch_target, batch_coarse_v, batch_coarse_o)
                # 注意：这里传递的是 filtered 之后的 f_filtered
                predict_pack = (p_v, p_o, f_filtered, hyp_features)
                
                # 计算总损失 (包含 Align + Entail)
                loss = hyperbolic_loss_calu(predict_pack, target_pack, config)
                
                # For Logging only (Raw logits -> CE Loss)
                # 使用 f_filtered 确保维度匹配
                loss_com = Loss_fn(f_filtered, batch_target)
                loss_verb = Loss_fn(p_v, batch_verb)
                loss_obj = Loss_fn(p_o, batch_obj)

                loss = loss / config.gradient_accumulation_steps

            # Backward
            scaler.scale(loss).backward()

            if ((bid + 1) % config.gradient_accumulation_steps == 0) or (bid + 1 == len(train_dataloader)):
                
                # [CRITICAL FIX] 必须先 Unscale 变成真实梯度，再 Clip
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_train_losses.append(loss.item())
            epoch_com_losses.append(loss_com.item())
            epoch_vv_losses.append(loss_verb.item())
            epoch_oo_losses.append(loss_obj.item())

            progress_bar.set_postfix({"train loss": np.mean(epoch_train_losses[-50:])})
            progress_bar.update()

        lr_scheduler.step()
        progress_bar.close()
        # Logging ... (保持原样)
        progress_bar.write(f"epoch {i + 1} train loss {np.mean(epoch_train_losses)}")
        train_losses.append(np.mean(epoch_train_losses))
        log_training.write('\n')
        log_training.write(f"epoch {i + 1} train loss {np.mean(epoch_train_losses)}\n")
        log_training.write(f"epoch {i + 1} com loss {np.mean(epoch_com_losses)}\n")
        log_training.write(f"epoch {i + 1} vv loss {np.mean(epoch_vv_losses)}\n")
        log_training.write(f"epoch {i + 1} oo loss {np.mean(epoch_oo_losses)}\n")

        if (i + 1) % config.save_every_n == 0:
            save_checkpoint({
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict(),
                'scaler': scaler.state_dict(),
            }, config.save_path, i)
            
        key_set = ["attr_acc", "obj_acc", "ub_seen", "ub_unseen", "ub_all", "best_seen", "best_unseen", "best_hm", "AUC"]
        if i % config.eval_every_n == 0 or i + 1 == config.epochs or i >= config.val_epochs_ts:
            print("Evaluating val dataset:")
            loss_avg, val_result = evaluate(model, val_dataset, config)
            result = ""
            for key in val_result:
                if key in key_set:
                    result = result + key + "  " + str(round(val_result[key], 4)) + "| "
            log_training.write('\n')
            log_training.write(result)
            print("Loss average on val dataset: {}".format(loss_avg))
            log_training.write('\n')
            log_training.write("Loss average on val dataset: {}\n".format(loss_avg))
            if config.best_model_metric == "best_loss":
                 if loss_avg.cpu().float() < best_loss:
                    best_loss = loss_avg.cpu().float()
                    print("Evaluating test dataset:")
                    loss_avg, val_result = evaluate(model, test_dataset, config)
                    torch.save(model.state_dict(), os.path.join(config.save_path, f"best.pt"))
                    result = ""
                    for key in val_result:
                        if key in key_set:
                            result = result + key + "  " + str(round(val_result[key], 4)) + "| "
                    log_training.write('\n')
                    log_training.write(result)
            else:
                if val_result[config.best_model_metric] > best_metric:
                    best_metric = val_result[config.best_model_metric]
                    print('find best!')
                    loss_avg, val_result = evaluate(model, test_dataset, config)
                    torch.save(model.state_dict(), os.path.join(config.save_path, f"best.pt"))
                    result = ""
                    for key in val_result:
                        if key in key_set:
                            result = result + key + "  " + str(round(val_result[key], 4)) + "| "
                    log_training.write('\n')
                    log_training.write(result)
                    
        log_training.write('\n')
        log_training.flush()
        
        if i + 1 == config.epochs:
            print("Evaluating test dataset on Closed World")
            model.load_state_dict(torch.load(os.path.join(config.save_path, "best.pt")))
            loss_avg, val_result = evaluate(model, test_dataset, config)
            result = ""
            for key in val_result:
                if key in key_set:
                    result = result + key + "  " + str(round(val_result[key], 4)) + "| "
            log_training.write('\n')
            log_training.write(result)