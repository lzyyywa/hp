import os
import random

from torch.utils.data.dataloader import DataLoader
import tqdm
import test as test
from loss import *
from loss import KLLoss
from loss import hyperbolic_loss_calu
import utils.hyperbolic_utils as L

import torch.multiprocessing
import numpy as np
import json
import math
from utils.ade_utils import emd_inference_opencv_test
from collections import Counter
from clip import clip  

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
        result = result + key + "  " + str(round(test_stats[key], 4)) + "| "
    print(result)
    model.train()
    return loss_avg, test_stats


def save_checkpoint(state, save_path, epoch, best=False):
    filename = os.path.join(save_path, f"epoch_resume.pt")
    torch.save(state, filename)


def rand_bbox(size, lam):
    W = size[-2]
    H = size[-1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int_(W * cut_rat)
    cut_h = np.int_(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


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
    Loss_fn = CrossEntropyLoss()
    log_training = open(os.path.join(config.save_path, 'log.txt'), 'w')

    attr2idx = train_dataset.attr2idx
    obj2idx = train_dataset.obj2idx

    train_pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                                for attr, obj in train_dataset.train_pairs]).cuda()

    # =========================================================================
    # 【核心修复 1】恢复安全的冻结 CLIP 提取，并仅保存 Raw Feature (512维)
    # =========================================================================
    print(">>> [Train] Pre-computing coarse text raw features using frozen CLIP...")
    device = torch.device("cuda")
    
    frozen_clip, _ = clip.load(config.backbone, device=device)
    frozen_clip.eval()
    
    with torch.no_grad():
        v_list = []
        for cv in train_dataset.coarse_attrs:
            tokens = clip.tokenize(f"an action of {cv}").cuda()
            feat = frozen_clip.encode_text(tokens).float()
            v_list.append(feat)
        coarse_v_raw = torch.cat(v_list, dim=0)

        o_list = []
        for co in train_dataset.coarse_objs:
            tokens = clip.tokenize(f"a photo of a {co}").cuda()
            feat = frozen_clip.encode_text(tokens).float()
            o_list.append(feat)
        coarse_o_raw = torch.cat(o_list, dim=0)
        
    del frozen_clip
    print("    Pre-extraction complete.")

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
            
            c_attr_idx = batch[6].cuda()
            c_obj_idx = batch[7].cuda()

            # ==========================================
            # 1. 允许在 FP16 下提取欧式特征
            # ==========================================
            with torch.cuda.amp.autocast(enabled=True):
                p_v, p_o, f, hyp_features = model(batch_img)
                
                train_v_inds, train_o_inds = train_pairs[:, 0], train_pairs[:, 1]
                f_filtered = f[:, train_v_inds, train_o_inds]

                # 【核心修复 2】在每一步前向传播中进行投影，确保 c2c_text 层获得梯度！
                model_ref = model.module if hasattr(model, 'module') else model
                coarse_v_euc = model_ref.c2c_text_v(coarse_v_raw)
                coarse_o_euc = model_ref.c2c_text_o(coarse_o_raw)

            # ==========================================
            # 2. 必须关闭 FP16，纯 FP32 计算双曲映射和 Loss
            # ==========================================
            with torch.cuda.amp.autocast(enabled=False):
                _curv = hyp_features['curv'].float()
                
                v_alpha_fp32 = model_ref.text_alpha_v.exp().float()
                o_alpha_fp32 = model_ref.text_alpha_o.exp().float()
                
                all_coarse_v_hyp = L.exp_map0(coarse_v_euc.float() * v_alpha_fp32, _curv)
                all_coarse_o_hyp = L.exp_map0(coarse_o_euc.float() * o_alpha_fp32, _curv)
                
                batch_coarse_v = all_coarse_v_hyp[c_attr_idx]
                batch_coarse_o = all_coarse_o_hyp[c_obj_idx]
                
                target_pack = (batch_img, batch_verb, batch_obj, batch_target, batch_coarse_v, batch_coarse_o)
                
                # 转换 hyp_features 内容为 fp32
                hyp_features_fp32 = {
                    'v_feat_hyp': hyp_features['v_feat_hyp'].float(),
                    'o_feat_hyp': hyp_features['o_feat_hyp'].float(),
                    'verb_text_hyp': hyp_features['verb_text_hyp'].float(),
                    'obj_text_hyp': hyp_features['obj_text_hyp'].float(),
                    'curv': _curv
                }
                
                predict_pack = (p_v.float(), p_o.float(), f_filtered.float(), hyp_features_fp32)
                
                # 计算损失（内部已含 * 20.0 缩放）
                loss = hyperbolic_loss_calu(predict_pack, target_pack, config)
                
                # 日志打印（显式转 fp32 并 * 20.0）
                loss_com = Loss_fn(f_filtered.float() * 20.0, batch_target)
                loss_verb = Loss_fn(p_v.float() * 20.0, batch_verb)
                loss_obj = Loss_fn(p_o.float() * 20.0, batch_obj)

                loss = loss / config.gradient_accumulation_steps

            # Backward
            scaler.scale(loss).backward()

            if ((bid + 1) % config.gradient_accumulation_steps == 0) or (bid + 1 == len(train_dataloader)):
                
                scaler.unscale_(optimizer)
                # 【对齐上一版】加入梯度截断保护双曲流形
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                
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