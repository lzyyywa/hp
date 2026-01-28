import os
import random
import tqdm
import numpy as np
import json
import torch
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F

import test as test
# [NEW] Import Helper and the Encapsulated Loss
from loss import H2EMTotalLoss
from utils.hierarchy_helper import HierarchyHelper

def cal_conditional(attr2idx, obj2idx, set_name, daset):
    # (Keep the original logic unchanged)
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

def c2c_vanilla(model, optimizer, lr_scheduler, config, train_dataset, val_dataset, test_dataset, scaler):
    # =========================================================================
    # [STEP 1] Initialize Hierarchy Helper
    # =========================================================================
    print("[Train] Initializing Hierarchy Helper...")
    helper = HierarchyHelper(train_dataset, root_dir='codes/dataset')
    
    # Inject text information into the model
    coarse_verbs, coarse_objs = helper.get_coarse_info()
    model.set_hierarchy_prompts(coarse_verbs, coarse_objs, train_dataset.pairs)
    
    # Get mapping tables (GPU Tensor)
    v2cv, o2co, p2v, p2o = helper.get_mappings()
    v2cv, o2co, p2v, p2o = v2cv.cuda(), o2co.cuda(), p2v.cuda(), p2o.cuda()
    
    print("[Train] Hierarchy mappings ready.")

    # =========================================================================
    # [STEP 2] Instantiate the Encapsulated Loss
    # =========================================================================
    # Use the H2EMTotalLoss we defined in loss.py
    # Coefficients: Beta1(DA)=1.0, Beta2(TE)=0.1, Beta3(Prim)=0.5
    criterion = H2EMTotalLoss(temperature=0.1, beta1=1.0, beta2=0.1, beta3=0.5).cuda()
    
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
    log_training = open(os.path.join(config.save_path, 'log.txt'), 'w')

    for i in range(config.epoch_start, config.epochs):
        progress_bar = tqdm.tqdm(total=len(train_dataloader), desc="epoch % 3d" % (i + 1))
        
        # Loss recorders
        loss_meters = {"Total": [], "Prim": [], "DA": [], "TE": []}

        temp_lr = optimizer.param_groups[-1]['lr']
        print(f'Current_lr:{temp_lr}')
        
        for bid, batch in enumerate(train_dataloader):
            batch_img = batch[0].cuda()
            batch_verb = batch[1].cuda()
            batch_obj = batch[2].cuda()
            batch_target = batch[3].cuda()

            with torch.cuda.amp.autocast(enabled=True):
                # 1. Forward propagation
                out = model(batch_img)
                
                # 2. Calculate loss (call the encapsulated logic in loss.py)
                # Pass all required mapping tables: p2v, p2o (for DA and Comp constraints), v2cv, o2co (for Prim constraints)
                loss, loss_dict = criterion(out, batch_verb, batch_obj, batch_target, 
                                          p2v, p2o, v2cv, o2co)
                
                loss = loss / config.gradient_accumulation_steps

            # 3. Backward propagation
            scaler.scale(loss).backward()

            if ((bid + 1) % config.gradient_accumulation_steps == 0) or (bid + 1 == len(train_dataloader)):
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # 4. Record loss values
            for k, v in loss_dict.items():
                loss_meters[k].append(v)

            progress_bar.set_postfix({
                "Total": f"{np.mean(loss_meters['Total'][-50:]):.3f}",
                "DA":    f"{loss_dict['DA']:.3f}",
                "Prim":  f"{loss_dict['Prim']:.3f}"
            })
            progress_bar.update()

        lr_scheduler.step()
        progress_bar.close()
        
        # Epoch Summary
        avgs = {k: np.mean(v) for k, v in loss_meters.items()}
        print(f"Epoch {i+1} | Total: {avgs['Total']:.4f} | Prim: {avgs['Prim']:.4f} | DA: {avgs['DA']:.4f} | TE: {avgs['TE']:.4f}")
        
        log_training.write(f"\nEpoch {i+1}\n")
        log_training.write(str(avgs) + "\n")

        # Save Checkpoint
        if (i + 1) % config.save_every_n == 0:
            save_checkpoint({
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict(),
                'scaler': scaler.state_dict(),
            }, config.save_path, i)

        # Model Evaluation
        if i % config.eval_every_n == 0 or i + 1 == config.epochs or i >= config.val_epochs_ts:
            print("Evaluating val dataset:")
            val_loss_avg, val_result = evaluate(model, val_dataset, config)
            
            result_str = ""
            key_set = ["attr_acc", "obj_acc", "ub_seen", "ub_unseen", "ub_all", "best_seen", "best_unseen", "best_hm", "AUC"]
            for key in val_result:
                if key in key_set:
                    result_str += f"{key} {round(val_result[key], 4)}| "
            
            print(f"Val: {result_str}")
            log_training.write(f"Val Results: {result_str}\n")
            
            # Save Best Model
            current_metric = val_result[config.best_model_metric] if config.best_model_metric != "best_loss" else -val_loss_avg.item()
            if config.best_model_metric == "best_loss":
                is_best = val_loss_avg < best_loss
                if is_best:
                    best_loss = val_loss_avg
            else:
                is_best = current_metric > best_metric
                if is_best:
                    best_metric = current_metric
            
            if is_best:
                print('Found best model!')
                torch.save(model.state_dict(), os.path.join(config.save_path, "best.pt"))
                print("Evaluating test dataset:")
                _, test_result = evaluate(model, test_dataset, config)
                # ... log test result ...

        log_training.flush()

    # Final Evaluation
    if i + 1 == config.epochs:
        print("Final Evaluation")
        model.load_state_dict(torch.load(os.path.join(config.save_path, "best.pt")))
        evaluate(model, test_dataset, config)