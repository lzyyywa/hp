import torch

def get_optimizer_vm(cfg, model):
    comp_param = []
    video_en_param = []
    for name, param in model.named_parameters():
        if 'video_encoder' in name:
            video_en_param.append(param)
        else:
            comp_param.append(param)
    optimizer = torch.optim.Adam([
        {'params': comp_param, 'lr': cfg.com_lr, 'weight_decay': cfg.com_wd},
        {'params': video_en_param, 'lr': cfg.ve_lr, 'weight_decay': cfg.ve_wd}],
        lr=cfg.ve_lr, eps=1e-8, weight_decay=cfg.ve_wd)

    return optimizer

def get_optimizer_vlm(cfg, model):
    # ============================================================
    # 1. Prepare Parameter Groups
    # ============================================================
    vision_no_wd = []
    vision_with_wd = []
    prompt_param = []
    c2c_with_wd = []
    c2c_no_wd = []
    hyperbolic_params = [] 
    
    processed_params = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        pid = id(param)
        if pid in processed_params:
            continue
        processed_params.add(pid)

        # [CRITICAL] Hyperbolic Parameters MUST have 0 weight decay
        if any(k in name for k in ['visual_alpha', 'textual_alpha', 'c_param', 'logit_scale']):
            hyperbolic_params.append(param)
            
        elif 'video_encoder' in name:
            if any(k in name for k in ['temporal_embedding', 'ln_post', 'bias', 'LayerNorm.weight', 'bn.weight']):
                vision_no_wd.append(param)
            elif 'Adapter' in name or 'clip_proj' in name:
                vision_with_wd.append(param)
            else:
                vision_with_wd.append(param)
        
        elif 'c2c' in name:
            if 'bias' in name or 'LayerNorm' in name or 'ln' in name:
                c2c_no_wd.append(param)
            else:
                c2c_with_wd.append(param)

        elif 'prompt_learner' in name:
            prompt_param.append(param)
        
        else:
            prompt_param.append(param)

    # ============================================================
    # 2. Build Optimizer
    # ============================================================
    # Standard VLM Optimizer Construction
    optimizer = torch.optim.AdamW([
        {'params': prompt_param, 'lr': cfg.text_lr, 'weight_decay': cfg.text_wd},
        {'params': vision_with_wd, 'lr': cfg.visual_lr, 'weight_decay': cfg.visual_wd},
        {'params': vision_no_wd, 'lr': cfg.visual_lr, 'weight_decay': 0.0},
        
        {'params': c2c_with_wd, 'lr': cfg.visual_lr, 'weight_decay': cfg.visual_wd},
        {'params': c2c_no_wd, 'lr': cfg.visual_lr, 'weight_decay': 0.0},
        
        # [Hyperbolic Group] Learning rate follows visual, but WD must be 0
        {'params': hyperbolic_params, 'lr': cfg.visual_lr, 'weight_decay': 0.0}, 
        ],
        betas=(0.9, 0.999), lr=cfg.visual_lr, eps=1e-8,
        weight_decay=cfg.visual_wd)
        
    return optimizer

def get_optimizer(cfg, model):
    if cfg.framework == 'vm':
        return get_optimizer_vm(cfg, model)
    elif cfg.framework == 'vlm':
        return get_optimizer_vlm(cfg, model)