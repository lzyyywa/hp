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
    
    # [HyCoCLIP Alignment] Hyperbolic Scalars
    # These MUST have 0 weight decay to prevent collapsing to 0.
    hyperbolic_params = [] 

    # Track processed parameters to avoid double counting
    processed_params = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Unique ID for the parameter
        pid = id(param)
        if pid in processed_params:
            continue
        processed_params.add(pid)

        # --------------------------------------------------------
        # Group 1: Hyperbolic Scalars (Highest Priority)
        # Catch: visual_alpha, textual_alpha, c_param, logit_scale
        # --------------------------------------------------------
        if any(k in name for k in ['visual_alpha', 'textual_alpha', 'c_param', 'logit_scale']):
            hyperbolic_params.append(param)
            
        # --------------------------------------------------------
        # Group 2: Vision Encoder
        # --------------------------------------------------------
        elif 'video_encoder' in name:
            if any(k in name for k in ['temporal_embedding', 'ln_post', 'bias', 'LayerNorm.weight', 'bn.weight']):
                vision_no_wd.append(param)
            elif 'Adapter' in name or 'clip_proj' in name:
                vision_with_wd.append(param)
            else:
                # Default for other vision params (if any remain)
                vision_with_wd.append(param)
        
        # --------------------------------------------------------
        # Group 3: C2C Modules (Projection Layers)
        # --------------------------------------------------------
        elif 'c2c' in name:
            # Standard Practice: No Weight Decay for Bias and LayerNorm
            if 'bias' in name or 'LayerNorm' in name or 'ln' in name:
                c2c_no_wd.append(param)
            else:
                c2c_with_wd.append(param)

        # --------------------------------------------------------
        # Group 4: Text / Prompt Learner
        # --------------------------------------------------------
        elif 'prompt_learner' in name:
             # Exclude token_embedding if necessary (based on your original logic)
            if 'token_embedding' in name:
                # Often kept frozen or handled separately, but if requires_grad is True:
                prompt_param.append(param)
            else:
                prompt_param.append(param)
        
        # --------------------------------------------------------
        # Group 5: Others (Fallback)
        # --------------------------------------------------------
        else:
            # Any other learnable parameters default to text settings or specific handling
            # Assuming they belong to text/prompt side if not vision/c2c
            prompt_param.append(param)

    # ============================================================
    # 2. Build Optimizer
    # ============================================================
    if cfg.text_encoding_manner == 'composition':
        # Composition Manner specific logic
        # (Assuming 'dfsp' was the identifier here, but logic above handles 'c2c' generically)
        # Re-using the groups collected above
        
        optimizer = torch.optim.AdamW([
            {'params': prompt_param, 'lr': cfg.text_lr, 'weight_decay': cfg.text_wd},
            {'params': vision_with_wd, 'lr': cfg.visual_lr, 'weight_decay': cfg.visual_wd},
            {'params': vision_no_wd, 'lr': cfg.visual_lr, 'weight_decay': 0.0},
            # C2C Params (if they exist in this mode)
            {'params': c2c_with_wd, 'lr': cfg.visual_lr, 'weight_decay': cfg.visual_wd},
            {'params': c2c_no_wd, 'lr': cfg.visual_lr, 'weight_decay': 0.0},
            # Hyperbolic Params
            {'params': hyperbolic_params, 'lr': cfg.visual_lr, 'weight_decay': 0.0},
            ],
            betas=(0.9, 0.999), lr=cfg.visual_lr, eps=1e-8,
            weight_decay=cfg.visual_wd)
            
    elif cfg.text_encoding_manner == 'component':
        optimizer = torch.optim.AdamW([
            {'params': prompt_param, 'lr': cfg.text_lr, 'weight_decay': cfg.text_wd},
            {'params': vision_with_wd, 'lr': cfg.visual_lr, 'weight_decay': cfg.visual_wd},
            {'params': vision_no_wd, 'lr': cfg.visual_lr, 'weight_decay': 0.0},
            
            # C2C Layers - Weights get WD, Biases get 0 WD
            {'params': c2c_with_wd, 'lr': cfg.visual_lr, 'weight_decay': cfg.visual_wd},
            {'params': c2c_no_wd, 'lr': cfg.visual_lr, 'weight_decay': 0.0},
            
            # [HyCoCLIP Group] Scalars with visual_lr and NO weight decay
            {'params': hyperbolic_params, 'lr': cfg.visual_lr, 'weight_decay': 0.0}, 
            ],
            betas=(0.9, 0.999), lr=cfg.visual_lr, eps=1e-8,
            weight_decay=cfg.visual_wd)
    else:
        raise NotImplementedError
        
    return optimizer

def get_optimizer(cfg, model):
    if cfg.framework == 'vm':
        return get_optimizer_vm(cfg, model)
    elif cfg.framework == 'vlm':
        return get_optimizer_vlm(cfg, model)