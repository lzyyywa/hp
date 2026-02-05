import torch
import torch.nn as nn
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from models.vlm_models.text_learner import get_text_learner
import torch.nn.functional as F
from einops import rearrange
import math

# [HyCoCLIP Requirement]
try:
    from utils.hyperbolic_ops import LorentzMath
except ImportError:
    print("[Error] Could not import LorentzMath! Please check your path.")
    exit(1)

_tokenizer = _Tokenizer()



class MLP(nn.Module):
    def __init__(self, inp_dim, out_dim, num_layers=1, relu=True, bias=True, dropout=False, norm=False, layers=[]):
        super(MLP, self).__init__()
        mod = []
        incoming = inp_dim
        for layer_ind in range(num_layers - 1):
            if len(layers) == 0:
                outgoing = incoming
            else:
                outgoing = layers[layer_ind]
            
            mod.append(nn.Linear(incoming, outgoing, bias=bias))
            
            incoming = outgoing
            if norm:
                mod.append(nn.LayerNorm(outgoing))
            mod.append(nn.ReLU(inplace=True))
            if dropout:
                mod.append(nn.Dropout(p=0.5))
    
        mod.append(nn.Linear(incoming, out_dim, bias=bias))
        
        if relu:
            mod.append(nn.ReLU(inplace=True))
        self.mod = nn.Sequential(*mod)

    def forward(self, x):
        return self.mod(x)

class MLP_ST(nn.Module):
    def __init__(self, inp_dim, out_dim, num_layers=1, relu=True, bias=True, dropout=False, norm=False, layers=[]):
        super(MLP_ST, self).__init__()
        mod = []
        incoming = inp_dim
        for layer_ind in range(num_layers - 1):
            if len(layers) == 0:
                outgoing = incoming
            else:
                outgoing = layers[layer_ind]
            
            mod.append(nn.Conv1d(incoming, outgoing, kernel_size=3, bias=bias, padding=1))
            
            incoming = outgoing
            if norm:
                mod.append(nn.LayerNorm(outgoing))
            mod.append(nn.ReLU(inplace=True))
            if dropout:
                mod.append(nn.Dropout(p=0.5))
        
        mod.append(nn.Conv1d(incoming, out_dim, kernel_size=3, bias=bias, padding=1))
        
        if relu:
            mod.append(nn.ReLU(inplace=True))
        self.mod = nn.Sequential(*mod)

    def forward(self, x):
        for o in self.mod:
            if isinstance(o, nn.LayerNorm):
                x = x.transpose(1, 2)
                x = o(x)
                x = x.transpose(1, 2)
            else:
                x = o(x)
        return x

class TextEncoder(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        full_mask = self.transformer.resblocks[0].attn_mask.clone()
        self.register_buffer('causal_mask', full_mask)
        self.dtype = clip_model.dtype

    def forward(self, x, tokenized_prompts):
        x = x.permute(1, 0, 2)
        L = x.shape[0] 
        current_mask = self.causal_mask[:L, :L]
        for block in self.transformer.resblocks:
            block.attn_mask = current_mask
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

class VideoEncoder(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        from models.vlm_models.AIM import get_aim
        self.visual = get_aim(cfg)
        self.clip_proj = clip_model.visual.proj
        self.num_frames = cfg.num_frames

    def forward(self, x):
        out = self.visual(x)
        if self.clip_proj is not None:
            out = out @ self.clip_proj
        out = rearrange(out, '(b t) d -> b d t', t=self.num_frames)
        return out

class CustomCLIP(nn.Module):
    def __init__(self, cfg, train_dataset, clip_model):
        super().__init__()
        self.verb_prompt_learner = get_text_learner(cfg, train_dataset, clip_model, 'verb')
        self.verb_tokenized_prompts = self.verb_prompt_learner.token_ids
        self.obj_prompt_learner = get_text_learner(cfg, train_dataset, clip_model, 'object')
        self.obj_tokenized_prompts = self.obj_prompt_learner.token_ids

        self.text_encoder = TextEncoder(cfg, clip_model)
        self.video_encoder = VideoEncoder(cfg, clip_model)

        self.c_param = nn.Parameter(torch.tensor(1.0).log())
        self._curv_minmax = {
            "max": math.log(1.0 * 10), 
            "min": math.log(1.0 / 10), 
        }

        init_scale_val = 100.0
        print(f"[CustomCLIP] Initializing Logit Scale to: {init_scale_val} (log space)")
        self.logit_scale = nn.Parameter(torch.tensor(math.log(init_scale_val)))

        init_alpha = 0.5
        print(f"[CustomCLIP] Initializing visual/textual alpha to {init_alpha} (Log space)")
        self.visual_alpha = nn.Parameter(torch.tensor(init_alpha).log())
        self.textual_alpha = nn.Parameter(torch.tensor(init_alpha).log())

        try:
            fc_emb = cfg.fc_emb.split(',')
        except:
            fc_emb = [cfg.fc_emb]
        
        layers = []
        if isinstance(fc_emb, list) and len(fc_emb) == 1 and fc_emb[0] == '':
             layers = []
        else:
             layers = [int(a) for a in fc_emb if a != '']

        # ---------------------------------------------------------------------
        # [修正 3 & 4] Bias = False
        # ---------------------------------------------------------------------
        # 实例化 MLP 时显式传入 bias=False
        self.c2c_OE1 = MLP(cfg.feat_dim, int(cfg.emb_dim), relu=cfg.relu, num_layers=cfg.nlayers,
                           dropout=False, norm=True, layers=layers, bias=False)
        self.c2c_VE1 = MLP_ST(cfg.feat_dim, int(cfg.emb_dim), relu=cfg.relu, num_layers=cfg.nlayers,
                              dropout=False, norm=True, layers=layers, bias=False)

        # 实例化 Linear 时显式传入 bias=False
        self.c2c_text_v = nn.Linear(cfg.feat_dim, cfg.emb_dim, bias=False)
        self.c2c_text_o = nn.Linear(cfg.feat_dim, cfg.emb_dim, bias=False)

        self.cached_hierarchy = {} 
        self.clip_tokenize = clip.tokenize
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.model_device = next(self.parameters()).device
        
        self._init_hyperbolic_modules()

    def _init_hyperbolic_modules(self):
        print("[CustomCLIP] Applying Identity Initialization...")
        for m in [self.c2c_OE1, self.c2c_VE1, self.c2c_text_v, self.c2c_text_o]:
            if isinstance(m, nn.Module):
                for sub_m in m.modules():
                    if isinstance(sub_m, nn.Linear):
                        if sub_m.in_features == sub_m.out_features:
                            nn.init.eye_(sub_m.weight)
                            sub_m.weight.data.add_(torch.randn_like(sub_m.weight) * 0.001)
                        else:
                            nn.init.orthogonal_(sub_m.weight, gain=1.0)
                        if sub_m.bias is not None:
                            nn.init.constant_(sub_m.bias, 0.0)
                    elif isinstance(sub_m, nn.Conv1d):
                         nn.init.dirac_(sub_m.weight) 
                         if sub_m.bias is not None:
                            nn.init.constant_(sub_m.bias, 0.0)
                    elif isinstance(sub_m, nn.LayerNorm):
                        nn.init.constant_(sub_m.bias, 0.0)
                        nn.init.constant_(sub_m.weight, 1.0)      

    def _encode_plain_text(self, tokenized_prompts):
        tokenized_prompts = tokenized_prompts.to(self.model_device)
        x = self.token_embedding(tokenized_prompts).type(self.text_encoder.dtype)
        x = x + self.positional_embedding.type(self.text_encoder.dtype)
        return self.text_encoder(x, tokenized_prompts)

    def set_hierarchy_prompts(self, coarse_verbs, coarse_objs, pairs):
        print(f"[CustomCLIP] Setting up hierarchy prompts...")
        self.coarse_verb_tokens = self.clip_tokenize([f"a photo of {v} something" for v in coarse_verbs])
        self.coarse_obj_tokens = self.clip_tokenize([f"a photo of something {o}" for o in coarse_objs])
        pair_texts = [f"a photo of {v} {o}" for v, o in pairs]
        self.comp_tokens = self.clip_tokenize(pair_texts)
        print(f"[CustomCLIP] Ready to cache.")

    def _ensure_hierarchy_cached(self, device):
        if "coarse_verb_backbone" in self.cached_hierarchy:
            return 
        print(f"[CustomCLIP] Caching BACKBONE outputs (Frozen) on {device}...")
        self.model_device = device
        with torch.no_grad():
            cv_backbone = self._encode_plain_text(self.coarse_verb_tokens)
            self.cached_hierarchy["coarse_verb_backbone"] = cv_backbone.float()
            co_backbone = self._encode_plain_text(self.coarse_obj_tokens)
            self.cached_hierarchy["coarse_obj_backbone"] = co_backbone.float()
            batch_size = 500
            comp_embs = []
            total = self.comp_tokens.shape[0]
            for i in range(0, total, batch_size):
                batch_tokens = self.comp_tokens[i : i + batch_size]
                emb = self._encode_plain_text(batch_tokens)
                comp_embs.append(emb.cpu())
            comp_emb_all = torch.cat(comp_embs, dim=0).to(device)
            self.cached_hierarchy["comp_backbone"] = comp_emb_all.float()
        print(f"[CustomCLIP] Backbone Cache finished.")

    def forward(self, video, pairs=None):
        device = video.device
        
        if self.coarse_verb_tokens is not None and "coarse_verb_backbone" not in self.cached_hierarchy:
            self._ensure_hierarchy_cached(device)

        verb_prompts = self.verb_prompt_learner()
        verb_text_features = self.text_encoder(verb_prompts, self.verb_tokenized_prompts)
        obj_prompts = self.obj_prompt_learner()
        obj_text_features = self.text_encoder(obj_prompts, self.obj_tokenized_prompts)

        video_features = self.video_encoder(video)

        with torch.cuda.amp.autocast(enabled=False):
            # -----------------------------------------------------------------
            # [修正 1] 参数钳制 (无 clip_norm)
            # -----------------------------------------------------------------
            self.c_param.data = torch.clamp(self.c_param.data, **self._curv_minmax)
            current_c = self.c_param.exp() 
            
            # 限制 Alpha 不超过 0.0 (Scale <= 1.0)
            self.visual_alpha.data = torch.clamp(self.visual_alpha.data, max=0.0) 
            self.textual_alpha.data = torch.clamp(self.textual_alpha.data, max=0.0)
            
            verb_text_features = verb_text_features.float()
            obj_text_features = obj_text_features.float()
            video_features = video_features.float()

            verb_text_features = self.c2c_text_v(verb_text_features)
            obj_text_features = self.c2c_text_o(obj_text_features)
            o_feat = self.c2c_OE1(video_features.mean(dim=-1))
            v_feat_t = self.c2c_VE1(video_features)
            v_feat = v_feat_t.mean(dim=-1)

            cv_backbone = self.cached_hierarchy["coarse_verb_backbone"]
            co_backbone = self.cached_hierarchy["coarse_obj_backbone"]
            comp_backbone = self.cached_hierarchy["comp_backbone"]

            cv_euc = self.c2c_text_v(cv_backbone)
            co_euc = self.c2c_text_o(co_backbone)
            comp_v_euc = self.c2c_text_v(comp_backbone)
            comp_o_euc = self.c2c_text_o(comp_backbone)

            # -----------------------------------------------------------------
            # [修正 1] 软性缩放 (Soft Scaling)
            # -----------------------------------------------------------------
            # 完全移除 clip_norm，只使用 Alpha 的指数进行全局缩放
            v_scale = self.visual_alpha.exp()
            t_scale = self.textual_alpha.exp()
            
            verb_text_features = verb_text_features * t_scale
            obj_text_features = obj_text_features * t_scale
            o_feat = o_feat * v_scale
            v_feat = v_feat * v_scale

            cv_euc = cv_euc * t_scale
            co_euc = co_euc * t_scale
            comp_v_euc = comp_v_euc * t_scale
            comp_o_euc = comp_o_euc * t_scale

            # Map to Hyperbolic
            verb_text_hyp = LorentzMath.exp_map_0(verb_text_features, c=current_c)
            obj_text_hyp = LorentzMath.exp_map_0(obj_text_features, c=current_c)
            o_feat_hyp = LorentzMath.exp_map_0(o_feat, c=current_c)
            v_feat_hyp = LorentzMath.exp_map_0(v_feat, c=current_c)

            coarse_verb_hyp = LorentzMath.exp_map_0(cv_euc, c=current_c)
            coarse_obj_hyp  = LorentzMath.exp_map_0(co_euc, c=current_c)
            comp_hyp_v = LorentzMath.exp_map_0(comp_v_euc, c=current_c)
            comp_hyp_o = LorentzMath.exp_map_0(comp_o_euc, c=current_c)

            dist_v = LorentzMath.hyp_distance(v_feat_hyp.unsqueeze(1), verb_text_hyp.unsqueeze(0), c=current_c)
            dist_o = LorentzMath.hyp_distance(o_feat_hyp.unsqueeze(1), obj_text_hyp.unsqueeze(0), c=current_c)
            
            # [HyCoCLIP Style] Clamp Logit Scale max to ~100 (ln 4.6)
            with torch.no_grad():
                self.logit_scale.clamp_(max=4.6052)
            logit_scale = self.logit_scale.exp()
            
            verb_logits = -dist_v * logit_scale
            obj_logits = -dist_o * logit_scale

            if self.training:
                return {
                    "verb_logits": verb_logits,
                    "obj_logits": obj_logits,
                    "v_feat_hyp": v_feat_hyp,
                    "o_feat_hyp": o_feat_hyp,
                    "verb_text_hyp": verb_text_hyp,
                    "obj_text_hyp": obj_text_hyp,
                    "coarse_verb_hyp": coarse_verb_hyp,
                    "coarse_obj_hyp": coarse_obj_hyp,
                    "comp_hyp_v": comp_hyp_v,  
                    "comp_hyp_o": comp_hyp_o,
                    "logit_scale": logit_scale,
                    "curvature": current_c  
                }
            else:
                if pairs is not None:
                    verb_idx, obj_idx = pairs[:, 0], pairs[:, 1]
                    combined_logits = verb_logits[:, verb_idx] + obj_logits[:, obj_idx]
                    return combined_logits 
                else:
                    return verb_logits, obj_logits

def load_clip_to_cpu(cfg):
    backbone_name = cfg.backbone
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(state_dict or model.state_dict())
    return model

def build_model(train_dataset, cfg):
    print(f"Loading CLIP (backbone: {cfg.backbone})")
    clip_model = load_clip_to_cpu(cfg)
    clip_model.float()
    print("Building custom CLIP (Hyperbolic Version - HyCoCLIP Aligned)")
    model = CustomCLIP(cfg, train_dataset, clip_model)
    for name, param in model.named_parameters():
        param.requires_grad_(False)
        if "prompt_learner" in name:
            if cfg.learn_input_method != 'zero':
                if cfg.learn_input_method == 'coop':
                    if 'prompt_vectors' in name:
                        param.requires_grad_(True)
                        print(f'{name}: {param.requires_grad}')
                elif cfg.learn_input_method == 'csp':
                    if 'obj_embedding' in name or 'verb_embedding' in name or 'comp_embedding' in name:
                        param.requires_grad_(True)
                        print(f'{name}: {param.requires_grad}')
                elif cfg.learn_input_method == 'spm':
                    if 'prompt_vectors' in name or 'obj_embedding' in name or 'verb_embedding' in name or 'comp_embedding' in name:
                        param.requires_grad_(True)
                        print(f'{name}: {param.requires_grad}')
                else:
                    raise NotImplementedError
        elif 'video_encoder' in name:
            if 'temporal_embedding' in name or 'ln_post' in name or 'Adapter' in name or 'clip_proj' in name:
                param.requires_grad = True
                print(f'{name}: {param.requires_grad}')
        elif 'c2c' in name:
            param.requires_grad = True
            print(f'{name}: {param.requires_grad}')
        # [补充] 确保双曲参数可学习
        elif 'c_param' in name:
            param.requires_grad = True
        elif 'visual_alpha' in name or 'textual_alpha' in name:
            param.requires_grad = True
    return model