import torch
import torch.nn as nn
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from models.vlm_models.text_learner import get_text_learner
import torch.nn.functional as F
from einops import rearrange

# Import the upgraded stable hyperbolic ops
try:
    from utils.hyperbolic_ops import LorentzMath
except ImportError:
    print("[Error] Could not import LorentzMath!")

_tokenizer = _Tokenizer()

# ==================================================================================
# Helper: Feature Clipping (HyCoCLIP Practice)
# ==================================================================================
def clip_norm(x, max_norm=1.0):
    """
    Clamps the norm of vectors in Euclidean space before projecting to Hyperbolic space.
    Prevents numerical instability (NaN) caused by extremely large norms in exp_map.
    Ref: HyCoCLIP uses max_norm around 15.0.
    """
    norm = x.norm(p=2, dim=-1, keepdim=True)
    # If norm > max_norm, scale it down. Otherwise, keep it as is.
    target_norm = torch.clamp(norm, max=max_norm)
    return x * (target_norm / (norm + 1e-6))

# ==================================================================================
# 1. Basic Components
# ==================================================================================

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
        # Dynamic Mask Fix
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


# ==================================================================================
# 2. CustomCLIP (Optimized for Hyperbolic Hierarchy & Stability)
# ==================================================================================

class CustomCLIP(nn.Module):
    def __init__(self, cfg, train_dataset, clip_model):
        super().__init__()
        self.verb_prompt_learner = get_text_learner(cfg, train_dataset, clip_model, 'verb')
        self.verb_tokenized_prompts = self.verb_prompt_learner.token_ids
        self.obj_prompt_learner = get_text_learner(cfg, train_dataset, clip_model, 'object')
        self.obj_tokenized_prompts = self.obj_prompt_learner.token_ids

        self.text_encoder = TextEncoder(cfg, clip_model)
        self.video_encoder = VideoEncoder(cfg, clip_model)
        
        # [NEW] Learnable Curvature Parameter
        # Initialized to 1.0. Wrapped in nn.Parameter to be optimized.
        self.c_param = nn.Parameter(torch.tensor(1.0))

        # [OPTIMIZATION] Use CLIP's learnable logit scale
        self.logit_scale = clip_model.logit_scale

        try:
            fc_emb = cfg.fc_emb.split(',')
        except:
            fc_emb = [cfg.fc_emb]
        layers = [int(a) for a in fc_emb]

        # Projection Layers
        self.c2c_OE1 = MLP(cfg.feat_dim, int(cfg.emb_dim), relu=cfg.relu, num_layers=cfg.nlayers,
                           dropout=False, norm=True, layers=layers)
        self.c2c_VE1 = MLP_ST(cfg.feat_dim, int(cfg.emb_dim), relu=cfg.relu, num_layers=cfg.nlayers,
                              dropout=False, norm=True, layers=layers)

        self.c2c_text_v = nn.Linear(cfg.feat_dim, cfg.emb_dim, bias=True)
        self.c2c_text_o = nn.Linear(cfg.feat_dim, cfg.emb_dim, bias=True)

        self.cached_hierarchy = {} 
        self.clip_tokenize = clip.tokenize
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.model_device = next(self.parameters()).device
        
        # Apply Hyperbolic Initialization
        self._init_hyperbolic_modules()

    def _init_hyperbolic_modules(self):
        """
        [HyCoCLIP Strategy]
        Apply Orthogonal Initialization to preserve CLIP's pre-trained geometry
        as much as possible in the early stages.
        Replaces the destructive Gaussian noise initialization.
        """
        print("[CustomCLIP] Applying SOTA Orthogonal Initialization (Preserving Geometry)...")
        
        # Iterate through all trainable hyperbolic projection layers
        for m in [self.c2c_OE1, self.c2c_VE1, self.c2c_text_v, self.c2c_text_o]:
            if isinstance(m, nn.Module):
                for sub_m in m.modules():
                    if isinstance(sub_m, nn.Linear):
                        nn.init.orthogonal_(sub_m.weight, gain=1.0)
                        if sub_m.bias is not None:
                            nn.init.constant_(sub_m.bias, 0.0)
                    elif isinstance(sub_m, nn.Conv1d):
                        nn.init.orthogonal_(sub_m.weight, gain=1.0)
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
        """
        [CACHING STRATEGY UPDATE - CRITICAL]
        We now cache the EUCLIDEAN embeddings (Static), NOT the Hyperbolic ones.
        This allows us to re-project them using the updated projection layers 
        and the updated curvature 'c' in every forward pass.
        """
        if "coarse_verb_euc" in self.cached_hierarchy:
            return 

        print(f"[CustomCLIP] Computing and caching EUCLIDEAN hierarchy features on {device}...")
        self.model_device = device
        
        with torch.no_grad():
            # 1. Coarse Verbs (Compute EUCLIDEAN)
            cv_emb = self._encode_plain_text(self.coarse_verb_tokens)
            cv_emb = self.c2c_text_v(cv_emb) # Linear projection
            # Store Euclidean, normalized (clipped) for stability, but NO exp_map here
            self.cached_hierarchy["coarse_verb_euc"] = clip_norm(cv_emb)

            # 2. Coarse Objects (Compute EUCLIDEAN)
            co_emb = self._encode_plain_text(self.coarse_obj_tokens)
            co_emb = self.c2c_text_o(co_emb) # Linear projection
            self.cached_hierarchy["coarse_obj_euc"] = clip_norm(co_emb)

            # 3. Compositions (Compute EUCLIDEAN)
            batch_size = 500
            comp_embs = []
            total = self.comp_tokens.shape[0]
            for i in range(0, total, batch_size):
                batch_tokens = self.comp_tokens[i : i + batch_size]
                emb = self._encode_plain_text(batch_tokens)
                comp_embs.append(emb.cpu())
            
            comp_emb_all = torch.cat(comp_embs, dim=0).to(device)
            
            # Project Compositions to Verb Space (Euclidean)
            comp_v_euc = self.c2c_text_v(comp_emb_all)
            self.cached_hierarchy["comp_euc_v"] = clip_norm(comp_v_euc)

            # Project Compositions to Object Space (Euclidean)
            comp_o_euc = self.c2c_text_o(comp_emb_all)
            self.cached_hierarchy["comp_euc_o"] = clip_norm(comp_o_euc)
            
        print(f"[CustomCLIP] Euclidean Cache finished.")

    def forward(self, video, pairs=None):
        device = video.device
        
        # [NEW] Dynamic Curvature Calculation
        # Ensure c > 0 using softplus and minimum bound
        current_c = F.softplus(self.c_param) + 1e-5
        
        # Check for cached EUCLIDEAN features
        if self.coarse_verb_tokens is not None and "coarse_verb_euc" not in self.cached_hierarchy:
            self._ensure_hierarchy_cached(device)

        # ----------------------------------------------------------------------
        # 1. Fine-grained Feature Extraction (Euclidean) -> Projection -> Hyperbolic
        # ----------------------------------------------------------------------
        
        # Verbs
        verb_prompts = self.verb_prompt_learner()
        verb_text_features = self.text_encoder(verb_prompts, self.verb_tokenized_prompts)
        verb_text_features = self.c2c_text_v(verb_text_features)
        verb_text_hyp = LorentzMath.exp_map_0(clip_norm(verb_text_features), c=current_c)

        # Objects
        obj_prompts = self.obj_prompt_learner()
        obj_text_features = self.text_encoder(obj_prompts, self.obj_tokenized_prompts)
        obj_text_features = self.c2c_text_o(obj_text_features)
        obj_text_hyp = LorentzMath.exp_map_0(clip_norm(obj_text_features), c=current_c)

        # Video
        video_features = self.video_encoder(video)
        o_feat = self.c2c_OE1(video_features.mean(dim=-1))
        v_feat_t = self.c2c_VE1(video_features)
        v_feat = v_feat_t.mean(dim=-1)

        # Project Video to Hyperbolic (using current_c)
        o_feat_hyp = LorentzMath.exp_map_0(clip_norm(o_feat), c=current_c)
        v_feat_hyp = LorentzMath.exp_map_0(clip_norm(v_feat), c=current_c)

        # ----------------------------------------------------------------------
        # 2. Hierarchy Features (LIVE PROJECTION using current_c)
        # ----------------------------------------------------------------------
        # Retrieve Euclidean Cache -> Apply ExpMap(c)
        
        coarse_verb_hyp = LorentzMath.exp_map_0(self.cached_hierarchy["coarse_verb_euc"], c=current_c)
        coarse_obj_hyp  = LorentzMath.exp_map_0(self.cached_hierarchy["coarse_obj_euc"],  c=current_c)
        
        comp_hyp_v      = LorentzMath.exp_map_0(self.cached_hierarchy["comp_euc_v"],      c=current_c)
        comp_hyp_o      = LorentzMath.exp_map_0(self.cached_hierarchy["comp_euc_o"],      c=current_c)

        # ----------------------------------------------------------------------
        # 3. Compute Logits
        # ----------------------------------------------------------------------
        dist_v = LorentzMath.hyp_distance(v_feat_hyp.unsqueeze(1), verb_text_hyp.unsqueeze(0), c=current_c)
        dist_o = LorentzMath.hyp_distance(o_feat_hyp.unsqueeze(1), obj_text_hyp.unsqueeze(0), c=current_c)
        
        logit_scale = self.logit_scale.exp()
        logit_scale = torch.clamp(logit_scale, max=100.0)
        
        verb_logits = -dist_v 
        obj_logits = -dist_o

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
                "curvature": current_c  # [PASS] Pass learnable c to Loss
            }
        else:
            if pairs is not None:
                verb_idx, obj_idx = pairs[:, 0], pairs[:, 1]
                combined_logits = verb_logits[:, verb_idx] + obj_logits[:, obj_idx]
                return combined_logits * logit_scale
            else:
                return verb_logits * logit_scale, obj_logits * logit_scale

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
    print("Building custom CLIP (Hyperbolic Version)")
    model = CustomCLIP(cfg, train_dataset, clip_model)
    for name, param in model.named_parameters():
        param.requires_grad_(False)
        if "prompt_learner" in name and cfg.learn_input_method != 'zero':
            param.requires_grad_(True)
        elif 'video_encoder' in name:
            if 'temporal_embedding' in name or 'ln_post' in name or 'Adapter' in name or 'clip_proj' in name:
                param.requires_grad = True
        elif 'c2c' in name:
            param.requires_grad = True
        # Ensure curvature is trainable (it is by default as a Parameter, but being safe)
        elif 'c_param' in name:
            param.requires_grad = True
    return model