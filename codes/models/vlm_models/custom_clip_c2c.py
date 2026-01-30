import torch
import torch.nn as nn
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from models.vlm_models.text_learner import get_text_learner
import torch.nn.functional as F
from einops import rearrange

# [NEW] Import hyperbolic math library
try:
    from utils.hyperbolic_ops import LorentzMath
except ImportError:
    print("[Error] Could not import LorentzMath! Please create codes/utils/hyperbolic_ops.py first.")

_tokenizer = _Tokenizer()

# ==================================================================================
# 1. Basic Components (MLP, MLP_ST, VideoEncoder remain unchanged)
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

# ==================================================================================
# [Key Modification] TextEncoder: Support dynamic lengths (10 or 77)
# ==================================================================================
class TextEncoder(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        
        # [Modification 1] No longer permanently truncate mask, save full mask for later use
        # full_mask here is typically 77x77
        full_mask = self.transformer.resblocks[0].attn_mask.clone()
        self.register_buffer('causal_mask', full_mask)
        
        self.dtype = clip_model.dtype

    def forward(self, x, tokenized_prompts):
        # x shape: [batch, n_ctx, d_model]
        x = x.permute(1, 0, 2) # [n_ctx, batch, d_model]
        
        # [Modification 2] Dynamically truncate mask
        # If input length is 10, cut to 10x10; if 77, cut to 77x77
        L = x.shape[0] 
        current_mask = self.causal_mask[:L, :L]
        
        # Temporarily assign correct mask to all blocks
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
# 2. CustomCLIP (Hyperbolic Core Logic)
# ==================================================================================
# -------------------------------------------------------------
# 请将下面的 CustomCLIP 类替换掉你文件中的 CustomCLIP 类
# -------------------------------------------------------------

class CustomCLIP(nn.Module):
    def __init__(self, cfg, train_dataset, clip_model):
        super().__init__()
        # Text Prompt Learners
        self.verb_prompt_learner = get_text_learner(cfg, train_dataset, clip_model, 'verb')
        self.verb_tokenized_prompts = self.verb_prompt_learner.token_ids
        self.obj_prompt_learner = get_text_learner(cfg, train_dataset, clip_model, 'object')
        self.obj_tokenized_prompts = self.obj_prompt_learner.token_ids

        # Encoders
        self.text_encoder = TextEncoder(cfg, clip_model)
        self.video_encoder = VideoEncoder(cfg, clip_model)
        self.logit_scale = clip_model.logit_scale

        # Independent Learning Modules
        try:
            fc_emb = cfg.fc_emb.split(',')
        except:
            fc_emb = [cfg.fc_emb]
        layers = [int(a) for a in fc_emb]

        self.c2c_OE1 = MLP(cfg.feat_dim, int(cfg.emb_dim), relu=cfg.relu, num_layers=cfg.nlayers,
                           dropout=False, norm=True, layers=layers)
        self.c2c_VE1 = MLP_ST(cfg.feat_dim, int(cfg.emb_dim), relu=cfg.relu, num_layers=cfg.nlayers,
                              dropout=False, norm=True, layers=layers)

        self.c2c_text_v = nn.Linear(cfg.feat_dim, cfg.emb_dim, bias=True)
        self.c2c_text_o = nn.Linear(cfg.feat_dim, cfg.emb_dim, bias=True)

        # [NEW] Caching Mechanism
        self.cached_hierarchy = {} # Stores pre-computed hyperbolic embeddings
        
        # Save references
        self.clip_tokenize = clip.tokenize
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        
        # Store device reference (will be updated in forward or explicit call)
        self.model_device = next(self.parameters()).device

    def _encode_plain_text(self, tokenized_prompts):
        """Helper to encode plain text (no grad needed usually)"""
        # Ensure input is on correct device
        tokenized_prompts = tokenized_prompts.to(self.model_device)
        
        # 1. Token Embeddings
        x = self.token_embedding(tokenized_prompts).type(self.text_encoder.dtype)
        # 2. Positional Embeddings
        x = x + self.positional_embedding.type(self.text_encoder.dtype)
        # 3. Transformer
        return self.text_encoder(x, tokenized_prompts)

    def set_hierarchy_prompts(self, coarse_verbs, coarse_objs, pairs):
        """
        Setup and PRE-COMPUTE hierarchy embeddings to save memory/time.
        """
        print(f"[CustomCLIP] Setting up and caching hierarchy prompts...")
        
        # Ensure model is on GPU before encoding
        # If this is called before .cuda(), we might need to handle it.
        # Typically train_models.py calls .cuda() then this.
        # But DataParallel wraps it.
        # Safest way: just tokenize here, compute in first forward or explicit call.
        
        self.coarse_verb_tokens = self.clip_tokenize([f"a photo of {v} something" for v in coarse_verbs])
        self.coarse_obj_tokens = self.clip_tokenize([f"a photo of something {o}" for o in coarse_objs])
        
        # Batching for large pairs list (to avoid OOM during pre-computation)
        pair_texts = [f"a photo of {v} {o}" for v, o in pairs]
        self.comp_tokens = self.clip_tokenize(pair_texts)
        
        print(f"[CustomCLIP] Tokenized. Ready to cache on device.")

    def _ensure_hierarchy_cached(self, device):
        """
        Compute and cache embeddings if not already done.
        This is called inside forward() to ensure device is correct.
        """
        if "comp_hyp" in self.cached_hierarchy:
            return # Already cached

        print(f"[CustomCLIP] Computing hierarchy cache on {device}...")
        self.model_device = device
        
        # We use no_grad because these are fixed text targets (not learned prompts)
        with torch.no_grad():
            # 1. Coarse Verbs
            cv_emb = self._encode_plain_text(self.coarse_verb_tokens)
            cv_hyp = LorentzMath.exp_map_0(F.normalize(cv_emb, dim=-1))
            self.cached_hierarchy["coarse_verb_hyp"] = cv_hyp

            # 2. Coarse Objs
            co_emb = self._encode_plain_text(self.coarse_obj_tokens)
            co_hyp = LorentzMath.exp_map_0(F.normalize(co_emb, dim=-1))
            self.cached_hierarchy["coarse_obj_hyp"] = co_hyp

            # 3. Compositions (Batch processing to avoid OOM)
            # Split comp_tokens into chunks
            batch_size = 500 # Adjust if still OOM
            comp_embs = []
            total = self.comp_tokens.shape[0]
            
            for i in range(0, total, batch_size):
                batch_tokens = self.comp_tokens[i : i + batch_size]
                emb = self._encode_plain_text(batch_tokens)
                comp_embs.append(emb.cpu()) # Store on CPU first to save GPU mem
            
            comp_emb_all = torch.cat(comp_embs, dim=0).to(device) # Move back to GPU or keep CPU?
            # Ideally keep on GPU for Loss calculation, but if too big, Loss needs adjustment.
            # Assuming 24GB GPU, storing ~10k vectors (10k * 512 * 4 bytes = 20MB) is trivial.
            # The OOM was caused by the Intermediate Activations during Backward/Forward of the Transformer.
            
            self.cached_hierarchy["comp_hyp"] = LorentzMath.exp_map_0(F.normalize(comp_emb_all, dim=-1))
            
        print(f"[CustomCLIP] Cache finished.")

    def forward(self, video, pairs=None):
        device = video.device
        
        # Ensure cache is ready (Lazy loading)
        if self.coarse_verb_tokens is not None and "comp_hyp" not in self.cached_hierarchy:
            self._ensure_hierarchy_cached(device)

        # --- 1. Text Encoding (Fine-grained, Length 10) ---
        verb_prompts = self.verb_prompt_learner()
        verb_text_features = self.text_encoder(verb_prompts, self.verb_tokenized_prompts)
        verb_text_features = self.c2c_text_v(verb_text_features)

        obj_prompts = self.obj_prompt_learner()
        obj_text_features = self.text_encoder(obj_prompts, self.obj_tokenized_prompts)
        obj_text_features = self.c2c_text_o(obj_text_features)

        # --- 2. Video Encoding ---
        video_features = self.video_encoder(video)
        o_feat = self.c2c_OE1(video_features.mean(dim=-1))
        v_feat_t = self.c2c_VE1(video_features)
        v_feat = v_feat_t.mean(dim=-1)

        # --- 3. Hyperbolic Transformation ---
        o_feat_hyp = LorentzMath.exp_map_0(F.normalize(o_feat, dim=1))
        v_feat_hyp = LorentzMath.exp_map_0(F.normalize(v_feat, dim=1))
        
        verb_text_hyp = LorentzMath.exp_map_0(F.normalize(verb_text_features, dim=-1))
        obj_text_hyp = LorentzMath.exp_map_0(F.normalize(obj_text_features, dim=-1))

        # --- 4. Retrieve Cached Hierarchy Features ---
        coarse_verb_hyp = self.cached_hierarchy.get("coarse_verb_hyp")
        coarse_obj_hyp = self.cached_hierarchy.get("coarse_obj_hyp")
        comp_hyp = self.cached_hierarchy.get("comp_hyp")

        # --- 5. Logits (Distance) ---
        dist_v = LorentzMath.hyp_distance(v_feat_hyp.unsqueeze(1), verb_text_hyp.unsqueeze(0))
        dist_o = LorentzMath.hyp_distance(o_feat_hyp.unsqueeze(1), obj_text_hyp.unsqueeze(0))
        
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
                "comp_hyp": comp_hyp
            }
        else:
            if pairs is not None:
                verb_idx, obj_idx = pairs[:, 0], pairs[:, 1]
                return verb_logits[:, verb_idx] + obj_logits[:, obj_idx]
            else:
                return verb_logits, obj_logits


# ... (load_clip_to_cpu and build_model remain unchanged)
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
    return model