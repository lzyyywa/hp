import torch
import torch.nn as nn
import math
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from models.vlm_models.text_learner import get_text_learner
import torch.nn.functional as F
from einops import rearrange

# [HYPERBOLIC] 引入双曲工具库
import utils.hyperbolic_utils as L

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

        for block in self.transformer.resblocks:
            block.attn_mask = block.attn_mask[:cfg.ctx_length, :cfg.ctx_length]
        self.dtype = clip_model.dtype

    def forward(self, x, tokenized_prompts):
        x = x.permute(1, 0, 2)
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
        # Text Prompt Learners
        self.verb_prompt_learner = get_text_learner(cfg, train_dataset, clip_model, 'verb')
        self.verb_tokenized_prompts = self.verb_prompt_learner.token_ids
        self.obj_prompt_learner = get_text_learner(cfg, train_dataset, clip_model, 'object')
        self.obj_tokenized_prompts = self.obj_prompt_learner.token_ids

        # Encoders
        self.text_encoder = TextEncoder(cfg, clip_model)
        self.video_encoder = VideoEncoder(cfg, clip_model)
        
        # [HYPERBOLIC] Removed scalar logit_scale in favor of distance based metric
        # self.logit_scale = clip_model.logit_scale 

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

        # =====================================================================
        # [HYPERBOLIC] Initialization (Reference: HyCoCLIP/MERU)
        # =====================================================================
        curv_init = 1.0
        self.curv = nn.Parameter(torch.tensor(math.log(curv_init)))
        # Restrict curvature range to prevent instability
        self._curv_minmax = {
            "max": math.log(curv_init * 10),
            "min": math.log(curv_init / 10),
        }

        # Learnable scalars to ensure that features have an expected unit norm 
        # before exponential map. C2C splits Verb/Obj, so we need pairs.
        embed_dim = cfg.emb_dim
        
        # For Video Features
        self.visual_alpha_v = nn.Parameter(torch.tensor(embed_dim**-0.5).log())
        self.visual_alpha_o = nn.Parameter(torch.tensor(embed_dim**-0.5).log())
        
        # For Text Features
        self.text_alpha_v = nn.Parameter(torch.tensor(embed_dim**-0.5).log())
        self.text_alpha_o = nn.Parameter(torch.tensor(embed_dim**-0.5).log())
        # =====================================================================

    def forward(self, video, pairs=None):
        # [HYPERBOLIC] Clamp curvature and alphas
        self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)
        _curv = self.curv.exp()
        
        # Clamp alphas (max=0.0 means scale <= 1.0)
        self.visual_alpha_v.data = torch.clamp(self.visual_alpha_v.data, max=0.0)
        self.visual_alpha_o.data = torch.clamp(self.visual_alpha_o.data, max=0.0)
        self.text_alpha_v.data = torch.clamp(self.text_alpha_v.data, max=0.0)
        self.text_alpha_o.data = torch.clamp(self.text_alpha_o.data, max=0.0)

        # ---------------------------------------------------
        # 1. Text Features (Euclidean -> Hyperbolic)
        # ---------------------------------------------------
        verb_prompts = self.verb_prompt_learner()
        verb_text_features_euc = self.text_encoder(verb_prompts, self.verb_tokenized_prompts)
        verb_text_features_euc = self.c2c_text_v(verb_text_features_euc)

        obj_prompts = self.obj_prompt_learner()
        obj_text_features_euc = self.text_encoder(obj_prompts, self.obj_tokenized_prompts)
        obj_text_features_euc = self.c2c_text_o(obj_text_features_euc)
        
        # [HYPERBOLIC] Map Text to Hyperbolic Space
        # Step 1: Scale Euclidean vectors
        verb_text_scaled = verb_text_features_euc * self.text_alpha_v.exp()
        obj_text_scaled = obj_text_features_euc * self.text_alpha_o.exp()
        
        # Step 2: Exponential Map (Tangent -> Manifold)
        verb_text_hyp = L.exp_map0(verb_text_scaled, _curv)
        obj_text_hyp = L.exp_map0(obj_text_scaled, _curv)

        # ---------------------------------------------------
        # 2. Video Features (Euclidean -> Hyperbolic)
        # ---------------------------------------------------
        video_features_raw = self.video_encoder(video)

        # Video Object Feature (Mean Pooling -> MLP)
        o_feat_euc = self.c2c_OE1(video_features_raw.mean(dim=-1))
        
        # Video Verb Feature (Conv1d -> Mean Pooling)
        v_feat_t_euc = self.c2c_VE1(video_features_raw)
        v_feat_euc = v_feat_t_euc.mean(dim=-1)

        # [HYPERBOLIC] Map Video to Hyperbolic Space
        # Step 1: Scale
        o_feat_scaled = o_feat_euc * self.visual_alpha_o.exp()
        v_feat_scaled = v_feat_euc * self.visual_alpha_v.exp()
        
        # Step 2: Exp Map
        o_feat_hyp = L.exp_map0(o_feat_scaled, _curv)
        v_feat_hyp = L.exp_map0(v_feat_scaled, _curv)

        # ---------------------------------------------------
        # 4. Logits Calculation (-Distance)
        # ---------------------------------------------------
        # NOTE: Original C2C used cosine similarity (dot product).
        # Hyperbolic equivalent is negative distance.
        # pairwise_dist returns distance >= 0. We negate it for compatibility with CrossEntropy.
        
        verb_logits = -L.pairwise_dist(v_feat_hyp, verb_text_hyp, _curv)
        obj_logits = -L.pairwise_dist(o_feat_hyp, obj_text_hyp, _curv)
        
        # NOTE: C2C vanilla scaled logits: verb_logits * 0.5 + 0.5
        # In hyperbolic, logits are unbounded negative numbers. Scaling is handled by Temperature in HyCoCLIP
        # but here we output raw negative distances. The Loss function typically handles scaling (logit_scale).
        # We will assume Loss function or external scalar handles it, OR we can apply a fixed scale here.
        # HyCoCLIP applies `logit_scale` in the model forward. C2C applies it in the CLIP model but 
        # custom_c2c replaced the head. 
        # Let's keep raw negative distance here, compatible with `CrossEntropy`.

        # ---------------------------------------------------
        # 5. Composition (Sum of Logits / Product of Probabilities)
        # ---------------------------------------------------
        # P(v, o) = P(v) * P(o)
        # log P(v, o) = log P(v) + log P(o)
        # Since Logits ~ log P, we Sum the logits.
        # verb_logits: [B, N_v], obj_logits: [B, N_o]
        # pred_com: [B, N_v, N_o]
        
        pred_com = verb_logits.unsqueeze(2) + obj_logits.unsqueeze(1)

        if self.training:
            # [HYPERBOLIC] We return the features needed for Hierarchical Entailment Loss
            # Pack them into a dictionary to avoid breaking `v, o, com = model()` unpacking if we can help it,
            # BUT we HAVE to return 4 items now. The training loop MUST be updated.
            hyperbolic_features = {
                "v_feat_hyp": v_feat_hyp,
                "o_feat_hyp": o_feat_hyp,
                "verb_text_hyp": verb_text_hyp,
                "obj_text_hyp": obj_text_hyp,
                "curv": _curv
            }
            return verb_logits, obj_logits, pred_com, hyperbolic_features
        else:
            verb_idx, obj_idx = pairs[:, 0], pairs[:, 1]
            com_logits = pred_com[:, verb_idx, obj_idx]
            return com_logits


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
    print("Building custom CLIP (Hyperbolic C2C Version)")
    model = CustomCLIP(cfg, train_dataset, clip_model)

    print("Turning off gradients in both the image and the text encoder")
    for name, param in model.named_parameters():
        param.requires_grad_(False)
        if "prompt_learner" in name:
            if cfg.learn_input_method != 'zero':
                if cfg.learn_input_method == 'coop':
                    if 'prompt_vectors' in name:
                        param.requires_grad_(True)
                elif cfg.learn_input_method == 'csp':
                    if 'obj_embedding' in name or 'verb_embedding' in name or 'comp_embedding' in name:
                        param.requires_grad_(True)
                elif cfg.learn_input_method == 'spm':
                    if 'prompt_vectors' in name or 'obj_embedding' in name or 'verb_embedding' in name or 'comp_embedding' in name:
                        param.requires_grad_(True)
                else:
                    raise NotImplementedError
        elif 'video_encoder' in name:
            if 'temporal_embedding' in name or 'ln_post' in name or 'Adapter' in name or 'clip_proj' in name:
                param.requires_grad = True
        elif 'c2c' in name:
            param.requires_grad = True
        # [HYPERBOLIC] Enable gradients for new parameters
        elif 'curv' in name or 'alpha' in name:
            param.requires_grad = True
            print(f'[Hyperbolic] Enable gradient for {name}')
            
    return model