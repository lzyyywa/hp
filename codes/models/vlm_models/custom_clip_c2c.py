import torch
import torch.nn as nn
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from models.vlm_models.text_learner import get_text_learner
import torch.nn.functional as F
from einops import rearrange
import math

# [HYPERBOLIC] 引入双曲工具库
import utils.hyperbolic_utils as L

_tokenizer = _Tokenizer()


# ==============================================================================
# [Infrastructure] 基础组件保持原样，完全不修改
# ==============================================================================

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


# ==============================================================================
# [Modification] 仅修改 CustomCLIP 以实现双曲迁移
# ==============================================================================

class CustomCLIP(nn.Module):
    def __init__(self, cfg, train_dataset, clip_model):
        super().__init__()
        # Text Prompt Learners
        self.verb_prompt_learner = get_text_learner(cfg, train_dataset, clip_model, 'verb')
        self.verb_tokenized_prompts = self.verb_prompt_learner.token_ids
        self.obj_prompt_learner = get_text_learner(cfg, train_dataset, clip_model, 'object')
        self.obj_tokenized_prompts = self.obj_prompt_learner.token_ids

        # Encoders (Infrastructure preserved)
        self.text_encoder = TextEncoder(cfg, clip_model)
        self.video_encoder = VideoEncoder(cfg, clip_model)
        
        # [BASELINE] 必须保留 logit_scale，这在双曲空间同样用于缩放距离
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

        # ----------------------------------------------------------------------
        # [HYPERBOLIC] 新增参数初始化
        # ----------------------------------------------------------------------
        # 1. 曲率 Curvature (c)
        curv_init = 1.0
        self.curv = nn.Parameter(torch.tensor(math.log(curv_init)))
        self._curv_minmax = {
            "max": math.log(curv_init * 10),
            "min": math.log(curv_init / 10),
        }

        # 2. 模长缩放因子 (Alpha)
        # 用于在进入双曲空间前调整欧氏向量的模长，防止全部挤在原点或边缘
        embed_dim = cfg.emb_dim
        init_scale_val = math.log(embed_dim**-0.5)
        
        self.visual_alpha_v = nn.Parameter(torch.tensor(init_scale_val))
        self.visual_alpha_o = nn.Parameter(torch.tensor(init_scale_val))
        self.text_alpha_v = nn.Parameter(torch.tensor(init_scale_val))
        self.text_alpha_o = nn.Parameter(torch.tensor(init_scale_val))

    def forward(self, video, pairs=None):
        # [HYPERBOLIC] 参数裁剪 (Stability)
        self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)
        _curv = self.curv.exp()
        
        # Alpha 限制 (max=0.0 表示缩放因子 <= 1.0)
        self.visual_alpha_v.data = torch.clamp(self.visual_alpha_v.data, max=0.0)
        self.visual_alpha_o.data = torch.clamp(self.visual_alpha_o.data, max=0.0)
        self.text_alpha_v.data = torch.clamp(self.text_alpha_v.data, max=0.0)
        self.text_alpha_o.data = torch.clamp(self.text_alpha_o.data, max=0.0)

        # ----------------------------------------------------------------------
        # 1. 提取欧氏特征 (Pipeline 保持不变)
        # ----------------------------------------------------------------------
        # Text Features
        verb_prompts = self.verb_prompt_learner()
        verb_text_features = self.text_encoder(verb_prompts, self.verb_tokenized_prompts)
        verb_text_features = self.c2c_text_v(verb_text_features)

        obj_prompts = self.obj_prompt_learner()
        obj_text_features = self.text_encoder(obj_prompts, self.obj_tokenized_prompts)
        obj_text_features = self.c2c_text_o(obj_text_features)

        # Video Features
        video_features = self.video_encoder(video)

        # Independent Learning (MLPs)
        o_feat = self.c2c_OE1(video_features.mean(dim=-1))
        v_feat_t = self.c2c_VE1(video_features)
        v_feat = v_feat_t.mean(dim=-1)

        # ----------------------------------------------------------------------
        # 2. [HYPERBOLIC] 空间映射 (关键修改点)
        # ----------------------------------------------------------------------
        # 逻辑：Feature * Alpha.exp() -> ExpMap -> Hyperbolic Space
        
        # Text Mapping
        verb_text_hyp = L.exp_map0(verb_text_features * self.text_alpha_v.exp(), _curv)
        obj_text_hyp = L.exp_map0(obj_text_features * self.text_alpha_o.exp(), _curv)
        
        # Video Mapping
        v_feat_hyp = L.exp_map0(v_feat * self.visual_alpha_v.exp(), _curv)
        o_feat_hyp = L.exp_map0(o_feat * self.visual_alpha_o.exp(), _curv)

        # ----------------------------------------------------------------------
        # 3. [HYPERBOLIC] Logits 计算 (距离替代相似度)
        # ----------------------------------------------------------------------
        # 原版是 @ .t() 点积，这里改为双曲距离
        logit_scale = self.logit_scale.exp()
        
        # 负距离作为 Logits (距离越小，相似度越高)
        verb_logits = -L.pairwise_dist(v_feat_hyp, verb_text_hyp, _curv) * logit_scale
        obj_logits = -L.pairwise_dist(o_feat_hyp, obj_text_hyp, _curv) * logit_scale

        # ----------------------------------------------------------------------
        # 4. 组合逻辑
        # ----------------------------------------------------------------------
        # 原版: einsum (乘积)。 双曲/Logit空间: 加法 (Log P(v) + Log P(o))
        pred_com = verb_logits.unsqueeze(2) + obj_logits.unsqueeze(1)

        if self.training:
            # [ADDITION] 训练时必须返回双曲特征，供 Loss 使用
            # 使用字典包裹，避免修改外层太多的解包逻辑
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
    # 梯度控制逻辑保持原样
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
            
        # [HYPERBOLIC] 必须开启新参数的梯度
        elif 'curv' in name or 'alpha' in name:
            param.requires_grad = True
            print(f'[Hyperbolic Init] Enable gradient for {name}')
            
    return model