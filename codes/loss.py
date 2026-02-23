from torch.nn.modules.loss import CrossEntropyLoss
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np

# [HYPERBOLIC] 引入双曲工具库
import utils.hyperbolic_utils as L

# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

def loss_calu(predict, target, config):
    loss_fn = CrossEntropyLoss()
    batch_img, batch_attr, batch_obj, batch_target = target
    batch_attr = batch_attr.cuda()
    batch_obj = batch_obj.cuda()
    batch_target = batch_target.cuda()
    logits, logits_att, logits_obj, logits_soft_prompt = predict
    loss_logit_df = loss_fn(logits, batch_target)
    loss_logit_sp = loss_fn(logits_soft_prompt, batch_target)
    loss_att = loss_fn(logits_att, batch_attr)
    loss_obj = loss_fn(logits_obj, batch_obj)
    
    # 兼容性处理：如果config没有这些属性，给予默认值
    att_obj_w = getattr(config, 'att_obj_w', 0.2)
    sp_w = getattr(config, 'sp_w', 0.0)
    
    loss = loss_logit_df + att_obj_w * (loss_att + loss_obj) + sp_w * loss_logit_sp
    return loss


# =============================================================================
# [HYPERBOLIC] 新增损失函数组件 (HyCoCLIP / Troika 风格)
# =============================================================================

class EntailmentConeLoss(nn.Module):
    """
    Hyperbolic Entailment Cone Loss.
    Penalizes child embeddings that fall outside the entailment cone of their parent.
    Reference: HyCoCLIP Eq. 10
    """
    def __init__(self, margin=0.01):
        super().__init__()
        self.margin = margin

    def forward(self, child_emb, parent_emb, curv):
        """
        Args:
            child_emb: (Batch, Dim) Child concept (Specific)
            parent_emb: (Batch, Dim) Parent concept (General)
            curv: curvature scalar
        """
        # Calculate the angle between the two points at the origin
        angle = L.oxy_angle(parent_emb, child_emb, curv)
        
        # Calculate the aperture (half-angle) of the cone at the parent
        aperture = L.half_aperture(parent_emb, curv)
        
        # Violation: if angle > aperture, the child is outside the cone.
        violation = torch.clamp(angle - aperture + self.margin, min=0.0)
        
        norm_child = torch.norm(child_emb, dim=-1)
        norm_parent = torch.norm(parent_emb, dim=-1)
        norm_penalty = torch.clamp(norm_parent - norm_child, min=0.0)
        
        # 【对齐上一版】严格保持 0.1 的范数惩罚权重
        return (violation + 0.1 * norm_penalty).mean()


def hyperbolic_loss_calu(predict, target, config):
    """
    Troika-Aligned Hyperbolic Loss Calculation
    Includes:
    1. Discriminative Alignment (Hyperbolic Contrastive)
    2. Hierarchical Entailment (Two-Level Cascade: Video->FineText->CoarseText)
    """
    loss_fn = CrossEntropyLoss()
    entail_loss_fn = EntailmentConeLoss(margin=0.01)

    # 1. Unpack
    logits_v, logits_o, logits_com, hyp_feats = predict
    batch_img, batch_attr, batch_obj, batch_target, coarse_v_hyp, coarse_o_hyp = target
    
    batch_attr = batch_attr.cuda()
    batch_obj = batch_obj.cuda()
    batch_target = batch_target.cuda()
    
    # =========================================================================
    # 1. Discriminative Alignment Loss (判别对齐)
    # =========================================================================
    # 【核心修复】显式转为 FP32 并使用 20.0 的温度缩放，对齐上一版
    loss_com = loss_fn(logits_com.float() * 20.0, batch_target)
    loss_v = loss_fn(logits_v.float() * 20.0, batch_attr)
    loss_o = loss_fn(logits_o.float() * 20.0, batch_obj)
    
    weights = getattr(config, 'loss_weights', {})
    w_att_obj = weights.get('att_obj_w', 0.2)
    
    loss_align_total = loss_com + w_att_obj * (loss_v + loss_o)
    
    # =========================================================================
    # 2. Hierarchical Entailment Loss (层级蕴含 - 双层级联)
    # =========================================================================
    curv = hyp_feats['curv']
    
    # --- Level 1: Video entails Fine Text (Action -> Verb/Object Concept) ---
    loss_entail_v_1 = entail_loss_fn(hyp_feats['v_feat_hyp'], hyp_feats['verb_text_hyp'], curv)
    loss_entail_o_1 = entail_loss_fn(hyp_feats['o_feat_hyp'], hyp_feats['obj_text_hyp'], curv)
    
    # --- Level 2: Fine Text entails Coarse Text (Concept -> Category) ---
    loss_entail_v_2 = entail_loss_fn(hyp_feats['verb_text_hyp'], coarse_v_hyp, curv)
    loss_entail_o_2 = entail_loss_fn(hyp_feats['obj_text_hyp'], coarse_o_hyp, curv)
    
    loss_entail_total = (loss_entail_v_1 + loss_entail_o_1 + loss_entail_v_2 + loss_entail_o_2)

    # =========================================================================
    # 3. Total Weighted Loss
    # =========================================================================
    w_align = weights.get('lambda_align', 1.0)
    w_entail = weights.get('lambda_entail', 0.1)  # 根据config，这里是0.1
    
    total_loss = w_align * loss_align_total + w_entail * loss_entail_total
    
    return total_loss


# =============================================================================
# Original Utils (Preserved - DO NOT DELETE)
# =============================================================================

class KLLoss(nn.Module):
    def __init__(self, error_metric=nn.KLDivLoss(size_average=True, reduce=True)):
        super().__init__()
        print('=========using KL Loss=and has temperature and * bz==========')
        self.error_metric = error_metric

    def forward(self, prediction, label,mul=False):
        batch_size = prediction.shape[0]
        probs1 = F.log_softmax(prediction, 1)
        probs2 = F.softmax(label, 1)
        loss = self.error_metric(probs1, probs2)
        if mul:
            return loss* batch_size
        else:
            return loss


def hsic_loss(input1, input2, unbiased=False):
    def _kernel(X, sigma):
        X = X.view(len(X), -1)
        XX = X @ X.t()
        X_sqnorms = torch.diag(XX)
        X_L2 = -2 * XX + X_sqnorms.unsqueeze(1) + X_sqnorms.unsqueeze(0)
        gamma = 1 / (2 * sigma ** 2)

        kernel_XX = torch.exp(-gamma * X_L2)
        return kernel_XX

    N = len(input1)
    if N < 4:
        return torch.tensor(0.0).to(input1.device)
    sigma_x = np.sqrt(input1.size()[1])
    sigma_y = np.sqrt(input2.size()[1])

    kernel_XX = _kernel(input1, sigma_x)
    kernel_YY = _kernel(input2, sigma_y)

    if unbiased:
        tK = kernel_XX - torch.diag(kernel_XX)
        tL = kernel_YY - torch.diag(kernel_YY)
        hsic = (
                torch.trace(tK @ tL)
                + (torch.sum(tK) * torch.sum(tL) / (N - 1) / (N - 2))
                - (2 * torch.sum(tK, 0).dot(torch.sum(tL, 0)) / (N - 2))
        )
        loss = hsic / (N * (N - 3))
    else:
        KH = kernel_XX - kernel_XX.mean(0, keepdim=True)
        LH = kernel_YY - kernel_YY.mean(0, keepdim=True)
        loss = torch.trace(KH @ LH / (N - 1) ** 2)
    return loss


class Gml_loss(nn.Module):
    def __init__(self, error_metric=nn.KLDivLoss(size_average=True, reduce=True)):
        super().__init__()

    def forward(self, p_o_on_v, v_label, n_c, t=100.0):
        n_c = n_c[:, 0]
        b = p_o_on_v.shape[0]
        n_o = p_o_on_v.shape[-1]
        p_o = p_o_on_v[range(b), v_label, :]  

        num_c = n_c.sum().view(1, -1)  

        p_o_exp = torch.exp(p_o * t)
        p_o_exp_wed = num_c * p_o_exp  
        p_phi = p_o_exp_wed / torch.sum(p_o_exp_wed, dim=0, keepdim=True)  

        p_ba = torch.sum(p_phi * n_c, dim=0, keepdim=True) / (num_c + 1.0e-6)  
        p_ba[torch.where(p_ba < 1.0e-8)] = 1.0
        p_ba_log = torch.log(p_ba)
        loss = (-1.0 / n_o) * p_ba_log.sum()

        return loss