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
    loss = loss_logit_df + config.att_obj_w * (loss_att + loss_obj) + config.sp_w * loss_logit_sp
    return loss


# =============================================================================
# [HYPERBOLIC] 新增损失函数组件
# =============================================================================

class EntailmentConeLoss(nn.Module):
    """
    Hyperbolic Entailment Cone Loss.
    Penalizes child embeddings that fall outside the entailment cone of their parent.
    Reference: Ganea et al., "Hyperbolic Entailment Cones for Learning Hierarchical Embeddings"
    """
    def __init__(self, margin=0.01):
        super().__init__()
        self.margin = margin

    def forward(self, fine_emb, coarse_emb, curv):
        """
        Args:
            fine_emb: (Batch, Dim) Child concept (e.g., specific action video feature)
            coarse_emb: (Batch, Dim) Parent concept (e.g., coarse action text feature)
            curv: curvature scalar
        """
        # Calculate the angle between the two points at the origin
        angle = L.oxy_angle(coarse_emb, fine_emb, curv)
        
        # Calculate the aperture (half-angle) of the cone at the parent
        # Note: aperture is determined by the norm of the parent
        aperture = L.half_aperture(coarse_emb, curv)
        
        # Violation: if angle > aperture, the child is outside the cone.
        # We minimize max(0, angle - aperture + margin)
        violation = torch.clamp(angle - aperture + self.margin, min=0.0)
        
        return violation.mean()


def hyperbolic_loss_calu(predict, target, config):
    """
    专门用于双曲模式的损失计算函数
    Predict: (verb_logits, obj_logits, com_logits, hyp_features)
    Target: (batch_img, batch_attr, batch_obj, batch_target, coarse_v_hyp, coarse_o_hyp)
    """
    loss_fn = CrossEntropyLoss()
    entail_loss_fn = EntailmentConeLoss(margin=0.01)

    # 1. Unpack Predictions
    logits_v, logits_o, logits_com, hyp_feats = predict
    
    # 2. Unpack Targets
    # 注意：train_models.py 需要被修改以传递 coarse_v_hyp 和 coarse_o_hyp
    batch_img, batch_attr, batch_obj, batch_target, coarse_v_hyp, coarse_o_hyp = target
    
    batch_attr = batch_attr.cuda()
    batch_obj = batch_obj.cuda()
    batch_target = batch_target.cuda()
    
    # 3. Discriminative Alignment Loss (判别对齐损失)
    # 在双曲空间中，Logits = -Distance * Scale。
    # CrossEntropyLoss = -log(softmax(logits)) = -log(softmax(-dist))
    # 最小化 CE Loss 等价于最大化正确类别的 Logit (即最小化距离)。
    # 这自然实现了 "Aligning fine-grained video features with fine-grained text features"
    loss_com = loss_fn(logits_com, batch_target)
    loss_v = loss_fn(logits_v, batch_attr)
    loss_o = loss_fn(logits_o, batch_obj)
    
    # 4. Hierarchical Entailment Loss (层次蕴含损失)
    # 约束：Fine Video Feature 应当位于 Coarse Text Feature 的蕴含锥内
    curv = hyp_feats['curv']
    
    # Video-Verb Fine vs Text-Verb Coarse
    loss_entail_v = entail_loss_fn(hyp_feats['v_feat_hyp'], coarse_v_hyp, curv)
    # Video-Obj Fine vs Text-Obj Coarse
    loss_entail_o = entail_loss_fn(hyp_feats['o_feat_hyp'], coarse_o_hyp, curv)
    
    loss_entail = loss_entail_v + loss_entail_o

    # 5. Total Loss
    # 获取权重参数，如果 config 中没有定义则使用默认值
    lambda_entail = getattr(config, 'lambda_entail', 0.1) 
    
    # C2C 原始结构: Composition + 0.2 * (Verb + Obj)
    # 新增: + lambda * Entailment
    loss = loss_com + config.att_obj_w * (loss_v + loss_o) + lambda_entail * loss_entail
    
    return loss


# =============================================================================
# Original Utils (Preserved)
# =============================================================================

class KLLoss(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """

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
    # we simply use the squared dimension of feature as the sigma for RBF kernel
    sigma_x = np.sqrt(input1.size()[1])
    sigma_y = np.sqrt(input2.size()[1])

    # compute the kernels
    kernel_XX = _kernel(input1, sigma_x)
    kernel_YY = _kernel(input2, sigma_y)

    if unbiased:
        """Unbiased estimator of Hilbert-Schmidt Independence Criterion
        Song, Le, et al. "Feature selection via dependence maximization." 2012.
        """
        # tK = kernel_XX - torch.diag(torch.diag(kernel_XX))
        # tL = kernel_YY - torch.diag(torch.diag(kernel_YY))
        tK = kernel_XX - torch.diag(kernel_XX)
        tL = kernel_YY - torch.diag(kernel_YY)
        hsic = (
                torch.trace(tK @ tL)
                + (torch.sum(tK) * torch.sum(tL) / (N - 1) / (N - 2))
                - (2 * torch.sum(tK, 0).dot(torch.sum(tL, 0)) / (N - 2))
        )
        loss = hsic / (N * (N - 3))
    else:
        """Biased estimator of Hilbert-Schmidt Independence Criterion
        Gretton, Arthur, et al. "Measuring statistical dependence with Hilbert-Schmidt norms." 2005.
        """
        KH = kernel_XX - kernel_XX.mean(0, keepdim=True)
        LH = kernel_YY - kernel_YY.mean(0, keepdim=True)
        loss = torch.trace(KH @ LH / (N - 1) ** 2)
    return loss


class Gml_loss(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    Loss from No One Left Behind: Improving the Worst Categories in Long-Tailed Learning
    """

    def __init__(self, error_metric=nn.KLDivLoss(size_average=True, reduce=True)):
        super().__init__()

    def forward(self, p_o_on_v, v_label, n_c, t=100.0):
        '''

        Args:
            p_o_on_v: b,n_v,n_o
            o_label: b,
            n_c: b,n_o

        Returns:

        '''
        n_c = n_c[:, 0]
        b = p_o_on_v.shape[0]
        n_o = p_o_on_v.shape[-1]
        p_o = p_o_on_v[range(b), v_label, :]  # b,n_o

        num_c = n_c.sum().view(1, -1)  # 1,n_o

        p_o_exp = torch.exp(p_o * t)
        p_o_exp_wed = num_c * p_o_exp  # b,n_o
        p_phi = p_o_exp_wed / torch.sum(p_o_exp_wed, dim=0, keepdim=True)  # b,n_o

        p_ba = torch.sum(p_phi * n_c, dim=0, keepdim=True) / (num_c + 1.0e-6)  # 1,n_o
        p_ba[torch.where(p_ba < 1.0e-8)] = 1.0
        p_ba_log = torch.log(p_ba)
        loss = (-1.0 / n_o) * p_ba_log.sum()

        return loss