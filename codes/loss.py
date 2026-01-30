from torch.nn.modules.loss import CrossEntropyLoss
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np

try:
    from utils.hyperbolic_ops import LorentzMath
except ImportError:
    print("Warning: Could not import LorentzMath.")

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

class HyperbolicPrototypicalLoss(nn.Module):
    def __init__(self, temperature=0.1, c=1.0):
        super(HyperbolicPrototypicalLoss, self).__init__()
        self.temperature = temperature
        self.c = c
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, query_emb, prototype_emb, targets):
        q = query_emb.unsqueeze(1)
        p = prototype_emb.unsqueeze(0)
        dists = LorentzMath.hyp_distance(q, p, c=self.c)
        logits = -dists / self.temperature
        return self.loss_fn(logits, targets)

class EntailmentConeLoss(nn.Module):
    def __init__(self, aperture=0.01, margin=0.01):
        super(EntailmentConeLoss, self).__init__()
        self.aperture = aperture
        self.margin = margin

    def forward(self, child_emb, parent_emb):
        child_r = child_emb[..., 0]
        parent_r = parent_emb[..., 0]
        depth_loss = F.relu(parent_r - child_r + self.margin)

        child_space = child_emb[..., 1:]
        parent_space = parent_emb[..., 1:]
        cos_sim = F.cosine_similarity(child_space, parent_space, dim=-1)
        angle_loss = F.relu((1.0 - self.aperture) - cos_sim)
        return depth_loss.mean() + angle_loss.mean()

class H2EMTotalLoss(nn.Module):
    def __init__(self, temperature=0.1, beta1=1.0, beta2=0.1, beta3=0.5):
        super(H2EMTotalLoss, self).__init__()
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        
        self.loss_cls = HyperbolicPrototypicalLoss(temperature=temperature)
        self.loss_cone = EntailmentConeLoss(aperture=0.1, margin=0.01)

    def forward(self, out, batch_verb, batch_obj, batch_target, p2v_map, p2o_map, v2cv_map, o2co_map):
        # Unpack hyperbolic features
        v_feat_hyp = out['v_feat_hyp']
        o_feat_hyp = out['o_feat_hyp']
        verb_text_hyp = out['verb_text_hyp']
        obj_text_hyp = out['obj_text_hyp']
        coarse_verb_hyp = out['coarse_verb_hyp']
        coarse_obj_hyp = out['coarse_obj_hyp']
        
        # [FIX] Use composition features projected to respective spaces
        comp_hyp_v = out['comp_hyp_v'] # For Comp -> Verb alignment
        comp_hyp_o = out['comp_hyp_o'] # For Comp -> Object alignment
        
        verb_logits = out['verb_logits']
        obj_logits = out['obj_logits']

        # A. Primitive Auxiliary Loss (Verb/Object Classification)
        L_s = self.loss_cls(v_feat_hyp, verb_text_hyp, batch_verb)
        L_o = self.loss_cls(o_feat_hyp, obj_text_hyp, batch_obj)
        L_prim = L_s + L_o

        # B. Discriminative Alignment Loss (Pair Classification)
        pair_logits = verb_logits[:, p2v_map] + obj_logits[:, p2o_map]
        L_DA = F.cross_entropy(pair_logits / self.loss_cls.temperature, batch_target)

        # C. Taxonomic Entailment Loss (Hierarchy Preservation)
        loss_h1 = self.loss_cone(verb_text_hyp, coarse_verb_hyp[v2cv_map])
        loss_h2 = self.loss_cone(obj_text_hyp, coarse_obj_hyp[o2co_map])
        # [FIX] Corresponding alignment for composition features
        loss_h3 = self.loss_cone(comp_hyp_v, verb_text_hyp[p2v_map])
        loss_h4 = self.loss_cone(comp_hyp_o, obj_text_hyp[p2o_map])
        L_TE = loss_h1 + loss_h2 + loss_h3 + loss_h4

        # Total loss with weighting factors
        loss = self.beta1 * L_DA + self.beta2 * L_TE + self.beta3 * L_prim
        
        return loss, {
            "Total": loss.item(),
            "Prim": L_prim.item(),
            "DA": L_DA.item(),
            "TE": L_TE.item()
        }

# ... (KLLoss, hsic_loss, Gml_loss remain unchanged - ensure these are at the end of the file)
class KLLoss(nn.Module):
    def __init__(self, error_metric=nn.KLDivLoss(size_average=True, reduce=True)):
        super().__init__()
        self.error_metric = error_metric
        
    def forward(self, prediction, label, mul=False):
        probs1 = F.log_softmax(prediction, 1)
        probs2 = F.softmax(label, 1)
        loss = self.error_metric(probs1, probs2)
        if mul: 
            return loss * prediction.shape[0]
        else: 
            return loss

def hsic_loss(input1, input2, unbiased=False):
    return torch.tensor(0.0).to(input1.device)

class Gml_loss(nn.Module):
    def __init__(self, error_metric=nn.KLDivLoss(size_average=True, reduce=True)):
        super().__init__()
        
    def forward(self, p_o_on_v, v_label, n_c, t=100.0):
        return torch.tensor(0.0).to(p_o_on_v.device)