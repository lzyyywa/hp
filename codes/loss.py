from torch.nn.modules.loss import CrossEntropyLoss
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np

# Import the upgraded hyperbolic operations
try:
    from utils.hyperbolic_ops import LorentzMath
except ImportError:
    print("Warning: Could not import LorentzMath. Hyperbolic losses will fail.")

def loss_calu(predict, target, config):
    """Original Euclidean loss for backward compatibility"""
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
    """
    Discriminative Loss in Hyperbolic Space.
    Uses negative hyperbolic distance as logits.
    """
    def __init__(self, temperature=0.1, c=1.0):
        super(HyperbolicPrototypicalLoss, self).__init__()
        self.temperature = temperature
        self.c = c
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, query_emb, prototype_emb, targets):
        q = query_emb.unsqueeze(1)
        p = prototype_emb.unsqueeze(0)
        # Uses the stable distance from upgraded LorentzMath
        dists = LorentzMath.hyp_distance(q, p, c=self.c)
        logits = -dists / self.temperature
        return self.loss_fn(logits, targets)

class EntailmentConeLoss(nn.Module):
    """
    [UPGRADED] Hierarchical Entailment Cone Loss (HyCoCLIP Style).
    Enforces u < v (v is inside the cone of u).
    u: Parent (Abstract)
    v: Child (Specific)
    
    FEATURE: Uses DYNAMIC APERTURE based on parent norm.
    """
    def __init__(self, min_radius=0.1, margin=0.01, aperture_scale=1.2):
        super(EntailmentConeLoss, self).__init__()
        self.min_radius = min_radius
        self.margin = margin
        self.aperture_scale = aperture_scale # Scales the calculated dynamic aperture
        self.c = 1.0 # Default curvature

    def forward(self, child_emb, parent_emb):
        """
        Args:
            child_emb: [B, d+1] (Specific concept)
            parent_emb: [B, d+1] (Abstract concept)
        We want: Child to be INSIDE the cone of Parent.
        """
        
        # 1. Calculate actual hyperbolic angle between Parent and Child
        # This uses the hyperbolic law of cosines (from upgraded LorentzMath)
        angle = LorentzMath.oxy_angle(parent_emb, child_emb, c=self.c)
        
        # 2. [NEW] Calculate DYNAMIC aperture based on Parent's norm
        # Closer to origin (abstract) -> Wider cone (easier to contain children)
        # Further from origin (specific) -> Narrower cone
        dynamic_aperture = LorentzMath.half_aperture(parent_emb, c=self.c, min_radius=self.min_radius)
        
        # 3. Apply scaling factor (HyCoCLIP practice)
        # Scale > 1.0 widens the cone slightly to help convergence
        target_aperture = self.aperture_scale * dynamic_aperture
        
        # 4. Cone Constraint Loss
        # We want: angle < target_aperture
        # Loss = ReLU(angle - target_aperture)
        cone_loss = F.relu(angle - target_aperture + self.margin)
        
        return cone_loss.mean()

class H2EMTotalLoss(nn.Module):
    """
    H2EM Paper Eq. (16) Wrapper.
    """
    def __init__(self, temperature=0.1, beta1=1.0, beta2=0.1, beta3=0.5):
        super(H2EMTotalLoss, self).__init__()
        self.beta1 = beta1 # DA
        self.beta2 = beta2 # TE (Can be kept small now because aperture is dynamic)
        self.beta3 = beta3 # Prim
        
        self.loss_cls = HyperbolicPrototypicalLoss(temperature=temperature)
        
        # [UPDATED] Initialize with dynamic aperture parameters
        # aperture_scale=1.2 gives the model 20% more room than strict geometry, helping early convergence
        self.loss_cone = EntailmentConeLoss(min_radius=0.1, margin=0.01, aperture_scale=1.2)

    def forward(self, out, batch_verb, batch_obj, batch_target, p2v_map, p2o_map, v2cv_map, o2co_map):
        # Unpack
        v_feat_hyp = out['v_feat_hyp']
        o_feat_hyp = out['o_feat_hyp']
        verb_text_hyp = out['verb_text_hyp']
        obj_text_hyp = out['obj_text_hyp']
        coarse_verb_hyp = out['coarse_verb_hyp']
        coarse_obj_hyp = out['coarse_obj_hyp']
        
        comp_hyp_v = out['comp_hyp_v'] 
        comp_hyp_o = out['comp_hyp_o'] 
        
        verb_logits = out['verb_logits']
        obj_logits = out['obj_logits']

        # A. Primitive Auxiliary (L_s + L_o)
        L_s = self.loss_cls(v_feat_hyp, verb_text_hyp, batch_verb)
        L_o = self.loss_cls(o_feat_hyp, obj_text_hyp, batch_obj)
        L_prim = L_s + L_o

        # B. Discriminative Alignment (L_DA)
        pair_logits = verb_logits[:, p2v_map] + obj_logits[:, p2o_map]
        L_DA = F.cross_entropy(pair_logits / self.loss_cls.temperature, batch_target)

        # C. Taxonomic Entailment (L_TE)
        # Cone directions: Parent contains Child
        # 1. Coarse Verb contains Verb
        loss_h1 = self.loss_cone(child_emb=verb_text_hyp, parent_emb=coarse_verb_hyp[v2cv_map])
        # 2. Coarse Object contains Object
        loss_h2 = self.loss_cone(child_emb=obj_text_hyp, parent_emb=coarse_obj_hyp[o2co_map])
        # 3. Verb contains Composition
        loss_h3 = self.loss_cone(child_emb=comp_hyp_v, parent_emb=verb_text_hyp[p2v_map])
        # 4. Object contains Composition
        loss_h4 = self.loss_cone(child_emb=comp_hyp_o, parent_emb=obj_text_hyp[p2o_map])
        
        L_TE = loss_h1 + loss_h2 + loss_h3 + loss_h4

        # Total Loss
        loss = self.beta1 * L_DA + self.beta2 * L_TE + self.beta3 * L_prim
        
        return loss, {
            "Total": loss.item(),
            "Prim": L_prim.item(),
            "DA": L_DA.item(),
            "TE": L_TE.item()
        }

# =========================================================================
# Legacy Losses (Keep unchanged)
# =========================================================================

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
        if mul: return loss* batch_size
        else: return loss

def hsic_loss(input1, input2, unbiased=False):
    return torch.tensor(0.0).to(input1.device)

class Gml_loss(nn.Module):
    def __init__(self, error_metric=nn.KLDivLoss(size_average=True, reduce=True)):
        super().__init__()
    def forward(self, p_o_on_v, v_label, n_c, t=100.0):
        return torch.tensor(0.0).to(p_o_on_v.device)