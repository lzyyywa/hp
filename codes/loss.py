import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss

# [HyCoCLIP Alignment] Import correct ops
try:
    from utils.hyperbolic_ops import LorentzMath
except ImportError:
    print("Warning: Could not import LorentzMath. Hyperbolic losses will fail.")

def loss_calu(predict, target, config):
    """
    [Legacy] Original Euclidean loss for backward compatibility.
    NOT used in the Hyperbolic H2EM framework.
    """
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

# =========================================================================
# Hyperbolic Losses (Strict HyCoCLIP Logic)
# =========================================================================

class HyperbolicPrototypicalLoss(nn.Module):
    """
    Discriminative Loss in Hyperbolic Space.
    Computes distance-based logits and applies CrossEntropy.
    """
    def __init__(self):
        super(HyperbolicPrototypicalLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, query_emb, prototype_emb, targets, logit_scale=100.0, c=1.0):
        """
        Args:
            query_emb: [B, D+1] (Hyperbolic Visual Features)
            prototype_emb: [N, D+1] (Hyperbolic Text Prototypes)
            targets: [B] (Ground Truth Indices)
            logit_scale: Scalar or Tensor (Learnable Scale)
            c: Scalar (Curvature)
        """
        # query: [B, D+1], prototype: [N, D+1]
        q = query_emb.unsqueeze(1)  # [B, 1, D+1]
        p = prototype_emb.unsqueeze(0)  # [1, N, D+1]
        
        # Calculate Hyperbolic Distance (Broadcasting happens inside LorentzMath)
        # Returns [B, N]
        dists = LorentzMath.hyp_distance(q, p, c=c)
        
        # Logits = -distance * scale
        # Note: logit_scale is passed from custom_clip_c2c.py (already exp-ed)
        logits = -dists * logit_scale
        
        return self.loss_fn(logits, targets)

class EntailmentConeLoss(nn.Module):
    """
    Hierarchical Entailment Cone Loss (HyCoCLIP Style).
    Enforces that 'child' is within the cone of 'parent'.
    """
    def __init__(self, min_radius=0.1, margin=0.01, aperture_scale=1.5):
        super(EntailmentConeLoss, self).__init__()
        self.min_radius = min_radius
        self.margin = margin
        self.aperture_scale = aperture_scale 

    def forward(self, child_emb, parent_emb, c=1.0):
        """
        Args:
            child_emb: [K, D+1]
            parent_emb: [K, D+1]
        """
        # [HyCoCLIP Logic] Angle calculation
        angle = LorentzMath.oxy_angle(parent_emb, child_emb, c=c)
        
        # [HyCoCLIP Logic] Aperture calculation
        dynamic_aperture = LorentzMath.half_aperture(parent_emb, c=c, min_radius=self.min_radius)
        
        # Scale aperture (Optional hyperparam from your method)
        target_aperture = self.aperture_scale * dynamic_aperture
        
        # Violation: angle > aperture
        # Loss = max(0, angle - aperture + margin)
        cone_loss = torch.clamp(angle - target_aperture + self.margin, min=0.0)
        
        return cone_loss.mean()

class H2EMTotalLoss(nn.Module):
    """
    H2EM Paper Eq. (16) Wrapper.
    Orchestrates the total loss calculation for the Hyperbolic framework.
    """
    def __init__(self, temperature=None, beta1=1.0, beta2=0.1, beta3=0.5):
        super(H2EMTotalLoss, self).__init__()
        self.beta1 = beta1  # Weight for DA (Discriminative Alignment)
        self.beta2 = beta2  # Weight for TE (Taxonomic Entailment)
        self.beta3 = beta3  # Weight for Prim (Primitive Auxiliary)
        
        self.loss_cls = HyperbolicPrototypicalLoss()
        self.loss_cone = EntailmentConeLoss(min_radius=0.1, margin=0.01, aperture_scale=1.5)

    def forward(self, out, batch_verb, batch_obj, batch_target, p2v_map, p2o_map, v2cv_map, o2co_map):
        # [Safety Check] Ensure input is a dictionary
        if not isinstance(out, dict):
            raise TypeError(f"[H2EMTotalLoss] Expected 'out' to be a dict from custom_clip_c2c, but got {type(out)}.")

        # 1. Unpack Features (Hyperbolic)
        v_feat_hyp = out['v_feat_hyp']
        o_feat_hyp = out['o_feat_hyp']
        verb_text_hyp = out['verb_text_hyp']
        obj_text_hyp = out['obj_text_hyp']
        coarse_verb_hyp = out['coarse_verb_hyp']
        coarse_obj_hyp = out['coarse_obj_hyp']
        
        comp_hyp_v = out['comp_hyp_v'] 
        comp_hyp_o = out['comp_hyp_o'] 
        
        # 2. Unpack Pre-Calculated Logits (Already Scaled in Model)
        verb_logits = out['verb_logits']
        obj_logits = out['obj_logits']
        
        # 3. Get Scale & Curvature
        logit_scale = out['logit_scale']
        current_c = out['curvature']

        # ---------------------------------------------------------------------
        # A. Primitive Auxiliary Loss (L_s + L_o)
        # ---------------------------------------------------------------------
        # Uses embeddings directly. We pass logit_scale to compute scaled logits inside.
        L_s = self.loss_cls(v_feat_hyp, verb_text_hyp, batch_verb, logit_scale, c=current_c)
        L_o = self.loss_cls(o_feat_hyp, obj_text_hyp, batch_obj, logit_scale, c=current_c)
        L_prim = L_s + L_o

        # ---------------------------------------------------------------------
        # B. Discriminative Alignment Loss (L_DA)
        # ---------------------------------------------------------------------
        # [CRITICAL LOGIC] Additive Fusion for Logits
        # verb_logits: [B, NumVerbs], obj_logits: [B, NumObjs]
        # p2v_map: [NumPairs] -> indices of verbs for each pair
        # We perform Late Fusion by adding the logits (Negative Distances).
        # This corresponds to multiplying probabilities: P(v,o) = P(v) * P(o)
        # DO NOT multiply logits here.
        pair_logits = verb_logits[:, p2v_map] + obj_logits[:, p2o_map]
        L_DA = F.cross_entropy(pair_logits, batch_target)

        # ---------------------------------------------------------------------
        # C. Taxonomic Entailment Loss (L_TE)
        # ---------------------------------------------------------------------
        # Checks 4 hierarchical relationships
        # 1. Fine-Verb -> Coarse-Verb
        loss_h1 = self.loss_cone(child_emb=verb_text_hyp, parent_emb=coarse_verb_hyp[v2cv_map], c=current_c)
        # 2. Fine-Obj -> Coarse-Obj
        loss_h2 = self.loss_cone(child_emb=obj_text_hyp, parent_emb=coarse_obj_hyp[o2co_map], c=current_c)
        # 3. Composition (Visual) -> Verb
        # Using pre-computed backbone features for composition text embeddings
        loss_h3 = self.loss_cone(child_emb=comp_hyp_v, parent_emb=verb_text_hyp[p2v_map], c=current_c)
        # 4. Composition (Visual) -> Object
        loss_h4 = self.loss_cone(child_emb=comp_hyp_o, parent_emb=obj_text_hyp[p2o_map], c=current_c)
        
        L_TE = loss_h1 + loss_h2 + loss_h3 + loss_h4

        # Total Loss
        loss = self.beta1 * L_DA + self.beta2 * L_TE + self.beta3 * L_prim
        
        return loss, {
            "Total": loss.item(),
            "Prim": L_prim.item(),
            "DA": L_DA.item(),
            "TE": L_TE.item(),
            "Curvature": current_c.item() if isinstance(current_c, torch.Tensor) else current_c
        }

# =========================================================================
# Legacy Losses (Stubs for compatibility)
# =========================================================================

class KLLoss(nn.Module):
    def __init__(self, error_metric=nn.KLDivLoss(size_average=True, reduce=True)):
        super().__init__()
        self.error_metric = error_metric
    def forward(self, prediction, label, mul=False):
        batch_size = prediction.shape[0]
        probs1 = F.log_softmax(prediction, 1)
        probs2 = F.softmax(label, 1)
        loss = self.error_metric(probs1, probs2)
        if mul: return loss * batch_size
        else: return loss

def hsic_loss(input1, input2, unbiased=False):
    return torch.tensor(0.0).to(input1.device)

class Gml_loss(nn.Module):
    def __init__(self, error_metric=nn.KLDivLoss(size_average=True, reduce=True)):
        super().__init__()
    def forward(self, p_o_on_v, v_label, n_c, t=100.0):
        return torch.tensor(0.0).to(p_o_on_v.device)