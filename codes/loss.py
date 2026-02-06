import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# [HyCoCLIP Alignment] Import correct ops
try:
    from utils.hyperbolic_ops import LorentzMath
except ImportError:
    print("Warning: Could not import LorentzMath. Hyperbolic losses will fail.")

# ... [保留原本的 loss_calu, KLLoss 等 legacy 代码，不要动] ...

# =========================================================================
# Hyperbolic Losses (Updated for H2EM Stability & Hard Negative Mining)
# =========================================================================

class HyperbolicPrototypicalLoss(nn.Module):
    """
    Discriminative Loss in Hyperbolic Space.
    """
    def __init__(self):
        super(HyperbolicPrototypicalLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, query_emb, prototype_emb, targets, logit_scale=100.0, c=1.0):
        # query: [B, D+1], prototype: [N, D+1]
        q = query_emb.unsqueeze(1)  # [B, 1, D+1]
        p = prototype_emb.unsqueeze(0)  # [1, N, D+1]
        
        # Calculate Hyperbolic Distance
        # [Safe Guard] c must be positive
        c = torch.clamp(c, min=1e-5)
        dists = LorentzMath.hyp_distance(q, p, c=c)
        
        # Logits = -distance * scale
        logits = -dists * logit_scale
        
        return self.loss_fn(logits, targets)

class EntailmentConeLoss(nn.Module):
    """
    Hierarchical Entailment Cone Loss.
    """
    def __init__(self, min_radius=0.1, margin=0.01, aperture_scale=1.2):
        super(EntailmentConeLoss, self).__init__()
        self.min_radius = min_radius
        self.margin = margin
        self.aperture_scale = aperture_scale 

    def forward(self, child_emb, parent_emb, c=1.0):
        c = torch.clamp(c, min=1e-5)
        angle = LorentzMath.oxy_angle(parent_emb, child_emb, c=c)
        dynamic_aperture = LorentzMath.half_aperture(parent_emb, c=c, min_radius=self.min_radius)
        target_aperture = self.aperture_scale * dynamic_aperture
        cone_loss = torch.clamp(angle - target_aperture + self.margin, min=0.0)
        return cone_loss.mean()

class H2EMTotalLoss(nn.Module):
    """
    H2EM Paper Eq. (16) Wrapper with HARD NEGATIVE MINING (Eq. 14).
    """
    def __init__(self, beta1=1.0, beta2=0.1, beta3=0.5, hard_neg_w=3.0):
        super(H2EMTotalLoss, self).__init__()
        self.beta1 = beta1   # Weight for L_DA
        self.beta2 = beta2   # Weight for L_TE
        self.beta3 = beta3   # Weight for L_prim
        self.hard_neg_w = hard_neg_w # Weight w in Eq. 14 (default 3.0)
        
        self.loss_cls = HyperbolicPrototypicalLoss()
        self.loss_cone = EntailmentConeLoss(min_radius=0.1, margin=0.01, aperture_scale=1.2)

    def hard_negative_mining(self, logits, targets, p2v_map, p2o_map):
        """
        Implements H2EM Eq. 14: Discriminative Alignment Loss with Hard Negative Mining.
        Instead of modifying the loss function formula directly, we add log(w) to the logits 
        of hard negatives. This is mathematically equivalent to weighting them by w in the denominator.
        
        Hard Negatives: Compositions sharing the same state OR same object, excluding self.
        """
        B, NumPairs = logits.shape
        device = logits.device
        
        # 1. Get State and Object IDs for the current batch targets
        # batch_states: [B], batch_objects: [B]
        batch_states = p2v_map[targets]
        batch_objects = p2o_map[targets]
        
        # 2. Broadcast to create masks against all pairs
        # all_states: [1, NumPairs]
        all_states = p2v_map.unsqueeze(0)
        all_objects = p2o_map.unsqueeze(0)
        
        # 3. Identify Hard Negatives
        # Mask: [B, NumPairs]
        # Condition: (Same State OR Same Object)
        is_hard = (all_states == batch_states.unsqueeze(1)) | (all_objects == batch_objects.unsqueeze(1))
        
        # 4. Exclude the Positive (Target) itself
        # We don't want to penalize the correct class
        target_mask = torch.arange(NumPairs, device=device).unsqueeze(0) == targets.unsqueeze(1)
        is_hard = is_hard & (~target_mask)
        
        # 5. Apply Weighting
        # Adding log(w) to logit is equivalent to multiplying exp(logit) by w
        # Safe guard: w must be > 0.
        w_log = math.log(max(self.hard_neg_w, 1.0))
        
        # Clone logits to avoid in-place modification errors in backward pass
        weighted_logits = logits.clone()
        weighted_logits[is_hard] += w_log
        
        return weighted_logits

    def forward(self, out, batch_verb, batch_obj, batch_target, p2v_map, p2o_map, v2cv_map, o2co_map):
        if not isinstance(out, dict):
            raise TypeError(f"[H2EMTotalLoss] Expected 'out' to be a dict, got {type(out)}.")

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
        
        logit_scale = out['logit_scale']
        current_c = out['curvature']

        # -----------------------------------------------------------
        # A. Primitive Auxiliary Loss (L_s + L_o)
        # -----------------------------------------------------------
        L_s = self.loss_cls(v_feat_hyp, verb_text_hyp, batch_verb, logit_scale, c=current_c)
        L_o = self.loss_cls(o_feat_hyp, obj_text_hyp, batch_obj, logit_scale, c=current_c)
        L_prim = L_s + L_o

        # -----------------------------------------------------------
        # B. Discriminative Alignment Loss (L_DA) with Hard Negative Mining
        # -----------------------------------------------------------
        # Base Logits (Additive Fusion)
        pair_logits = verb_logits[:, p2v_map] + obj_logits[:, p2o_map]
        
        # [CRITICAL FIX] Apply Hard Negative Mining
        weighted_pair_logits = self.hard_negative_mining(pair_logits, batch_target, p2v_map, p2o_map)
        
        L_DA = F.cross_entropy(weighted_pair_logits, batch_target)

        # -----------------------------------------------------------
        # C. Taxonomic Entailment Loss (L_TE)
        # -----------------------------------------------------------
        loss_h1 = self.loss_cone(child_emb=verb_text_hyp, parent_emb=coarse_verb_hyp[v2cv_map], c=current_c)
        loss_h2 = self.loss_cone(child_emb=obj_text_hyp, parent_emb=coarse_obj_hyp[o2co_map], c=current_c)
        loss_h3 = self.loss_cone(child_emb=comp_hyp_v, parent_emb=verb_text_hyp[p2v_map], c=current_c)
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