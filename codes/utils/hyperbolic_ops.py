"""
Implementation of common operations for the Lorentz model of hyperbolic geometry.
Optimized and stabilized version based on HyCoCLIP (ICLR 2025).

[Merged Version] Combines HyCoCLIP's robust math with C2C's Entailment Cone logic.
"""
import math
import torch
import torch.nn.functional as F

# =========================================================================
# Part 1: Robust Core Functions from HyCoCLIP (Stable Math)
# =========================================================================

def _exp_map0(x, curv=1.0, eps=1e-8):
    """
    Stable exponential map at origin.
    Input x: Euclidean tangent vector [..., D]
    Output: Lorentz vector [..., D+1]
    """
    # Norm of vector in tangent space
    x_norm = torch.norm(x, dim=-1, keepdim=True)
    
    # Scale by sqrt(c)
    rc_xnorm = (curv**0.5) * x_norm

    # Calculate scale factor: sinh(r)/r
    # [SAFETY] Clamp to prevent overflow in sinh/cosh (Max ~88 for float32)
    sinh_input = torch.clamp(rc_xnorm, min=eps, max=50.0) 
    
    scale_factor = torch.sinh(sinh_input) / torch.clamp(rc_xnorm, min=eps)
    
    # Space components
    x_space = scale_factor * x
    
    # Time component: x0 = cosh(sqrt(c) * |x|)
    x0 = torch.cosh(sinh_input)
    
    # Concatenate time (x0) and space (x_space)
    return torch.cat([x0, x_space], dim=-1)

def _dist_from_inner(inner, curv=1.0, eps=1e-7):
    """
    Calculate distance from inner product.
    d(x,y) = 1/sqrt(c) * acosh(-c * <x,y>)
    """
    # -c * <x,y> should be >= 1.0
    val = -curv * inner
    
    # [CRITICAL FIX] The Anti-NaN Clamp
    # Even if float error makes val 0.9999, we force it to 1.0000001
    val = torch.clamp(val, min=1.0 + eps)
    
    dist = torch.acosh(val) / (curv**0.5)
    return dist

# =========================================================================
# Part 2: LorentzMath Adapter (The Interface)
# =========================================================================

class LorentzMath:
    """
    Wrapper class to maintain compatibility with custom_clip_c2c and loss.py.
    """
    
    @staticmethod
    def exp_map_0(x, c=1.0):
        """
        Maps Euclidean features x to Hyperbolic features z (with time dim).
        """
        return _exp_map0(x, curv=c)
        
    @staticmethod
    def hyp_distance(x, y, c=1.0, keepdim=False):
        """
        Calculates hyperbolic distance with auto-broadcasting.
        Input x, y: [..., D+1]
        """
        # 1. Direct Inner Product Calculation for D+1 vectors
        # <x, y>_L = -x0*y0 + x1*y1 + ...
        prod = x * y
        time_prod = -prod[..., 0:1]
        space_prod = torch.sum(prod[..., 1:], dim=-1, keepdim=True)
        inner = time_prod + space_prod
        
        # 2. Distance Calculation with stability check
        dist = _dist_from_inner(inner, curv=c)
        
        if not keepdim:
            dist = dist.squeeze(-1)
        return dist

    @staticmethod
    def lorentz_product(x, y, keepdim=False):
        """
        Raw inner product for d+1 vectors. Used for debugging or advanced loss.
        """
        prod = x * y
        res = -prod[..., 0:1] + torch.sum(prod[..., 1:], dim=-1, keepdim=True)
        if not keepdim:
            res = res.squeeze(-1)
        return res

    # =====================================================================
    # [OPTIMIZED] Helper functions for Cone Loss
    # =====================================================================
    
    @staticmethod
    def half_aperture(x, c=1.0, min_radius=0.1):
        """
        Calculate K (aperture) for Entailment Cone.
        Approximation: sin(K) ~= min_radius / r_hyperbolic
        """
        x_space = x[..., 1:]
        eps = 1e-8
        
        # Using space norm as a proxy for radial distance in tangent space
        norm_x = torch.norm(x_space, dim=-1)
        
        # Formula adapted for stability
        # K = asin( 2*delta / (norm * sqrt(c)) )
        denom = norm_x * (c**0.5) + eps
        asin_input = (2 * min_radius) / denom
        
        # [SAFETY] Clamp for asin domain [-1, 1]
        asin_input = torch.clamp(asin_input, min=-1.0+eps, max=1.0-eps)
        return torch.asin(asin_input)

    @staticmethod
    def oxy_angle(x, y, c=1.0):
        """
        Calculate Angle at Origin (O) between x and y.
        In Lorentz model, this is simply the Euclidean angle between space components.
        This is much more stable than the complex hyperbolic formula.
        """
        # We only need space components [..., 1:]
        x_space = x[..., 1:]
        y_space = y[..., 1:]
        
        # Robust Euclidean Angle
        norm_x = torch.norm(x_space, p=2, dim=-1, keepdim=True)
        norm_y = torch.norm(y_space, p=2, dim=-1, keepdim=True)
        
        # Avoid division by zero
        norm_x = torch.clamp(norm_x, min=1e-6)
        norm_y = torch.clamp(norm_y, min=1e-6)
        
        dot = (x_space * y_space).sum(dim=-1, keepdim=True)
        cosine = dot / (norm_x * norm_y)
        
        # [SAFETY] Clamp for acos domain [-1, 1]
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)
        
        return torch.acos(cosine).squeeze(-1)