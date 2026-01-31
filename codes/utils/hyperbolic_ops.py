"""
Implementation of common operations for the Lorentz model of hyperbolic geometry.
Optimized and stabilized version based on HyCoCLIP (ICLR 2025).

This module serves as a robust backend for your custom_clip_c2c and loss functions.
It maintains the 'LorentzMath' interface for backward compatibility.
"""
import math
import torch
import torch.nn.functional as F

# =========================================================================
# Part 1: Robust Core Functions from HyCoCLIP
# (Internal use, stable math)
# =========================================================================

def _pairwise_inner(x, y, curv=1.0):
    """
    Robust inner product.
    Note: x, y are expected to be space components only.
    """
    x_sq = torch.sum(x**2, dim=-1, keepdim=True)
    y_sq = torch.sum(y**2, dim=-1, keepdim=True)
    
    # Calculate time components explicitly
    x_time = torch.sqrt(1.0 / curv + x_sq)
    y_time = torch.sqrt(1.0 / curv + y_sq)
    
    # Lorentz Inner Product: <x, y> = x_space * y_space - x_time * y_time
    # Handle broadcasting carefully
    if x.dim() == y.dim(): 
        # Element-wise (e.g., within same batch)
        xyl = torch.sum(x * y, dim=-1, keepdim=True) - x_time * y_time
    else:
        # Matrix multiplication (e.g., query vs prototypes)
        # Assuming x: [B, D], y: [N, D] -> [B, N]
        # x @ y.T shape: [B, N]
        # x_time @ y_time.T shape: [B, N]
        xyl = x @ y.transpose(-2, -1) - x_time @ y_time.transpose(-2, -1)
        
    return xyl

def _exp_map0(x, curv=1.0, eps=1e-8):
    """
    Stable exponential map at origin.
    x: Euclidean tangent vector at origin.
    """
    # Norm of vector in tangent space
    x_norm = torch.norm(x, dim=-1, keepdim=True)
    
    # Scale by sqrt(c)
    rc_xnorm = (curv**0.5) * x_norm

    # Calculate scale factor: sinh(r)/r
    # Use clamping to prevent overflow/underflow
    sinh_input = torch.clamp(rc_xnorm, min=eps, max=88.0) # 88.0 is safe for float32 exp
    
    scale_factor = torch.sinh(sinh_input) / torch.clamp(rc_xnorm, min=eps)
    
    # Space components
    x_space = scale_factor * x
    
    # Time component is calculated implicitly by the manifold constraint when needed,
    # BUT your code expects d+1 dimensional output.
    # So we must reconstruct the full d+1 vector here.
    
    # x0 = cosh(sqrt(c) * |x|)
    x0 = torch.cosh(sinh_input)
    
    # Concatenate time (x0) and space (x_space)
    return torch.cat([x0, x_space], dim=-1)

def _dist_from_inner(inner, curv=1.0, eps=1e-8):
    """
    Calculate distance from inner product.
    d(x,y) = 1/sqrt(c) * acosh(-c * <x,y>)
    """
    # -c * <x,y> should be >= 1
    val = -curv * inner
    
    # Crucial stability clamp from HyCoCLIP
    val = torch.clamp(val, min=1.0 + eps)
    
    dist = torch.acosh(val) / (curv**0.5)
    return dist

# =========================================================================
# Part 2: Adapter Class (Compatible with your existing code)
# =========================================================================

class LorentzMath:
    """
    Wrapper class to maintain compatibility with 'LorentzMath.exp_map_0' calls.
    Now backed by more stable math.
    """
    
    @staticmethod
    def exp_map_0(x, c=1.0):
        """
        Maps Euclidean features x to Hyperbolic features z (with time dim).
        Input x: [..., d]
        Output z: [..., d+1]
        """
        return _exp_map0(x, curv=c)
        
    @staticmethod
    def hyp_distance(x, y, c=1.0, keepdim=False):
        """
        Calculates hyperbolic distance.
        Input x, y: [..., d+1] (Full Hyperbolic Vectors)
        """
        # Note: HyCoCLIP functions work on SPACE components usually.
        # But your code passes d+1 vectors. We need to handle this.
        
        # 1. Direct Inner Product Calculation for d+1 vectors
        # <x, y> = -x0*y0 + x1*y1 + ...
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
        Raw inner product for d+1 vectors.
        """
        prod = x * y
        res = -prod[..., 0:1] + torch.sum(prod[..., 1:], dim=-1, keepdim=True)
        if not keepdim:
            res = res.squeeze(-1)
        return res

    # =====================================================================
    # [NEW] Helper functions for Cone Loss (HyCoCLIP Style)
    # Use these in your loss.py for better stability
    # =====================================================================
    
    @staticmethod
    def half_aperture(x, c=1.0, min_radius=0.1):
        """
        Calculate K (aperture) for Entailment Cone.
        Input x: [..., d+1] (Full vector) -> We only use space components [..., 1:]
        """
        x_space = x[..., 1:]
        eps = 1e-8
        
        norm_x = torch.norm(x_space, dim=-1)
        asin_input = 2 * min_radius / (norm_x * (c**0.5) + eps)
        
        # Clamp for stability
        asin_input = torch.clamp(asin_input, min=-1+eps, max=1-eps)
        return torch.asin(asin_input)

    @staticmethod
    def oxy_angle(x, y, c=1.0):
        """
        Calculate Angle between x and y in Hyperbolic Space.
        Input x, y: [..., d+1]
        """
        eps = 1e-8
        
        # We need space and time components explicitly
        x_time = x[..., 0]
        x_space = x[..., 1:]
        y_time = y[..., 0]
        y_space = y[..., 1:]
        
        # Inner product of space components
        xy_space_dot = torch.sum(x_space * y_space, dim=-1)
        
        # Full Lorentz Inner Product * c
        # <x,y>_L = x_space.y_space - x_time.y_time
        # c_xyl = c * ( <x,y>_L )
        c_xyl = c * (xy_space_dot - x_time * y_time)
        
        # Numerator: y_time + c_xyl * x_time
        numer = y_time + c_xyl * x_time
        
        # Denominator
        denom_sq = torch.clamp(c_xyl**2 - 1, min=eps)
        denom = torch.norm(x_space, dim=-1) * torch.sqrt(denom_sq) + eps
        
        acos_input = numer / denom
        acos_input = torch.clamp(acos_input, min=-1+eps, max=1-eps)
        
        return torch.acos(acos_input)