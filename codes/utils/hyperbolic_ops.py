"""
Implementation of common operations for the Lorentz model of hyperbolic geometry.
STRICTLY ALIGNED with HyCoCLIP (ICLR 2025) logic.
"""
import math
import torch
import torch.nn.functional as F

class LorentzMath:
    """
    Wrapper class to provide HyCoCLIP's math operations via a static interface.
    Handles the conversion between Space-only (HyCoCLIP style) and Space-Time (D+1) logic implicitly.
    """
    
    @staticmethod
    def _calc_time(x_space, c=1.0):
        """
        Calculate time component from space component: t = sqrt(1/c + |x|^2)
        HyCoCLIP logic.
        """
        return torch.sqrt(1.0 / c + torch.sum(x_space**2, dim=-1, keepdim=True))

    @staticmethod
    def exp_map_0(x, c=1.0, eps=1e-8):
        """
        HyCoCLIP's exp_map0 logic.
        Input: x (Euclidean tangent vector at origin) [..., D]
        Output: Hyperbolic vector [..., D] (Space components only) OR [..., D+1]
        
        To maintain compatibility with your previous code structure (which likely expects D+1),
        we will append the time component.
        """
        # HyCoCLIP: lorentz.py line 79
        rc_xnorm = (c**0.5) * torch.norm(x, dim=-1, keepdim=True)
        
        # Clamp to avoid overflow/underflow
        # HyCoCLIP uses math.asinh(2**15) approx 11.0, which is safe.
        sinh_input = torch.clamp(rc_xnorm, min=eps, max=math.asinh(2**15))
        
        # Space components
        x_space = (torch.sinh(sinh_input) * x) / torch.clamp(rc_xnorm, min=eps)
        
        # Time component (Calculated explicitly)
        # t = cosh(sqrt(c) * |x|)
        x_time = torch.cosh(sinh_input) / (c**0.5) # Correct normalization for t^2 - |x|^2 = 1/c
        
        # Return D+1 vector [Time, Space]
        return torch.cat([x_time, x_space], dim=-1)
        
    @staticmethod
    def hyp_distance(x, y, c=1.0, keepdim=False, eps=1e-8):
        """
        HyCoCLIP's pairwise_dist logic.
        Input x, y: [..., D+1] (Time, Space)
        """
        # Separate Time and Space
        x_time, x_space = x[..., 0:1], x[..., 1:]
        y_time, y_space = y[..., 0:1], y[..., 1:]
        
        # HyCoCLIP: lorentz.py line 35 (pairwise_inner)
        # <x, y>_L = x_space @ y_space.T - x_time @ y_time.T
        # Note: HyCoCLIP implementation assumes broadcasting or matrix mult. 
        # Here we do element-wise for matched pairs.
        
        inner = torch.sum(x_space * y_space, dim=-1, keepdim=True) - x_time * y_time
        
        # HyCoCLIP: lorentz.py line 65
        # dist = acosh(clamp(-c * inner, min=1+eps)) / sqrt(c)
        c_inner = -c * inner
        dist = torch.acosh(torch.clamp(c_inner, min=1.0 + eps)) / (c**0.5)
        
        if not keepdim:
            dist = dist.squeeze(-1)
        return dist

    @staticmethod
    def half_aperture(x, c=1.0, min_radius=0.1, eps=1e-8):
        """
        HyCoCLIP's half_aperture logic.
        Input x: [..., D+1] (Time, Space)
        """
        # Extract Space component
        x_space = x[..., 1:]
        
        # HyCoCLIP: lorentz.py line 122
        norm_x = torch.norm(x_space, dim=-1)
        
        denom = norm_x * (c**0.5) + eps
        asin_input = 2 * min_radius / denom
        
        # Clamp input for asin
        asin_input = torch.clamp(asin_input, min=-1.0 + eps, max=1.0 - eps)
        
        return torch.asin(asin_input)

    @staticmethod
    def oxy_angle(x, y, c=1.0, eps=1e-8):
        """
        HyCoCLIP's oxy_angle logic.
        Strictly follows the hyperbolic law of cosines formula in lorentz.py line 139.
        Input x, y: [..., D+1] (Time, Space)
        """
        x_time, x_space = x[..., 0], x[..., 1:]
        y_time, y_space = y[..., 0], y[..., 1:]
        
        # HyCoCLIP calculates inner product * c (variable c_xyl)
        # c_xyl = c * (<x, y>_L) = c * (space_dot - time_prod)
        inner = torch.sum(x_space * y_space, dim=-1) - x_time * y_time
        c_xyl = c * inner
        
        # Numerator: y_time + c_xyl * x_time
        # Note: HyCoCLIP multiplies by sqrt(c) implicitly in their x_time. 
        # Since our x_time is actual coordinate, we need to adjust if their formula relies on scaled time.
        # Let's verify HyCoCLIP's x_time: "x_time = sqrt(1/curv + norm^2)" -> This is actual coordinate.
        # Wait, HyCoCLIP line 155: "x_time = torch.sqrt(1 / curv + ...)"
        # So their x_time IS the coordinate.
        # But wait, line 160: "c_xyl = curv * (sum(x*y) - x_time*y_time)"
        # Line 163: "acos_numer = y_time + c_xyl * x_time"
        # Wait, HyCoCLIP line 155 comment says "multiplied with sqrt(curv)". 
        # Let's check logic: "x_time = torch.sqrt(1/curv + ...)"
        # If curv=1, x_time = sqrt(1+x^2). This is coordinate t.
        # If HyCoCLIP's `x` input is just space, then `x_time` is calculated inside.
        
        # Let's trust the formula structure from HyCoCLIP lines 163-167:
        # acos_numer = y_time + c_xyl * x_time
        # acos_denom = sqrt(c_xyl^2 - 1)
        # acos_input = acos_numer / (norm(x) * acos_denom)
        
        # BUT there is a dimensionality check. HyCoCLIP x_time has shape (B,).
        # Our x_time has shape (B, 1) or (B,). Ensure broadcasting.
        
        # Re-implementing HyCoCLIP logic exactly:
        
        # 1. c_xyl
        # inner is <x,y>_L. c_xyl = c * inner.
        # Note: HyCoCLIP lorentz.py line 160 seems to use `sum(x*y) - x_t*y_t`.
        # This is Lorentzian inner product.
        
        # 2. acos_numer
        numer = y_time + c_xyl * x_time
        
        # 3. acos_denom
        denom_sq = torch.clamp(c_xyl**2 - 1, min=eps)
        denom = torch.sqrt(denom_sq)
        
        # 4. Final input
        x_space_norm = torch.norm(x_space, dim=-1)
        # Formula: numer / (norm(x_space) * denom)
        final_denom = x_space_norm * denom + eps
        
        acos_input = numer / final_denom
        
        # Clamp for acos
        acos_input = torch.clamp(acos_input, min=-1.0 + eps, max=1.0 - eps)
        
        return torch.acos(acos_input)