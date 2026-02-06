"""
Implementation of common operations for the Lorentz model of hyperbolic geometry.
Modified from HyCoCLIP (originally from Facebook Research meru).

This model represents a hyperbolic space of `d` dimensions on the upper-half of
a two-sheeted hyperboloid in a Euclidean space of `(d+1)` dimensions.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor

def pairwise_inner(x: Tensor, y: Tensor, curv: float | Tensor = 1.0):
    """
    Compute pairwise Lorentzian inner product between input vectors.
    Minkowski inner product: <x, y> = x_space * y_space - x_time * y_time

    Args:
        x: Tensor of shape `(B1, D)` giving a space components of a batch
            of vectors on the hyperboloid.
        y: Tensor of shape `(B2, D)` giving a space components of another
            batch of points on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature (c).

    Returns:
        Tensor of shape `(B1, B2)` giving pairwise Lorentzian inner product.
    """
    # Calculate time component based on constraint: -t^2 + x^2 = -1/c  => t = sqrt(1/c + x^2)
    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True))
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1, keepdim=True))
    
    # Lorentzian inner product
    xyl = x @ y.T - x_time @ y_time.T
    return xyl


def pairwise_dist(
    x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8
) -> Tensor:
    """
    Compute the pairwise geodesic distance between two batches of points on
    the hyperboloid.
    Distance d(x,y) = 1/sqrt(c) * arccosh(-c * <x, y>_L)

    Args:
        x: Tensor of shape `(B1, D)` space components.
        y: Tensor of shape `(B2, D)` space components.
        curv: Positive scalar (c).
        eps: Small float number to avoid numerical instability.
    """
    # Ensure numerical stability in arc-cosh by clamping input.
    # The inner product <x,y>_L is always <= -1/c. 
    # Thus -c * <x,y>_L is always >= 1.
    c_xyl = -curv * pairwise_inner(x, y, curv)
    
    # Clamp min=1+eps to avoid NaN gradients at distance 0
    _distance = torch.acosh(torch.clamp(c_xyl, min=1 + eps))
    return _distance / (curv**0.5)


def exp_map0(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8) -> Tensor:
    """
    Map points from the tangent space at the vertex (origin) of hyperboloid, 
    on to the hyperboloid. (Exponential Map at Origin).
    
    This is used to project Euclidean feature vectors into Hyperbolic space.

    Args:
        x: Tensor of shape `(B, D)` Euclidean vectors (tangent vectors at origin).
        curv: Positive scalar (c).
        eps: Small float number to avoid division by zero.
    """
    # Norm of the tangent vector scaled by sqrt(c)
    rc_xnorm = curv**0.5 * torch.norm(x, dim=-1, keepdim=True)

    # Ensure numerical stability in sinh by clamping input.
    # Formula: exp_0(v) = sinh(sqrt(c)|v|) * (v / (sqrt(c)|v|))
    # (Note: Time component is handled implicitly by the manifold constraint later)
    sinh_input = torch.clamp(rc_xnorm, min=eps, max=math.asinh(2**15))
    _output = torch.sinh(sinh_input) * x / torch.clamp(rc_xnorm, min=eps)
    return _output


def half_aperture(
    x: Tensor, curv: float | Tensor = 1.0, min_radius: float = 0.1, eps: float = 1e-8
) -> Tensor:
    """
    Compute the half aperture angle of the entailment cone formed by vectors on
    the hyperboloid.
    
    Used for: Hierarchical Entailment Loss.

    Args:
        x: Tensor of shape `(B, D)` space components on hyperboloid.
        curv: Positive scalar (c).
        min_radius: Minimum radius for stability.
    """
    # Formula: sin(theta) = K / r  (approximation for entailment cone)
    # asin_input = 2 * min_radius / (sqrt(c) * ||x||)
    asin_input = 2 * min_radius / (torch.norm(x, dim=-1) * curv**0.5 + eps)
    _half_aperture = torch.asin(torch.clamp(asin_input, min=-1 + eps, max=1 - eps))

    return _half_aperture


def oxy_angle(x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8):
    """
    Given two vectors `x` and `y` on the hyperboloid, compute the exterior
    angle at `x` in the hyperbolic triangle `Oxy` where `O` is the origin.

    Used for: Hierarchical Entailment Loss (determining if y is 'inside' x's cone).

    Args:
        x: Tensor of shape `(B, D)` (Parent/Coarse candidate).
        y: Tensor of same shape as `x` (Child/Fine candidate).
        curv: Positive scalar (c).
    """
    # Calculate time components: t = sqrt(1/c + ||x||^2)
    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1))
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1))

    # Calculate Lorentzian inner product * curvature (only diagonal elements needed here)
    # <x, y>_L = x.y - x_t*y_t
    c_xyl = curv * (torch.sum(x * y, dim=-1) - x_time * y_time)

    # Formula derived from Hyperbolic law of cosines
    acos_numer = y_time + c_xyl * x_time
    acos_denom = torch.sqrt(torch.clamp(c_xyl**2 - 1, min=eps))

    acos_input = acos_numer / (torch.norm(x, dim=-1) * acos_denom + eps)
    _angle = torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))

    return _angle