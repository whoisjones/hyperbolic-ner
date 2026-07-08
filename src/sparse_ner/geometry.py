"""Output-space geometries for the bi-encoder.

Two geometries share one interface: given a mention vector `m` and a set of
label vectors `L` (both raw, in tangent/Euclidean coordinates), return a
similarity **logit** per label (higher = more likely the label applies).

- Euclidean: logit = scale * (m . l) + bias
- Hyperbolic: map both to the Poincare ball via the exponential map at the
  origin, then logit = -scale * d_ball(m, l) + bias

Keeping label embeddings as ordinary (tangent-space) parameters and mapping to
the ball only inside the forward pass means a standard optimizer (Adam) works;
no Riemannian optimizer is required.

The Poincare ball has curvature -c (c > 0). All ops are batched and numerically
guarded so points never reach the boundary.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-5


def _project(x: torch.Tensor, c: float) -> torch.Tensor:
    """Clamp points to stay strictly inside the ball of radius 1/sqrt(c)."""
    max_norm = (1.0 - _EPS) / (c ** 0.5)
    norm = x.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    cond = norm > max_norm
    projected = x / norm * max_norm
    return torch.where(cond, projected, x)


def expmap0(v: torch.Tensor, c: float) -> torch.Tensor:
    """Exponential map at the origin: tangent vector -> point on the ball."""
    sqrt_c = c ** 0.5
    norm = v.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    coef = torch.tanh(sqrt_c * norm) / (sqrt_c * norm)
    return _project(coef * v, c)


def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    """Mobius addition x (+)_c y, with broadcasting over leading dims."""
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    den = 1 + 2 * c * xy + (c * c) * x2 * y2
    return num / den.clamp_min(_EPS)


def poincare_distance(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    """Geodesic distance on the Poincare ball between broadcastable x, y."""
    sqrt_c = c ** 0.5
    diff = mobius_add(-x, y, c)
    norm = diff.norm(dim=-1).clamp(min=_EPS, max=(1.0 - _EPS) / sqrt_c)
    return (2.0 / sqrt_c) * torch.atanh(sqrt_c * norm)


class GeometryHead(nn.Module):
    """Scores a batch of mention vectors against all label vectors.

    forward(mentions [B, D], labels [K, D]) -> logits [B, K]
    """

    def __init__(self, geometry: str = "euclidean", c: float = 1.0,
                 learn_scale: bool = True, init_scale: float = 1.0):
        super().__init__()
        assert geometry in ("euclidean", "hyperbolic")
        self.geometry = geometry
        self.c = c
        self.log_scale = nn.Parameter(torch.tensor(float(init_scale)).log(),
                                      requires_grad=learn_scale)
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, mentions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        scale = self.log_scale.exp()
        if self.geometry == "euclidean":
            # cosine-style: L2-normalize so scale controls temperature
            m = F.normalize(mentions, dim=-1)
            l = F.normalize(labels, dim=-1)
            return scale * (m @ l.t()) + self.bias
        # hyperbolic: map both to the ball, then negative distance
        m = expmap0(mentions, self.c)                    # [B, D]
        l = expmap0(labels, self.c)                      # [K, D]
        dist = poincare_distance(m.unsqueeze(1), l.unsqueeze(0), self.c)  # [B, K]
        return -scale * dist + self.bias
