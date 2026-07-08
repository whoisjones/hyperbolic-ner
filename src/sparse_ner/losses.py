"""Training losses for the span-typing bi-encoder."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def bce_loss(logits: torch.Tensor, targets: torch.Tensor,
             pos_weight: float = 1.0) -> torch.Tensor:
    """Multi-label BCE with a scalar positive-class weight (long-tail up-weighting)."""
    pw = torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)


def distance_ranking_loss(logits: torch.Tensor, targets: torch.Tensor,
                          margin: float = 0.1) -> torch.Tensor:
    """Rank every positive label above every negative label per example.

    Geometry-agnostic: `logits` are similarity scores (higher = closer). For the
    hyperbolic head these are negative distances, so this pulls the mention
    toward its gold types and pushes it from the rest.
    """
    losses = []
    for b in range(logits.size(0)):
        pos = logits[b][targets[b] > 0.5]
        neg = logits[b][targets[b] <= 0.5]
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        # [P, N] pairwise hinge
        diff = margin - (pos.unsqueeze(1) - neg.unsqueeze(0))
        losses.append(F.relu(diff).mean())
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()
