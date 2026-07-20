"""Training losses for the span-typing bi-encoder."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def bce_loss(logits: torch.Tensor, targets: torch.Tensor,
             pos_weight: float = 1.0) -> torch.Tensor:
    """Multi-label BCE with a scalar positive-class weight (long-tail up-weighting)."""
    pw = torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)


def infonce_loss(logits: torch.Tensor, targets: torch.Tensor,
                 fneg_mask: torch.Tensor | None = None) -> torch.Tensor:
    """InfoNCE with in-batch label negatives.

    Candidates for each example are the labels present anywhere in the batch
    (union of gold sets) — the standard bi-encoder in-batch-negatives setup,
    which C1 showed carries false negatives for >90% of examples at B=32.
    Each gold label is a positive; the loss is the mean over gold labels of
    -log softmax(candidate logits). The GeometryHead's learned scale acts as
    the temperature.

    `fneg_mask` [B, K] (1 = label is an ancestor-of-gold for that example but
    not itself gold) removes those entries from the candidate set — the
    taxonomy-masked variant that deletes the conflicting signal. Positives are
    never masked (the mask excludes gold by construction).
    """
    present = targets.sum(0) > 0                      # [K] in-batch label set
    lc = logits[:, present]                           # [B, C]
    tc = targets[:, present]
    if fneg_mask is not None:
        lc = lc.masked_fill(fneg_mask[:, present] > 0.5, float("-inf"))
    log_probs = lc - torch.logsumexp(lc, dim=1, keepdim=True)
    pos = tc > 0.5
    per_ex = (log_probs * pos).sum(1) / pos.sum(1).clamp_min(1)
    valid = pos.any(1)
    if not valid.any():
        return logits.sum() * 0.0
    return -(per_ex[valid]).mean()


def soft_bce_loss(logits: torch.Tensor, targets: torch.Tensor,
                  closure_targets: torch.Tensor, alpha: float = 0.5,
                  pos_weight: float = 1.0) -> torch.Tensor:
    """BCE with soft ancestor targets: gold=1, ancestors-of-gold=alpha, rest=0.

    The supervision-side patch for the false-negative conflict: instead of a
    hard 0, a true-but-unannotated ancestor gets partial credit alpha.
    """
    soft = targets + alpha * (closure_targets - targets)
    pw = torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, soft, pos_weight=pw)


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
