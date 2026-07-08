"""Evaluation metrics for multi-label span typing.

Standard UFET-style metrics (Choi et al. 2018): strict exact-match, macro
(per-example P/R/F1 averaged), micro (aggregate TP/FP/FN). Plus two additions
central to this project:

  - tail-stratified F1: micro F1 within head / mid / tail frequency buckets,
    to test the hypothesis that hyperbolic geometry helps rare types.
  - hierarchical F1: gives partial credit when a predicted type is an ancestor
    of a gold type (predicting "person" for gold "actor" is not fully wrong).
"""
from __future__ import annotations

import numpy as np


def compute_metrics(preds: np.ndarray, golds: np.ndarray) -> dict[str, float]:
    """preds, golds: [N, K] binary arrays."""
    strict = float((preds == golds).all(axis=1).mean())

    tp_ex = (preds * golds).sum(axis=1)
    fp_ex = (preds * (1 - golds)).sum(axis=1)
    fn_ex = ((1 - preds) * golds).sum(axis=1)

    p_ex = tp_ex / np.maximum(tp_ex + fp_ex, 1)
    r_ex = tp_ex / np.maximum(tp_ex + fn_ex, 1)
    f1_ex = 2 * p_ex * r_ex / np.maximum(p_ex + r_ex, 1e-9)

    tp, fp, fn = tp_ex.sum(), fp_ex.sum(), fn_ex.sum()
    micro_p = tp / max(tp + fp, 1)
    micro_r = tp / max(tp + fn, 1)
    micro_f1 = 2 * micro_p * micro_r / max(micro_p + micro_r, 1e-9)

    return {
        "strict": strict,
        "macro_p": float(p_ex.mean()), "macro_r": float(r_ex.mean()),
        "macro_f1": float(f1_ex.mean()),
        "micro_p": float(micro_p), "micro_r": float(micro_r),
        "micro_f1": float(micro_f1),
    }


def tail_stratified_f1(preds: np.ndarray, golds: np.ndarray,
                       type_counts: np.ndarray,
                       tail_thresh: int = 5, head_thresh: int = 100) -> dict[str, float]:
    """Micro F1 restricted to type columns in each frequency bucket.

    type_counts[k] = training frequency of type k.
    """
    buckets = {
        "head": type_counts >= head_thresh,
        "mid": (type_counts < head_thresh) & (type_counts > tail_thresh),
        "tail": type_counts <= tail_thresh,
    }
    out = {}
    for name, mask in buckets.items():
        p, g = preds[:, mask], golds[:, mask]
        tp = (p * g).sum()
        fp = (p * (1 - g)).sum()
        fn = ((1 - p) * g).sum()
        pr = tp / max(tp + fp, 1)
        rc = tp / max(tp + fn, 1)
        out[f"{name}_f1"] = float(2 * pr * rc / max(pr + rc, 1e-9))
        out[f"{name}_support"] = int(g.sum())
    return out


def hierarchical_f1(preds: np.ndarray, golds: np.ndarray,
                    type_list: list[str], taxonomy) -> dict[str, float]:
    """Micro P/R/F1 where each label is expanded to its ancestor closure.

    Predicting an ancestor of a gold type then counts as a (partial) hit,
    following Kiritchenko et al.'s hierarchical evaluation.
    """
    K = len(type_list)
    type2idx = {t: i for i, t in enumerate(type_list)}
    # closure matrix C[k] = multi-hot of {k} u ancestors(k)
    C = np.eye(K, dtype=np.float32)
    for k, t in enumerate(type_list):
        for anc in taxonomy.ancestors(t):
            j = type2idx.get(anc)
            if j is not None:
                C[k, j] = 1.0

    p_aug = (preds @ C > 0).astype(np.float32)
    g_aug = (golds @ C > 0).astype(np.float32)
    tp = (p_aug * g_aug).sum()
    fp = (p_aug * (1 - g_aug)).sum()
    fn = ((1 - p_aug) * g_aug).sum()
    pr = tp / max(tp + fp, 1)
    rc = tp / max(tp + fn, 1)
    return {
        "h_precision": float(pr), "h_recall": float(rc),
        "h_f1": float(2 * pr * rc / max(pr + rc, 1e-9)),
    }
