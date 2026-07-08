"""Hyperbolic bi-encoder for long-tail fine-grained entity typing."""
from .geometry import GeometryHead, expmap0, poincare_distance
from .model import BiEncoderTyper
from .data import SpanTypingDataset, build_type_vocab, collate_fn, normalize_record
from .taxonomy import Taxonomy, propagate_ancestors
from .losses import bce_loss, distance_ranking_loss
from .metrics import compute_metrics, tail_stratified_f1, hierarchical_f1
from .trainer import train, evaluate

__all__ = [
    "GeometryHead", "expmap0", "poincare_distance", "BiEncoderTyper",
    "SpanTypingDataset", "build_type_vocab", "collate_fn", "normalize_record",
    "Taxonomy", "propagate_ancestors", "bce_loss", "distance_ranking_loss",
    "compute_metrics", "tail_stratified_f1", "hierarchical_f1", "train", "evaluate",
]
