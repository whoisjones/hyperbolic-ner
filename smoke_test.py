"""Smoke test: validates geometry math, model forward, data loading, and one
training step on a tiny slice of real data. Run before any real experiment.

    python smoke_test.py
"""
from __future__ import annotations

import torch
from transformers import AutoTokenizer

from sparse_ner import (BiEncoderTyper, SpanTypingDataset, Taxonomy,
                        build_type_vocab, collate_fn, evaluate,
                        hierarchical_f1, poincare_distance, expmap0)
from sparse_ner.metrics import tail_stratified_f1
import numpy as np

ENC = "bert-base-uncased"  # cached locally with a fast tokenizer
UFET = "/vol/tmp/goldejon/sparse_ner/data/ufet_crowd_train.jsonl"


def test_geometry():
    x = torch.randn(4, 8) * 0.3
    y = torch.randn(4, 8) * 0.3
    px, py = expmap0(x, 1.0), expmap0(y, 1.0)
    d = poincare_distance(px, py, 1.0)
    assert d.shape == (4,)
    assert (d >= 0).all(), "distances must be non-negative"
    assert torch.allclose(poincare_distance(px, px, 1.0),
                          torch.zeros(4), atol=1e-3), "d(x,x)=0"
    # points stay inside the unit ball
    assert (px.norm(dim=-1) < 1.0).all()
    print("[ok] geometry: distances non-negative, self-distance ~0, in-ball")


def test_pipeline():
    tok = AutoTokenizer.from_pretrained(ENC)
    vocab, counts = build_type_vocab([UFET])
    print(f"[ok] vocab: {len(vocab)} types from UFET crowd train")
    type2idx = {t: i for i, t in enumerate(vocab)}

    ds = SpanTypingDataset(UFET, tok, type2idx, max_records=200)
    print(f"[ok] dataset: {len(ds)} span examples")
    assert len(ds) > 0

    for geom in ("euclidean", "hyperbolic"):
        model = BiEncoderTyper(ENC, vocab, dim=32, geometry=geom,
                               init_labels_from_names=True)
        batch = collate_fn([ds[i] for i in range(8)])
        logits = model(batch["input_ids"], batch["attention_mask"],
                       batch["mention_starts"], batch["mention_ends"])
        assert logits.shape == (8, len(vocab))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, batch["labels"])
        loss.backward()
        assert model.mention_proj.weight.grad is not None
        print(f"[ok] {geom}: forward {tuple(logits.shape)}, loss {loss.item():.3f}, backward")

    # metrics smoke
    preds = np.random.randint(0, 2, (10, len(vocab)))
    golds = np.random.randint(0, 2, (10, len(vocab)))
    tc = np.array([counts.get(t, 0) for t in vocab])
    print("[ok] tail-strat:", {k: v for k, v in tail_stratified_f1(preds, golds, tc).items()
                               if k.endswith("f1")})


if __name__ == "__main__":
    test_geometry()
    test_pipeline()
    print("\nSMOKE TEST PASSED")
