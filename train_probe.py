"""P1 geometry probe: train one (geometry, supervision, dim) cell on UFET crowd.

Fair-comparison protocol:
  - identical encoder, data, schedule for every cell
  - decision threshold swept on dev (quantile grid over logits, targeting an
    average #predictions/example) — geometry-agnostic calibration
  - model selection: best dev macro-F1 epoch (state kept on CPU)
  - reports UFET P/R/F1 + mAP (threshold-free) + tail-stratified F1
    (train-frequency buckets) + hierarchical F1 (WordNet taxonomy)

Usage:
  python train_probe.py --geometry hyperbolic --supervision ancestor \
      --dim 64 --out results/p1/hyp-anc-d64.json
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from sparse_ner.data import SpanTypingDataset, build_type_vocab, collate_fn
from sparse_ner.losses import bce_loss
from sparse_ner.metrics import compute_metrics, hierarchical_f1, tail_stratified_f1
from sparse_ner.model import BiEncoderTyper
from sparse_ner.taxonomy import Taxonomy

DATA = "/vol/tmp/goldejon/sparse_ner/data"


def closure_matrix(vocab: list[str], tax: Taxonomy) -> torch.Tensor:
    """C[k, j] = 1 if j == k or j is an ancestor of k."""
    type2idx = {t: i for i, t in enumerate(vocab)}
    K = len(vocab)
    C = torch.eye(K)
    for k, t in enumerate(vocab):
        for anc in tax.ancestors(t):
            j = type2idx.get(anc)
            if j is not None:
                C[k, j] = 1.0
    return C


@torch.no_grad()
def collect_logits(model, ds, device, batch_size=128):
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn)
    L, G = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["attention_mask"],
                       batch["mention_starts"], batch["mention_ends"])
        L.append(logits.float().cpu().numpy())
        G.append(batch["labels"].cpu().numpy())
    return np.concatenate(L), np.concatenate(G)


def mean_average_precision(logits: np.ndarray, golds: np.ndarray) -> float:
    """Per-example AP over the label ranking, averaged (threshold-free)."""
    aps = []
    order = np.argsort(-logits, axis=1)
    for i in range(logits.shape[0]):
        gold = golds[i][order[i]]
        n_pos = int(gold.sum())
        if n_pos == 0:
            continue
        hits = np.cumsum(gold)
        prec = hits / np.arange(1, len(gold) + 1)
        aps.append(float((prec * gold).sum() / n_pos))
    return float(np.mean(aps))


def sweep_threshold(logits: np.ndarray, golds: np.ndarray) -> tuple[float, float]:
    """Pick the threshold maximizing macro-F1 on dev.

    Candidates are logit quantiles chosen so the average number of predicted
    labels per example spans ~0.5..25 — calibration-free across geometries.
    """
    N, K = logits.shape
    flat = np.sort(logits, axis=None)
    best_thr, best_f1 = None, -1.0
    for avg_preds in np.geomspace(0.5, 25.0, 60):
        q = 1.0 - avg_preds / K
        thr = float(np.quantile(flat, min(max(q, 0.0), 1.0 - 1e-9)))
        preds = (logits >= thr).astype(np.float32)
        f1 = compute_metrics(preds, golds)["macro_f1"]
        if f1 > best_f1:
            best_thr, best_f1 = thr, f1
    return best_thr, best_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", choices=["euclidean", "hyperbolic"], required=True)
    ap.add_argument("--supervision", choices=["flat", "ancestor"], required=True)
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--encoder", default="bert-base-uncased")
    ap.add_argument("--taxonomy", default="results/taxonomy/wordnet_parent.json")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr-encoder", type=float, default=2e-5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--pos-weight", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    splits = {s: f"{DATA}/ufet_crowd_{s}.jsonl" for s in ("train", "validation", "test")}
    vocab, _ = build_type_vocab(list(splits.values()))
    _, train_counts = build_type_vocab([splits["train"]])
    type2idx = {t: i for i, t in enumerate(vocab)}
    tax = Taxonomy.load(args.taxonomy)

    tok = AutoTokenizer.from_pretrained(args.encoder)
    ds = {s: SpanTypingDataset(p, tok, type2idx) for s, p in splits.items()}
    print(f"vocab={len(vocab)} | train={len(ds['train'])} dev={len(ds['validation'])} "
          f"test={len(ds['test'])}", flush=True)

    model = BiEncoderTyper(args.encoder, vocab, dim=args.dim,
                           geometry=args.geometry).to(device)

    C = closure_matrix(vocab, tax).to(device) if args.supervision == "ancestor" else None

    enc_params = list(model.encoder.parameters())
    enc_ids = {id(p) for p in enc_params}
    head_params = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([
        {"params": enc_params, "lr": args.lr_encoder},
        {"params": head_params, "lr": args.lr_head},
    ])

    loader = DataLoader(ds["train"], batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate_fn)
    best = {"dev_macro_f1": -1.0}
    best_state = None
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            targets = batch["labels"]
            if C is not None:
                targets = (targets @ C).clamp_max_(1.0)
            logits = model(batch["input_ids"], batch["attention_mask"],
                           batch["mention_starts"], batch["mention_ends"])
            loss = bce_loss(logits, targets, args.pos_weight)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        dev_logits, dev_golds = collect_logits(model, ds["validation"], device)
        thr, dev_f1 = sweep_threshold(dev_logits, dev_golds)
        dev_map = mean_average_precision(dev_logits, dev_golds)
        print(f"epoch {epoch:02d} | dev macro_f1 {dev_f1:.4f} mAP {dev_map:.4f} "
              f"thr {thr:.3f} | {time.time()-t0:.0f}s", flush=True)
        if dev_f1 > best["dev_macro_f1"]:
            best = {"epoch": epoch, "dev_macro_f1": dev_f1, "dev_map": dev_map,
                    "threshold": thr}
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})

    # ── test with best checkpoint & dev-chosen threshold ─────────────────
    model.load_state_dict(best_state)
    model.to(device)
    test_logits, test_golds = collect_logits(model, ds["test"], device)
    preds = (test_logits >= best["threshold"]).astype(np.float32)

    tc = np.array([train_counts.get(t, 0) for t in vocab])
    result = {
        "config": vars(args),
        "selection": best,
        "test": compute_metrics(preds, test_golds),
        "test_map": mean_average_precision(test_logits, test_golds),
        "test_tail": tail_stratified_f1(preds, test_golds, tc),
        "test_hier": hierarchical_f1(preds, test_golds, vocab, tax),
        "runtime_s": round(time.time() - t0),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
