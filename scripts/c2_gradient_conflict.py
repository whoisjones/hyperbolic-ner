"""C2 — measure conflicting gradients on label embeddings during training.

Claim under test: under Zipfian labels, the BCE training signal sends head
labels systematically contradictory gradients — pulled toward mentions they
are true of (positive terms), pushed away from mentions of their descendants
(negative terms where the label is actually true but unannotated).

Naive pos-vs-neg gradient cosine saturates at ~-1 for every label (pull toward
members vs push from the rest are trivially antiparallel in an anisotropic
space) and Adam momentum saturates update autocorrelation, so neither
discriminates. Instead we decompose the BCE negative term via the taxonomy
closure into:

  FALSE negatives — target 0 but the label is an ancestor of the mention's
                    gold (pushing away is WRONG; this is the conflict)
  TRUE  negatives — target 0 and genuinely unrelated (pushing away is fine)

and per label/step log the gradient-magnitude share of each part w.r.t. the
label embedding, plus cos(g_pos, g_fneg). Predictions: (i) false-negative
gradient share is frequency-concentrated (Zipf link); (ii) hyperbolic shows a
smaller false-negative share than Euclidean at matched frequency (distant
descendants saturate the sigmoid, so wrong pushes carry less gradient).

All grads are taken w.r.t. the label-embedding matrix only (cheap: autograd
prunes the encoder branch). Training itself is the standard P1 cell (UFET
crowd, flat BCE, pos_weight matched to train_probe).

Usage:
  python scripts/c2_gradient_conflict.py --geometry euclidean \
      --out results/c2/euc-d64.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import sys
sys.path.insert(0, "src")
from hyperbolic_ner.data import SpanTypingDataset, build_type_vocab, collate_fn  # noqa: E402
from hyperbolic_ner.model import BiEncoderTyper  # noqa: E402
from hyperbolic_ner.taxonomy import Taxonomy  # noqa: E402

DATA = "/vol/tmp/goldejon/hyperbolic_ner/data"


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", choices=["euclidean", "hyperbolic"], required=True)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--encoder", default="bert-base-uncased")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr-encoder", type=float, default=2e-5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--pos-weight", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--taxonomy", default="results/taxonomy/wordnet_parent.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    splits = [f"{DATA}/ufet_crowd_{s}.jsonl" for s in ("train", "validation", "test")]
    vocab, _ = build_type_vocab(splits)
    _, train_counts = build_type_vocab([splits[0]])
    type2idx = {t: i for i, t in enumerate(vocab)}
    K = len(vocab)

    tok = AutoTokenizer.from_pretrained(args.encoder)
    train_ds = SpanTypingDataset(splits[0], tok, type2idx)
    print(f"vocab={K} | train={len(train_ds)}", flush=True)

    tax = Taxonomy.load(args.taxonomy)
    Cmat = closure_matrix(vocab, tax).to(device)

    model = BiEncoderTyper(args.encoder, vocab, dim=args.dim,
                           geometry=args.geometry).to(device)
    W = model.label_emb.weight

    enc_params = list(model.encoder.parameters())
    enc_ids = {id(p) for p in enc_params}
    head_params = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([
        {"params": enc_params, "lr": args.lr_encoder},
        {"params": head_params, "lr": args.lr_head},
    ])

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate_fn)
    pw = torch.tensor(args.pos_weight, device=device)

    # per-label accumulators over steps where the part had nonzero weight
    sum_gpos = np.zeros(K); sum_gfneg = np.zeros(K); sum_gtneg = np.zeros(K)
    cnt_fneg = np.zeros(K)             # steps where label had a false-neg term
    sum_cos_pf = np.zeros(K)           # cos(g_pos, g_fneg), needs both present
    cnt_cos_pf = np.zeros(K)
    eps = 1e-12

    for epoch in range(args.epochs):
        model.train()
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            targets = batch["labels"]
            # closure of gold: gold + ancestors of gold, per mention
            closure_t = (targets @ Cmat).clamp_max_(1.0)
            w_fneg = closure_t * (1.0 - targets)   # ancestor-of-gold, target 0
            w_tneg = 1.0 - closure_t               # genuinely unrelated

            logits = model(batch["input_ids"], batch["attention_mask"],
                           batch["mention_starts"], batch["mention_ends"])
            numel = float(logits.numel())
            loss_pos = F.binary_cross_entropy_with_logits(
                logits, targets, weight=targets, pos_weight=pw,
                reduction="sum") / numel
            loss_fneg = F.binary_cross_entropy_with_logits(
                logits, targets, weight=w_fneg, pos_weight=pw,
                reduction="sum") / numel
            loss_tneg = F.binary_cross_entropy_with_logits(
                logits, targets, weight=w_tneg, pos_weight=pw,
                reduction="sum") / numel

            g_pos = torch.autograd.grad(loss_pos, W, retain_graph=True)[0]
            g_fneg = torch.autograd.grad(loss_fneg, W, retain_graph=True)[0]
            g_tneg = torch.autograd.grad(loss_tneg, W, retain_graph=True)[0]

            # magnitude shares per label (row norms)
            n_pos = g_pos.norm(dim=1); n_fn = g_fneg.norm(dim=1)
            n_tn = g_tneg.norm(dim=1)
            sum_gpos += n_pos.cpu().numpy()
            sum_gfneg += n_fn.cpu().numpy()
            sum_gtneg += n_tn.cpu().numpy()
            has_fn = (w_fneg.sum(0) > 0)
            cnt_fneg += has_fn.cpu().numpy()

            both = (n_pos > eps) & (n_fn > eps)
            if both.any():
                cs = F.cosine_similarity(g_pos[both], g_fneg[both], dim=1)
                idx = both.nonzero(as_tuple=True)[0].cpu().numpy()
                sum_cos_pf[idx] += cs.detach().cpu().numpy()
                cnt_cos_pf[idx] += 1

            loss = loss_pos + loss_fneg + loss_tneg
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        print(f"epoch {epoch:02d} loss {loss.item():.4f}", flush=True)

    # ── aggregate ────────────────────────────────────────────────────────
    per_label = []
    for k, t in enumerate(vocab):
        tot = sum_gpos[k] + sum_gfneg[k] + sum_gtneg[k]
        if tot <= eps:
            continue
        per_label.append({
            "label": t,
            "train_freq": train_counts.get(t, 0),
            "n_steps_with_fneg": int(cnt_fneg[k]),
            "fneg_grad_share": float(sum_gfneg[k] / tot),
            "pos_grad_share": float(sum_gpos[k] / tot),
            "fneg_to_pos_ratio": float(sum_gfneg[k] / max(sum_gpos[k], eps)),
            "mean_cos_pos_fneg": float(sum_cos_pf[k] / cnt_cos_pf[k])
            if cnt_cos_pf[k] > 0 else None,
        })

    freqs = np.array([r["train_freq"] for r in per_label])
    shares = np.array([r["fneg_grad_share"] for r in per_label])
    rho, pval = spearmanr(freqs, shares)

    buckets = {}
    for name, (lo, hi) in [("head", (100, 10**9)), ("mid", (6, 99)), ("tail", (0, 5))]:
        m = (freqs >= lo) & (freqs <= hi)
        if m.any():
            sel = [r for r, mm in zip(per_label, m) if mm]
            cosv = [r["mean_cos_pos_fneg"] for r in sel
                    if r["mean_cos_pos_fneg"] is not None]
            buckets[name] = {
                "n_labels": int(m.sum()),
                "mean_fneg_grad_share": float(shares[m].mean()),
                "mean_fneg_to_pos_ratio": float(np.mean(
                    [min(r["fneg_to_pos_ratio"], 1e6) for r in sel])),
                "mean_cos_pos_fneg": float(np.mean(cosv)) if cosv else None,
            }

    result = {
        "config": vars(args),
        "n_labels_observed": len(per_label),
        "spearman_freq_vs_fneg_share": {"rho": float(rho), "p": float(pval)},
        "buckets": buckets,
        "per_label": sorted(per_label, key=lambda r: -r["train_freq"])[:200],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "per_label"}, indent=2))


if __name__ == "__main__":
    main()
