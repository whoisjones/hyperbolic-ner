"""P3 zero-shot tail probe: can name-seeded label embeddings + geometry type
*held-out* types never seen in training?

Protocol (fair across geometries — the split is fixed, geometry is the only knob):
  - Split the vocab into SEEN / HELD-OUT. Held-out = a stratified 50% sample
    (fixed `--split-seed`) of tail types (train freq in [lo, hi]) that have
    >= `--min-test` test occurrences, so zero-shot eval has support.
  - Train a bi-encoder whose label matrix contains ONLY seen types, so held-out
    types receive no supervision at all (not even as BCE negatives). Training
    examples keep their seen labels; examples with only held-out labels drop out.
  - Zero-shot eval: re-encode ALL held-out type *names* through the TRAINED
    encoder+mention_proj, score test mentions, and rank over the held-out label
    space. Report held-out mAP (threshold-free) + P@1 + R@5. Also a generalized
    setting that ranks over seen+held-out jointly.

Usage:
  python train_zeroshot.py --geometry hyperbolic --dim 64 \
      --out results/p3/hyp-d64-s1.json
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

from hyperbolic_ner.data import SpanTypingDataset, build_type_vocab, collate_fn
from hyperbolic_ner.losses import bce_loss
from hyperbolic_ner.model import BiEncoderTyper

DATA = "/vol/tmp/goldejon/hyperbolic_ner/data"


def make_split(train_counts, test_counts, lo, hi, min_test, split_seed):
    """Deterministic seen/held-out split. Held-out = 50% of tail types with test
    support; identical for every geometry/seed so the comparison is controlled."""
    cand = sorted(t for t, c in train_counts.items()
                  if lo <= c <= hi and test_counts.get(t, 0) >= min_test)
    rng = np.random.RandomState(split_seed)
    mask = rng.rand(len(cand)) < 0.5
    held = {t for t, m in zip(cand, mask) if m}
    return held


@torch.no_grad()
def encode_type_names(model, names, encoder_name, device, batch_size=128):
    """Embed arbitrary type-name strings with the TRAINED encoder+projection.

    Mirrors BiEncoderTyper._init_labels but uses current (trained) weights and
    returns the vectors instead of copying them into the embedding table.
    """
    tok = AutoTokenizer.from_pretrained(encoder_name)
    model.eval()
    vecs = []
    for i in range(0, len(names), batch_size):
        chunk = names[i:i + batch_size]
        enc = tok(chunk, padding=True, truncation=True, max_length=16,
                  return_tensors="pt").to(device)
        out = model.encoder(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (out * m).sum(1) / m.sum(1).clamp_min(1.0)
        vecs.append(model.mention_proj(pooled))
    return torch.cat(vecs, 0)


@torch.no_grad()
def collect_mention_reps(model, ds, device, batch_size=128):
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn)
    reps, labels = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        r = model.encode_mentions(batch["input_ids"], batch["attention_mask"],
                                  batch["mention_starts"], batch["mention_ends"])
        reps.append(r)
        labels.append(batch["labels"].cpu().numpy())
    return torch.cat(reps, 0), np.concatenate(labels)


def rank_metrics(scores, golds):
    """scores [N, K], golds [N, K] binary over a label subspace.

    Returns mAP (per-example AP averaged), P@1, R@5 over examples with >=1 gold.
    """
    aps, p1, r5 = [], [], []
    order = np.argsort(-scores, axis=1)
    for i in range(scores.shape[0]):
        g = golds[i][order[i]]
        n_pos = int(g.sum())
        if n_pos == 0:
            continue
        hits = np.cumsum(g)
        prec = hits / np.arange(1, len(g) + 1)
        aps.append(float((prec * g).sum() / n_pos))
        p1.append(float(g[0]))
        r5.append(float(g[:5].sum() / min(n_pos, 5)))
    return {"map": float(np.mean(aps)), "p_at_1": float(np.mean(p1)),
            "r_at_5": float(np.mean(r5)), "n_eval": len(aps)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", choices=["euclidean", "hyperbolic"], required=True)
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--encoder", default="bert-base-uncased")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr-encoder", type=float, default=2e-5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--pos-weight", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42)
    # corpus (default: UFET crowd). Override for P3b FiNERweb scale-up.
    ap.add_argument("--train-file", default=f"{DATA}/ufet_crowd_train.jsonl")
    ap.add_argument("--test-file", default=f"{DATA}/ufet_crowd_test.jsonl")
    ap.add_argument("--max-len", type=int, default=128)
    # split definition (fixed across conditions)
    ap.add_argument("--split-seed", type=int, default=7)
    ap.add_argument("--held-lo", type=int, default=1, help="min train freq of held-out cand")
    ap.add_argument("--held-hi", type=int, default=50, help="max train freq of held-out cand")
    ap.add_argument("--min-test", type=int, default=2, help="min test occ for held-out cand")
    ap.add_argument("--train-max-records", type=int, default=None,
                    help="P3c: subsample train to N docs (vocab/split from full file)")
    ap.add_argument("--print-split", action="store_true", help="print split stats and exit")
    ap.add_argument("--out")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_file, test_file = args.train_file, args.test_file
    full_vocab, _ = build_type_vocab([train_file, test_file])
    _, train_counts = build_type_vocab([train_file])
    _, test_counts = build_type_vocab([test_file])

    held = make_split(train_counts, test_counts, args.held_lo, args.held_hi,
                      args.min_test, args.split_seed)
    seen_vocab = [t for t in full_vocab if t not in held]
    held_vocab = [t for t in full_vocab if t in held]
    seen_t2i = {t: i for i, t in enumerate(seen_vocab)}
    held_t2i = {t: i for i, t in enumerate(held_vocab)}

    n_held_train = sum(train_counts.get(t, 0) for t in held_vocab)
    print(f"full_vocab={len(full_vocab)} seen={len(seen_vocab)} held_out={len(held_vocab)}",
          flush=True)
    print(f"held-out train mentions removed={n_held_train} | "
          f"held-out test mentions={sum(test_counts.get(t,0) for t in held_vocab)}", flush=True)
    if args.print_split:
        ex = sorted(held_vocab, key=lambda t: -test_counts.get(t, 0))[:15]
        print("sample held-out types (by test freq):",
              [(t, train_counts.get(t, 0), test_counts.get(t, 0)) for t in ex], flush=True)
        return

    tok = AutoTokenizer.from_pretrained(args.encoder)
    # TRAIN: only seen labels exist -> held-out types get zero supervision.
    # --train-max-records subsamples docs (seeded) while vocab/split stay fixed
    # from the full file, so scale varies but the held-out set does not (P3c).
    train_ds = SpanTypingDataset(train_file, tok, seen_t2i, max_len=args.max_len,
                                 max_records=args.train_max_records, seed=args.seed)
    # TEST: keep full-vocab labels so we can slice seen/held-out columns at eval.
    full_t2i = {t: i for i, t in enumerate(full_vocab)}
    test_ds = SpanTypingDataset(test_file, tok, full_t2i, max_len=args.max_len)
    print(f"train(seen-only)={len(train_ds)} test(full)={len(test_ds)}", flush=True)

    model = BiEncoderTyper(args.encoder, seen_vocab, dim=args.dim,
                           geometry=args.geometry).to(device)

    enc_params = list(model.encoder.parameters())
    enc_ids = {id(p) for p in enc_params}
    head_params = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([
        {"params": enc_params, "lr": args.lr_encoder},
        {"params": head_params, "lr": args.lr_head},
    ])

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate_fn)
    # dev = held-out zero-shot mAP on a held-out slice of TRAIN? No dev leakage of
    # held-out; we select on training loss proxy via last epoch (simple + honest).
    best_state, best_loss = None, float("inf")
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        ep_loss = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"], batch["attention_mask"],
                           batch["mention_starts"], batch["mention_ends"])
            loss = bce_loss(logits, batch["labels"], args.pos_weight)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
        ep_loss /= max(len(loader), 1)
        if ep_loss < best_loss:
            best_loss = ep_loss
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
        print(f"epoch {epoch:02d} | train_loss {ep_loss:.4f} | {time.time()-t0:.0f}s",
              flush=True)

    model.load_state_dict(best_state)
    model.to(device)

    # ── zero-shot eval ───────────────────────────────────────────────────
    reps, test_labels_full = collect_mention_reps(model, test_ds, device)  # [N,D],[N,K_full]
    # gold slices
    seen_cols = np.array([full_t2i[t] for t in seen_vocab])
    held_cols = np.array([full_t2i[t] for t in held_vocab])
    gold_held = test_labels_full[:, held_cols]
    gold_all = test_labels_full  # ordered as full_vocab

    # label embeddings: held-out purely from names (trained encoder); seen trained.
    held_emb = encode_type_names(model, held_vocab, args.encoder, device)  # [Kh, D]
    seen_emb = model.label_emb.weight.detach()                             # [Ks, D]

    def score(mention_reps, label_emb):
        with torch.no_grad():
            return model.head(mention_reps, label_emb).float().cpu().numpy()

    # (a) zero-shot: rank held-out labels only, on examples with >=1 held-out gold
    s_held = score(reps, held_emb)                                  # [N, Kh]
    zs = rank_metrics(s_held, gold_held)

    # (b) generalized: rank seen+held jointly; score held-out golds vs all labels
    all_emb = torch.zeros(len(full_vocab), args.dim, device=device)
    all_emb[torch.as_tensor(seen_cols, device=device)] = seen_emb
    all_emb[torch.as_tensor(held_cols, device=device)] = held_emb
    s_all = score(reps, all_emb)                                    # [N, K_full]
    # keep only held-out positives as gold; ranking space is the full vocab
    held_mask = np.zeros(len(full_vocab), dtype=np.float32)
    held_mask[held_cols] = 1.0
    gen = rank_metrics(s_all, gold_all * held_mask[None, :])

    result = {
        "config": vars(args),
        "split": {"n_seen": len(seen_vocab), "n_held": len(held_vocab),
                  "held_train_mentions_removed": n_held_train,
                  "n_train_spans": len(train_ds)},
        "zeroshot_heldout": zs,
        "generalized_heldout": gen,
        "runtime_s": round(time.time() - t0),
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
