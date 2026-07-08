"""E0 - establish the premise: unified type vocabulary + long-tail stats.

Scans one or more span-typing JSONL files, counts type frequencies, writes a
frequency table, and prints long-tail statistics (share of types with <= N
mentions). This produces the motivation figure/numbers for the paper.

The ~1k-node taxonomy build (E0b) is a follow-up that consumes the vocab this
script writes; kept separate so the vocab scan can be inspected first.

Usage:
    python scripts/e0_build_vocab_taxonomy.py \
        --paths /vol/tmp/goldejon/sparse_ner/data/ufet_crowd_train.jsonl \
                /vol/tmp/goldejon/multilingual_ner/data/training_jsonl/finerweb/eng.jsonl \
        --out results/e0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sparse_ner.data import build_type_vocab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", nargs="+", required=True)
    ap.add_argument("--out", default="results/e0")
    ap.add_argument("--tail-thresh", type=int, default=5)
    args = ap.parse_args()

    vocab, counts = build_type_vocab(args.paths)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    freq = counts.most_common()
    with open(out / "type_counts.tsv", "w") as f:
        for t, c in freq:
            f.write(f"{t}\t{c}\n")

    total_types = len(counts)
    total_mentions = sum(counts.values())
    n_tail = sum(1 for _, c in freq if c <= args.tail_thresh)
    stats = {
        "n_types": total_types,
        "n_mentions": total_mentions,
        "pct_types_tail": round(100 * n_tail / max(total_types, 1), 2),
        "tail_thresh": args.tail_thresh,
        "top10": freq[:10],
    }
    with open(out / "stats.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
