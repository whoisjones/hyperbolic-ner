"""P3b prep: deterministic doc-level train/test split of FiNERweb-eng.

FiNERweb ships as one file with no split. We shuffle records with a fixed seed
and carve off a test fraction, writing both to /vol/tmp (heavy data lives there,
not on the space-limited working volume). The split is fixed so every geometry/
seed in the P3b grid trains and evaluates on the same docs.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SRC = "/vol/tmp/goldejon/multilingual_ner/data/training_jsonl/finerweb/eng.jsonl"
OUT = "/vol/tmp/goldejon/sparse_ner/data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out-dir", default=OUT)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--split-seed", type=int, default=7)
    args = ap.parse_args()

    recs = [line for line in open(args.src) if line.strip()]
    random.Random(args.split_seed).shuffle(recs)
    n_test = int(len(recs) * args.test_frac)
    test, train = recs[:n_test], recs[n_test:]

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("test", test)]:
        p = f"{args.out_dir}/finerweb_eng_{name}.jsonl"
        with open(p, "w") as f:
            f.writelines(part)
        print(f"wrote {p}: {len(part)} docs")


if __name__ == "__main__":
    main()
