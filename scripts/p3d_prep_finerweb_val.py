"""P3d prep: carve a validation set out of the FiNERweb-eng train pool.

The supervised probe (train_probe.py) needs a dev split for the threshold sweep
and model selection, but the P3b split only made train/test. Here we shuffle the
existing train pool (same split-seed) and peel off a fixed validation slice,
leaving a supervised-train file. Test is reused as-is. All under /vol/tmp.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

POOL = "/vol/tmp/goldejon/hyperbolic_ner/data/finerweb_eng_train.jsonl"
OUT = "/vol/tmp/goldejon/hyperbolic_ner/data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=POOL)
    ap.add_argument("--out-dir", default=OUT)
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--split-seed", type=int, default=7)
    args = ap.parse_args()

    recs = [line for line in open(args.pool) if line.strip()]
    random.Random(args.split_seed).shuffle(recs)
    val, train = recs[:args.n_val], recs[args.n_val:]

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    for name, part in [("suptrain", train), ("val", val)]:
        p = f"{args.out_dir}/finerweb_eng_{name}.jsonl"
        with open(p, "w") as f:
            f.writelines(part)
        print(f"wrote {p}: {len(part)} docs")


if __name__ == "__main__":
    main()
