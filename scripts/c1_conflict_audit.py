"""C1 — statistical audit of conflicting training signal (no training).

Claim under test: with Zipfian labels, losses that treat non-gold labels as
negatives (BCE over the full label space; InfoNCE with in-batch negatives)
systematically push mentions away from labels that are actually TRUE of them
(ancestors of the gold), and this conflict concentrates on head labels.

Two views, per corpus:

  BCE view (matches our training): for every span, every taxonomy-ancestor of
  a gold label that is not itself annotated gets target 0 — a false negative
  baked into the supervision. We count these per example and attribute each
  conflict to the ancestor label that receives the contradictory push.

  InfoNCE view (in-batch negatives): sample batches of size B; the negative
  set for example i is the union of gold labels of the other examples. A false
  negative occurs when that set contains (a) a label true of i (identical to
  one of i's golds) or (b) an ancestor of one of i's golds. Rate reported as
  fraction of examples with >=1 false negative, and mean false negatives per
  example, as a function of B.

Attribution by frequency: each conflicting label is bucketed by its train
frequency rank (head >=100, mid 6-99, tail <=5) to show the conflict is
Zipf-concentrated.

Usage:
  python scripts/c1_conflict_audit.py \
      --train <train.jsonl> --taxonomy <parent.json> --out results/c1/ufet.json
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, "src")
from hyperbolic_ner.data import build_type_vocab, normalize_record  # noqa: E402
from hyperbolic_ner.taxonomy import Taxonomy  # noqa: E402


def load_examples(path: str) -> list[list[str]]:
    """Per-span gold label lists (lowercased), in file order."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            for _, _, _, types in normalize_record(json.loads(line)):
                out.append(types)
    return out


def bucket_of(label: str, counts: Counter) -> str:
    c = counts.get(label, 0)
    return "head" if c >= 100 else ("mid" if c > 5 else "tail")


def bce_view(examples, tax: Taxonomy, counts: Counter) -> dict:
    """Hard-zero ancestor negatives baked into full-vocab BCE targets."""
    n = len(examples)
    n_with_conflict = 0
    total_conflicts = 0
    conflict_by_label: Counter = Counter()
    for gold in examples:
        gold_set = set(gold)
        conflicts = set()
        for g in gold_set:
            for anc in tax.ancestors(g):
                if anc not in gold_set:
                    conflicts.add(anc)
        if conflicts:
            n_with_conflict += 1
            total_conflicts += len(conflicts)
            conflict_by_label.update(conflicts)
    by_bucket = Counter()
    for lab, c in conflict_by_label.items():
        by_bucket[bucket_of(lab, counts)] += c
    total = sum(by_bucket.values()) or 1
    return {
        "n_examples": n,
        "pct_examples_with_conflict": round(100 * n_with_conflict / n, 2),
        "mean_conflicts_per_example": round(total_conflicts / n, 3),
        "conflict_share_by_bucket": {b: round(100 * by_bucket[b] / total, 1)
                                     for b in ("head", "mid", "tail")},
        "top_conflicted_labels": conflict_by_label.most_common(15),
    }


def infonce_view(examples, tax: Taxonomy, counts: Counter,
                 batch_sizes=(8, 16, 32, 64, 128, 256),
                 n_batches=500, seed=42) -> dict:
    """False-negative rates for in-batch negatives at varying batch size."""
    rng = random.Random(seed)
    n = len(examples)
    # precompute per-example closure (gold + ancestors of gold)
    gold_sets = [set(g) for g in examples]
    closures = []
    for gs in gold_sets:
        cl = set(gs)
        for g in gs:
            cl.update(tax.ancestors(g))
        closures.append(cl)

    out = {}
    for B in batch_sizes:
        ex_fn = 0          # examples with >=1 false negative
        fn_total = 0       # total false-negative labels
        fn_by_bucket = Counter()
        n_ex = 0
        for _ in range(n_batches):
            idxs = rng.sample(range(n), B)
            batch_golds = [gold_sets[i] for i in idxs]
            for j, i in enumerate(idxs):
                neg = set().union(*(batch_golds[k] for k in range(B) if k != j))
                false_negs = neg & closures[i]
                n_ex += 1
                if false_negs:
                    ex_fn += 1
                    fn_total += len(false_negs)
                    for lab in false_negs:
                        fn_by_bucket[bucket_of(lab, counts)] += 1
        total = sum(fn_by_bucket.values()) or 1
        out[str(B)] = {
            "pct_examples_with_false_negative": round(100 * ex_fn / n_ex, 2),
            "mean_false_negatives_per_example": round(fn_total / n_ex, 3),
            "fn_share_by_bucket": {b: round(100 * fn_by_bucket[b] / total, 1)
                                   for b in ("head", "mid", "tail")},
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--taxonomy", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-batches", type=int, default=500)
    args = ap.parse_args()

    _, counts = build_type_vocab([args.train])
    tax = Taxonomy.load(args.taxonomy)
    examples = load_examples(args.train)

    result = {
        "train_file": args.train,
        "taxonomy": args.taxonomy,
        "n_spans": len(examples),
        "n_types": len(counts),
        "bce_view": bce_view(examples, tax, counts),
        "infonce_view": infonce_view(examples, tax, counts,
                                     n_batches=args.n_batches),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
