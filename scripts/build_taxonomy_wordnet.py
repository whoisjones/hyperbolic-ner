"""Build a parent map over a type vocabulary via WordNet hypernym paths.

For each type name, find its WordNet noun synset (full string first, then the
head word for multiword types) and walk the hypernym chain rootward; the parent
is the nearest hypernym whose lemma is itself in the vocabulary. Types with no
in-vocab hypernym become roots (parent = null).

This is deliberately mechanical — no manual merging — so it is reproducible
and defensible. Taxonomy quality is ablated later (P4), not tuned here.

Usage:
    python scripts/build_taxonomy_wordnet.py \
        --paths <train.jsonl> <dev.jsonl> <test.jsonl> \
        --out results/taxonomy/wordnet_parent.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from nltk.corpus import wordnet as wn

from sparse_ner.data import build_type_vocab


def head_word(t: str) -> str:
    return t.split()[-1]


def synset_for(t: str):
    """Best noun synset: full phrase (spaces->underscores) else head word."""
    ss = wn.synsets(t.replace(" ", "_"), pos=wn.NOUN)
    if not ss and " " in t:
        ss = wn.synsets(head_word(t), pos=wn.NOUN)
    return ss[0] if ss else None


def build_parent_map(vocab: list[str], counts) -> dict[str, str | None]:
    vocab_set = set(vocab)
    parent: dict[str, str | None] = {}
    for t in vocab:
        syn = synset_for(t)
        parent[t] = None
        if syn is None:
            continue
        # walk rootward: immediate hypernyms first. A candidate parent must be
        # at least as frequent as the child (generality ~ frequency); this
        # prunes wrong-sense jumps like person -> cause -> event.
        frontier = syn.hypernyms()
        seen = set()
        while frontier:
            nxt = []
            for h in frontier:
                if h in seen:
                    continue
                seen.add(h)
                for lemma in h.lemma_names():
                    cand = lemma.replace("_", " ").lower()
                    if (cand in vocab_set and cand != t
                            and counts.get(cand, 0) >= counts.get(t, 1)):
                        parent[t] = cand
                        break
                if parent[t] is not None:
                    break
                nxt.extend(h.hypernyms())
            if parent[t] is not None:
                break
            frontier = nxt
    return parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    vocab, counts = build_type_vocab(args.paths)
    parent = build_parent_map(vocab, counts)

    # break any accidental 2-cycles (a<->b): keep the edge from the rarer node
    for t, p in list(parent.items()):
        if p is not None and parent.get(p) == t:
            if counts[t] >= counts[p]:
                parent[t] = None
            else:
                parent[p] = None

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(parent, f, ensure_ascii=False, indent=0)

    n = len(vocab)
    linked = sum(1 for p in parent.values() if p is not None)
    roots = n - linked
    # depth stats
    from sparse_ner.taxonomy import Taxonomy
    tax = Taxonomy(parent)
    depths = [tax.depth(t) for t in vocab]
    stats = {
        "n_types": n,
        "linked": linked,
        "pct_linked": round(100 * linked / n, 1),
        "roots": roots,
        "max_depth": max(depths),
        "mean_depth": round(sum(depths) / n, 2),
    }
    print(json.dumps(stats, indent=2))
    with open(out.parent / "taxonomy_stats.json", "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
