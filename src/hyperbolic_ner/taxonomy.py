"""Type taxonomy: a parent map over type names + ancestor utilities.

Stored as JSON: {"type_name": "parent_name_or_null", ...}. A type with a null
parent is a root. Used for (a) ancestor-propagated supervision and (b)
hierarchical evaluation, and to seed the hyperbolic label geometry (general
types near the origin, specific types near the boundary).
"""
from __future__ import annotations

import json
from functools import lru_cache


class Taxonomy:
    def __init__(self, parent: dict[str, str | None]):
        self.parent = parent

    @classmethod
    def load(cls, path: str) -> "Taxonomy":
        with open(path) as f:
            return cls(json.load(f))

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.parent, f, ensure_ascii=False, indent=0)

    @lru_cache(maxsize=None)
    def ancestors(self, t: str) -> tuple[str, ...]:
        """All strict ancestors of t (root-ward), excluding t itself."""
        out, cur, seen = [], self.parent.get(t), set()
        while cur is not None and cur not in seen:
            out.append(cur)
            seen.add(cur)
            cur = self.parent.get(cur)
        return tuple(out)

    def closure(self, t: str) -> tuple[str, ...]:
        """t plus all its ancestors."""
        return (t, *self.ancestors(t))

    def depth(self, t: str) -> int:
        return len(self.ancestors(t))


def propagate_ancestors(labels, taxonomy: Taxonomy, type_list: list[str],
                        type2idx: dict[str, int]):
    """Expand a [B, K] binary label tensor with each positive's ancestors.

    Turns "gold = actor" into positives {actor, artist, person, ...} so the
    model is not penalized for predicting a correct supertype.
    """
    import torch
    out = labels.clone()
    pos = labels.nonzero(as_tuple=False)
    for b, k in pos.tolist():
        for anc in taxonomy.ancestors(type_list[k]):
            j = type2idx.get(anc)
            if j is not None:
                out[b, j] = 1.0
    return out
