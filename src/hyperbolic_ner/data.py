"""Unified span-typing data loader.

Handles the two JSONL schemas in this project:

  UFET (multi-label, char offsets):
    {"text": ..., "spans": [{"start": s, "end": e, "type": ["actor","person"]}]}

  Distant NER dumps  (finerweb / pilener / nuner / euro_glinerx; single tag):
    {"text": ..., "spans_char": [{"start": s, "end": e, "tag": "publisher"}], ...}

Both are normalized to per-span examples: (text, char_start, char_end, [types]).
Type strings are lowercased and mapped through an optional `type2idx`; a span's
labels not present in the vocab are dropped (kept only if >=1 label survives).
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast


def normalize_record(rec: dict) -> list[tuple[str, int, int, list[str]]]:
    """Yield (text, char_start, char_end, types) tuples from one JSON record."""
    text = rec["text"]
    out = []
    if "spans" in rec:  # UFET multi-label
        for sp in rec["spans"]:
            types = [t.lower() for t in sp["type"]]
            out.append((text, sp["start"], sp["end"], types))
    elif "spans_char" in rec:  # distant single-tag
        for sp in rec["spans_char"]:
            out.append((text, sp["start"], sp["end"], [sp["tag"].lower()]))
    return out


def build_type_vocab(paths: list[str], min_count: int = 1) -> tuple[list[str], Counter]:
    """Count type frequencies across files; return (sorted vocab, counter)."""
    counts: Counter = Counter()
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                for _, _, _, types in normalize_record(json.loads(line)):
                    counts.update(types)
    vocab = sorted(t for t, c in counts.items() if c >= min_count)
    return vocab, counts


def _iter_records(paths: list[str], max_records: int | None, seed: int):
    if max_records is None:
        for path in paths:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
        return
    rng = random.Random(seed)
    reservoir: list[dict] = []
    seen = 0
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                seen += 1
                item = json.loads(line)
                if len(reservoir) < max_records:
                    reservoir.append(item)
                else:
                    j = rng.randint(0, seen - 1)
                    if j < max_records:
                        reservoir[j] = item
    rng.shuffle(reservoir)
    yield from reservoir


class SpanTypingDataset(Dataset):
    def __init__(
        self,
        paths: str | list[str],
        tokenizer: PreTrainedTokenizerFast,
        type2idx: dict[str, int],
        max_len: int = 128,
        max_records: int | None = None,
        seed: int = 42,
        noise_rate: float = 0.0,
        noise_mode: str = "sibling",
        taxonomy=None,
        noise_seed: int = 12345,
    ):
        if isinstance(paths, str):
            paths = [paths]
        self.tokenizer = tokenizer
        self.type2idx = type2idx
        self.num_types = len(type2idx)
        self.examples: list[dict] = []

        for rec in _iter_records(paths, max_records, seed):
            for text, cs, ce, types in normalize_record(rec):
                ex = self._process(text, cs, ce, types, max_len)
                if ex is not None:
                    self.examples.append(ex)

        if noise_rate > 0.0:
            self.noise_stats = self._corrupt_labels(
                noise_rate, noise_mode, taxonomy, noise_seed)
        else:
            self.noise_stats = {"corrupted": 0, "total_pos": 0}

    def _process(self, text, char_start, char_end, types, max_len):
        enc = self.tokenizer(text, max_length=max_len, truncation=True,
                             return_offsets_mapping=True)
        tok_start = tok_end = None
        for i, (a, b) in enumerate(enc["offset_mapping"]):
            if a == b == 0 and i > 0:
                continue
            if b > char_start and a < char_end:
                if tok_start is None:
                    tok_start = i
                tok_end = i
        if tok_start is None:
            return None

        idxs = [self.type2idx[t] for t in types if t in self.type2idx]
        if not idxs:
            return None
        label_vec = torch.zeros(self.num_types)
        label_vec[idxs] = 1.0
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "mention_start": tok_start,
            "mention_end": tok_end,
            "labels": label_vec,
        }

    def _corrupt_labels(self, noise_rate, noise_mode, taxonomy, noise_seed):
        """Inject seeded label noise into TRAIN labels only (per positive label).

        For each gold positive, with probability `noise_rate`, replace it with a
        wrong type: a taxonomy sibling (realistic) or a uniform-random type
        (adversarial control). Siblings share the same taxonomy parent; if a type
        has no in-vocab sibling, we fall back to uniform for that label so the
        target corruption rate is honored. Multi-label spans are corrupted
        per-label independently.
        """
        if noise_mode not in ("sibling", "uniform"):
            raise ValueError(f"unknown noise_mode {noise_mode!r}")
        idx2type = {i: t for t, i in self.type2idx.items()}

        # sibling map: parent -> [child idxs in vocab]
        children: dict[str, list[int]] = {}
        if noise_mode == "sibling":
            if taxonomy is None:
                raise ValueError("noise_mode='sibling' requires a taxonomy")
            for t, i in self.type2idx.items():
                p = taxonomy.parent.get(t)
                children.setdefault(p, []).append(i)

        rng = random.Random(noise_seed)
        corrupted = total_pos = 0
        for ex in self.examples:
            pos = (ex["labels"] > 0).nonzero(as_tuple=True)[0].tolist()
            total_pos += len(pos)
            for k in pos:
                if rng.random() >= noise_rate:
                    continue
                repl = None
                if noise_mode == "sibling":
                    parent = taxonomy.parent.get(idx2type[k])
                    sibs = [j for j in children.get(parent, []) if j != k]
                    if sibs:
                        repl = rng.choice(sibs)
                if repl is None:  # uniform mode, or sibling fallback
                    repl = rng.randrange(self.num_types)
                    while repl == k:
                        repl = rng.randrange(self.num_types)
                ex["labels"][k] = 0.0
                ex["labels"][repl] = 1.0
                corrupted += 1
        return {"corrupted": corrupted, "total_pos": total_pos,
                "effective_rate": corrupted / max(total_pos, 1)}

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch: list[dict]) -> dict:
    max_len = max(len(ex["input_ids"]) for ex in batch)
    input_ids, attention_mask = [], []
    for ex in batch:
        pad = max_len - len(ex["input_ids"])
        input_ids.append(ex["input_ids"] + [0] * pad)
        attention_mask.append(ex["attention_mask"] + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "mention_starts": torch.tensor([ex["mention_start"] for ex in batch], dtype=torch.long),
        "mention_ends": torch.tensor([ex["mention_end"] for ex in batch], dtype=torch.long),
        "labels": torch.stack([ex["labels"] for ex in batch]),
    }
