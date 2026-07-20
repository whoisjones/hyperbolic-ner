"""Span-level bi-encoder for fine-grained entity typing.

    mention span  --encoder--> pool [start,end] --> project to D
    type name k   --(label embedding, D)-------------------------┐
                                                                 v
                          GeometryHead (euclidean | hyperbolic) -> logit[b, k]

Label embeddings are ordinary parameters, optionally initialized from the
backbone's encoding of each type-name string so training starts from a
semantically meaningful configuration.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from .geometry import GeometryHead


def masked_span_mean(hidden: torch.Tensor, starts: torch.Tensor,
                     ends: torch.Tensor) -> torch.Tensor:
    """Mean-pool hidden states over each [start, end] token span (inclusive).

    hidden [B, L, H], starts/ends [B] -> [B, H]. Fully vectorized.
    """
    B, L, H = hidden.shape
    ar = torch.arange(L, device=hidden.device).unsqueeze(0)          # [1, L]
    mask = (ar >= starts.unsqueeze(1)) & (ar <= ends.unsqueeze(1))   # [B, L]
    mask = mask.unsqueeze(-1).to(hidden.dtype)                       # [B, L, 1]
    summed = (hidden * mask).sum(dim=1)                              # [B, H]
    count = mask.sum(dim=1).clamp_min(1.0)                           # [B, 1]
    return summed / count


class BiEncoderTyper(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        type_names: list[str],
        dim: int = 128,
        geometry: str = "euclidean",
        curvature: float = 1.0,
        dropout: float = 0.1,
        init_labels_from_names: bool = True,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size
        self.type_names = type_names
        self.num_types = len(type_names)
        self.dim = dim

        self.dropout = nn.Dropout(dropout)
        self.mention_proj = nn.Linear(hidden, dim)
        self.label_emb = nn.Embedding(self.num_types, dim)
        self.head = GeometryHead(geometry=geometry, c=curvature)

        if init_labels_from_names:
            self._init_labels(encoder_name, type_names)

    @torch.no_grad()
    def _init_labels(self, encoder_name: str, type_names: list[str]) -> None:
        """Seed label embeddings from mean-pooled backbone encodings of names."""
        tok = AutoTokenizer.from_pretrained(encoder_name)
        vecs = []
        self.encoder.eval()
        dev = next(self.encoder.parameters()).device
        for i in range(0, len(type_names), 128):
            chunk = type_names[i:i + 128]
            enc = tok(chunk, padding=True, truncation=True, max_length=16,
                      return_tensors="pt").to(dev)
            out = self.encoder(**enc).last_hidden_state           # [b, l, H]
            m = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out * m).sum(1) / m.sum(1).clamp_min(1.0)   # [b, H]
            vecs.append(self.mention_proj(pooled))
        self.label_emb.weight.copy_(torch.cat(vecs, 0))

    def encode_mentions(self, input_ids, attention_mask, starts, ends):
        hidden = self.encoder(input_ids, attention_mask).last_hidden_state
        rep = masked_span_mean(hidden, starts, ends)
        return self.mention_proj(self.dropout(rep))               # [B, D]

    def forward(self, input_ids, attention_mask, mention_starts, mention_ends):
        m = self.encode_mentions(input_ids, attention_mask, mention_starts, mention_ends)
        return self.head(m, self.label_emb.weight)                # logits [B, K]

    @torch.no_grad()
    def predict(self, logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        return (torch.sigmoid(logits) >= threshold).float()
