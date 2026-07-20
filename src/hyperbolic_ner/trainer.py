"""Minimal training / evaluation loop for the span-typing bi-encoder."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import collate_fn
from .losses import bce_loss, distance_ranking_loss
from .metrics import compute_metrics
from .taxonomy import propagate_ancestors


def _to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, dataset, device, batch_size=64, threshold=0.5):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    all_p, all_g = [], []
    for batch in loader:
        batch = _to_device(batch, device)
        logits = model(batch["input_ids"], batch["attention_mask"],
                       batch["mention_starts"], batch["mention_ends"])
        all_p.append(model.predict(logits, threshold).cpu().numpy())
        all_g.append(batch["labels"].cpu().numpy())
    preds, golds = np.concatenate(all_p), np.concatenate(all_g)
    return compute_metrics(preds, golds), preds, golds


def train(
    model, train_ds, val_ds, device,
    epochs=3, batch_size=32, lr=2e-5, loss_type="bce",
    pos_weight=1.0, taxonomy=None, type_list=None, type2idx=None,
    log_every=50,
):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_fn)
    use_ancestors = taxonomy is not None and type_list is not None

    for epoch in range(epochs):
        model.train()
        for step, batch in enumerate(loader):
            batch = _to_device(batch, device)
            targets = batch["labels"]
            if use_ancestors:
                targets = propagate_ancestors(targets.cpu(), taxonomy,
                                              type_list, type2idx).to(device)
            logits = model(batch["input_ids"], batch["attention_mask"],
                           batch["mention_starts"], batch["mention_ends"])
            if loss_type == "bce":
                loss = bce_loss(logits, targets, pos_weight)
            elif loss_type == "ranking":
                loss = distance_ranking_loss(logits, targets)
            else:
                raise ValueError(loss_type)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % log_every == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}", flush=True)
        metrics, _, _ = evaluate(model, val_ds, device, batch_size)
        print(f"[epoch {epoch}] val {metrics}", flush=True)
    return model
