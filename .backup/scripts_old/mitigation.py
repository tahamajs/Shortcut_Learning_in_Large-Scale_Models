"""Mitigation hooks: counterfactual augmentation and lightweight GroupDRO objective."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List

import torch


def counterfactual_augment_text(rows: Iterable[dict], mapping: Dict[str, str]) -> List[dict]:
    augmented = []
    for r in rows:
        text = r["text"]
        for src, dst in mapping.items():
            text = text.replace(src, dst)
        new_row = dict(r)
        new_row["text"] = text
        new_row["group"] = "counterfactual"
        augmented.append(new_row)
    return augmented


def group_dro_loss(losses: torch.Tensor, groups: List[str], step_size: float = 0.1) -> torch.Tensor:
    group_to_losses = defaultdict(list)
    for loss, grp in zip(losses, groups):
        group_to_losses[grp].append(loss)
    group_means = torch.stack([torch.stack(v).mean() for v in group_to_losses.values()])
    q = torch.softmax(step_size * group_means.detach(), dim=0)
    return (q * group_means).sum()
