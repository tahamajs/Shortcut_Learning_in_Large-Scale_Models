"""Mitigation methods for shortcut learning.

Implements:
  1. Counterfactual data augmentation
  2. Group DRO loss (Sagawa et al. 2020)
  3. Invariant Risk Minimisation (IRM) penalty (Arjovsky et al. 2020)
  4. JTT — Just Train Twice (Liu et al. 2021)
  5. Focal loss for hard-example emphasis
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Counterfactual data augmentation
# ---------------------------------------------------------------------------
def counterfactual_augment_text(
    rows: Iterable[dict],
    mapping: Dict[str, str],
) -> List[dict]:
    """Produce augmented examples by replacing tokens according to *mapping*."""
    augmented: List[dict] = []
    for r in rows:
        text = r.get("text", "")
        for src, dst in mapping.items():
            text = text.replace(src, dst)
        new_row = dict(r)
        new_row["text"] = text
        new_row["group"] = "counterfactual"
        augmented.append(new_row)
    return augmented


def counterfactual_augment_nli(
    rows: Iterable[dict],
    mapping: Dict[str, str],
) -> List[dict]:
    """Augment NLI pair data (premise/hypothesis) with token replacement."""
    augmented: List[dict] = []
    for r in rows:
        new_row = dict(r)
        for field in ("premise", "hypothesis", "text"):
            if field in new_row:
                text = new_row[field]
                for src, dst in mapping.items():
                    text = text.replace(src, dst)
                new_row[field] = text
        new_row["group"] = "counterfactual"
        augmented.append(new_row)
    return augmented


# ---------------------------------------------------------------------------
# 2. Group DRO loss (Sagawa et al. 2020)
# ---------------------------------------------------------------------------
class GroupDROLoss(torch.nn.Module):
    """Online Group DRO with exponential weight update.

    Maintains a distribution *q* over groups and upweights high-loss groups.
    """

    def __init__(self, n_groups: int, step_size: float = 0.01) -> None:
        super().__init__()
        self.step_size = step_size
        # Uniform initialisation on log scale
        self.register_buffer("log_q", torch.zeros(n_groups))
        self.group_names: List[str] = []

    def set_group_names(self, names: List[str]) -> None:
        self.group_names = list(names)

    def _group_index(self, g: str) -> int:
        if g not in self.group_names:
            self.group_names.append(g)
        return self.group_names.index(g)

    def forward(
        self,
        per_sample_loss: torch.Tensor,
        groups: Sequence[str],
    ) -> torch.Tensor:
        device = per_sample_loss.device
        group_indices = torch.tensor(
            [self._group_index(g) for g in groups], device=device
        )
        n_groups = max(len(self.group_names), int(group_indices.max().item()) + 1)
        # Ensure log_q is large enough
        if self.log_q.numel() < n_groups:
            new_log_q = torch.zeros(n_groups, device=device)
            new_log_q[: self.log_q.numel()] = self.log_q
            self.log_q = new_log_q

        group_losses = torch.zeros(n_groups, device=device)
        group_counts = torch.zeros(n_groups, device=device)
        for i in range(n_groups):
            mask = group_indices == i
            if mask.any():
                group_losses[i] = per_sample_loss[mask].mean()
                group_counts[i] = mask.sum()

        # Update weights
        valid = group_counts > 0
        self.log_q[:n_groups][valid] += self.step_size * group_losses[valid].detach()
        q = F.softmax(self.log_q[:n_groups], dim=0)
        return (q * group_losses).sum()


# ---------------------------------------------------------------------------
# 3. IRM penalty (Arjovsky et al. 2020)
# ---------------------------------------------------------------------------
def irm_penalty(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute IRMv1 penalty: || grad_{w=1} CE(w * logits, y) ||^2.

    The scalar *w* = 1.0 is a dummy variable. We differentiate the loss
    w.r.t. *w* and return the squared norm of that gradient.
    """
    w = torch.tensor(1.0, device=logits.device, requires_grad=True)
    loss = F.cross_entropy(logits * w, labels)
    grad = torch.autograd.grad(loss, w, create_graph=True)[0]
    return grad ** 2


def irm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    groups: Sequence[str],
    penalty_weight: float = 1.0,
) -> torch.Tensor:
    """ERM loss + penalty_weight * IRM penalty (averaged over environments)."""
    erm = F.cross_entropy(logits, labels)
    env_set = sorted(set(groups))
    if len(env_set) <= 1:
        return erm
    penalties = []
    for env in env_set:
        mask = torch.tensor([g == env for g in groups], device=logits.device)
        if mask.sum() < 2:
            continue
        penalties.append(irm_penalty(logits[mask], labels[mask]))
    if not penalties:
        return erm
    return erm + penalty_weight * torch.stack(penalties).mean()


# ---------------------------------------------------------------------------
# 4. JTT — Just Train Twice (Liu et al. 2021)
# ---------------------------------------------------------------------------
def identify_error_set(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
) -> List[int]:
    """Run model over dataloader and return indices of misclassified examples."""
    model.eval()
    error_indices: List[int] = []
    offset = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            logits = model(**inputs).logits
            preds = logits.argmax(dim=-1).cpu()
            labels = batch["labels"]
            for i, (p, y) in enumerate(zip(preds, labels)):
                if p.item() != y.item():
                    error_indices.append(offset + i)
            offset += len(labels)
    model.train()
    return error_indices


def upsample_error_set(
    dataset,
    error_indices: List[int],
    upsample_factor: int = 4,
) -> List[int]:
    """Return new index list with error examples upsampled by *upsample_factor*."""
    all_indices = list(range(len(dataset)))
    for idx in error_indices:
        all_indices.extend([idx] * (upsample_factor - 1))
    return all_indices


# ---------------------------------------------------------------------------
# 5. Focal loss
# ---------------------------------------------------------------------------
def focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Focal loss (Lin et al. 2017) for hard-example mining."""
    ce = F.cross_entropy(logits, labels, reduction="none")
    pt = torch.exp(-ce)
    loss = ((1 - pt) ** gamma) * ce
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss
