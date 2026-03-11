"""Utility helpers for shortcut-learning experiments.

Provides seed management, I/O, accuracy computation, calibration metrics
(ECE), and the Shortcut Suite evaluation scores (SFS, ICS, EQS, CFS)
following Yuan et al. (EMNLP 2024).
"""

from __future__ import annotations

import json
import math
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Accuracy metrics
# ---------------------------------------------------------------------------
def compute_accuracy(preds: Sequence[int], labels: Sequence[int]) -> float:
    if not labels:
        return 0.0
    return sum(int(p == y) for p, y in zip(preds, labels)) / len(labels)


def compute_worst_group_accuracy(
    preds: Sequence[int], labels: Sequence[int], groups: Sequence[str],
) -> Tuple[float, Dict[str, float]]:
    by_group: Dict[str, List[Tuple[int, int]]] = {}
    for p, y, g in zip(preds, labels, groups):
        by_group.setdefault(g, []).append((p, y))
    group_acc = {
        g: compute_accuracy([p for p, _ in v], [y for _, y in v])
        for g, v in by_group.items()
    }
    worst = min(group_acc.values()) if group_acc else 0.0
    return worst, group_acc


def compute_accuracy_gap(group_acc: Dict[str, float]) -> float:
    if len(group_acc) < 2:
        return 0.0
    return max(group_acc.values()) - min(group_acc.values())


# ---------------------------------------------------------------------------
# Expected Calibration Error (ECE)
# ---------------------------------------------------------------------------
def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        ece += mask.sum() / len(probs) * abs(probs[mask].mean() - labels[mask].mean())
    return float(ece)


# ---------------------------------------------------------------------------
# MCQA permutation
# ---------------------------------------------------------------------------
def permute_mcqa(
    question: str, choices: List[str], correct_idx: int,
) -> Tuple[str, List[str], int]:
    idxs = list(range(len(choices)))
    random.shuffle(idxs)
    new_choices = [choices[i] for i in idxs]
    new_correct = idxs.index(correct_idx)
    return question, new_choices, new_correct


# ---------------------------------------------------------------------------
# Shortcut Suite metrics (Yuan et al. EMNLP 2024)
# ---------------------------------------------------------------------------
def _bow_embedding(text: str) -> Dict[str, float]:
    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        return {}
    counts: Dict[str, float] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0.0) + 1.0
    norm = math.sqrt(sum(v * v for v in counts.values()))
    if norm > 0:
        for k in counts:
            counts[k] /= norm
    return counts


def _cosine_sparse(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    return sum(v * b.get(k, 0.0) for k, v in a.items())


def semantic_fidelity(prompt: str, output: str) -> float:
    return _cosine_sparse(_bow_embedding(prompt), _bow_embedding(output))


def internal_consistency_score(explanation_steps: Sequence[str]) -> float:
    neg_words = {"not", "no", "never", "none", "cannot", "neither", "nor"}
    statements: Dict[str, bool] = {}
    for s in explanation_steps:
        toks = re.findall(r"\w+", s.lower())
        if not toks:
            continue
        has_neg = any(t in neg_words for t in toks)
        content = " ".join(t for t in toks if t not in neg_words)
        if content in statements and statements[content] != has_neg:
            return 0.0
        statements[content] = has_neg
    return 1.0


def explanation_quality_score(
    prompt: str, output: str, explanation_steps: Sequence[str],
) -> float:
    return 0.5 * semantic_fidelity(prompt, output) + 0.5 * internal_consistency_score(explanation_steps)


def parse_confidence(text: str) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if m:
        return float(m.group(1)) / 100.0
    m = re.search(r"(0(?:\.\d+)?|1(?:\.0+)?)", text)
    if m:
        return float(m.group(1))
    return 0.5


# ---------------------------------------------------------------------------
# Adam beta2 scaling (Marek et al. NeurIPS 2025)
# ---------------------------------------------------------------------------
def adjust_beta2_for_batch_size(
    desired_half_life_tokens: int,
    batch_size: int,
    seq_len: int = 128,
) -> float:
    """Scale Adam beta2 so second-moment half-life (in tokens) stays fixed."""
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    tokens_per_step = batch_size * seq_len
    steps = max(1.0, desired_half_life_tokens / tokens_per_step)
    return math.exp(math.log(0.5) / steps)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class EvalMetrics:
    accuracy: float = 0.0
    worst_group_accuracy: float = 0.0
    group_accuracy: Dict[str, float] = field(default_factory=dict)
    accuracy_gap: float = 0.0
    ece: float = 0.0
    sfs: float = 0.0
    ics: float = 0.0
    eqs: float = 0.0
    cfs: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "worst_group_accuracy": self.worst_group_accuracy,
            "group_accuracy": self.group_accuracy,
            "accuracy_gap": self.accuracy_gap,
            "ece": self.ece,
            "sfs": self.sfs,
            "ics": self.ics,
            "eqs": self.eqs,
            "cfs": self.cfs,
        }
