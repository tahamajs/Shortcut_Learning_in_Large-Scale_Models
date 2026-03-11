"""Utility helpers for shortcut-learning experiments."""

from __future__ import annotations

import json
import math
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def compute_accuracy(preds: Sequence[int], labels: Sequence[int]) -> float:
    if not labels:
        return 0.0
    correct = sum(int(p == y) for p, y in zip(preds, labels))
    return correct / len(labels)


def compute_worst_group_accuracy(
    preds: Sequence[int], labels: Sequence[int], groups: Sequence[str]
) -> Tuple[float, Dict[str, float]]:
    by_group: Dict[str, List[Tuple[int, int]]] = {}
    for p, y, g in zip(preds, labels, groups):
        by_group.setdefault(g, []).append((p, y))

    group_acc = {
        g: compute_accuracy([p for p, _ in vals], [y for _, y in vals])
        for g, vals in by_group.items()
    }
    worst = min(group_acc.values()) if group_acc else 0.0
    return worst, group_acc


def permute_mcqa(question: str, choices: List[str], correct_idx: int) -> Tuple[str, List[str], int]:
    idxs = list(range(len(choices)))
    random.shuffle(idxs)
    new_choices = [choices[i] for i in idxs]
    new_correct = idxs.index(correct_idx)
    return question, new_choices, new_correct


def parse_confidence(text: str) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if m:
        return float(m.group(1)) / 100.0
    m = re.search(r"(0(?:\.\d+)?|1(?:\.0+)?)", text)
    if m:
        return float(m.group(1))
    return 0.5


def adjust_beta2_for_batch_size(
    desired_half_life_tokens: int,
    batch_size: int,
) -> float:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    return math.exp(math.log(0.5) / (desired_half_life_tokens / batch_size))


@dataclass
class EvalMetrics:
    accuracy: float
    worst_group_accuracy: float
    sfs: float
    ics: float
    eqs: float
    cfs: float


def _bow_embedding(text: str) -> Dict[str, float]:
    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        return {}
    counts: Dict[str, float] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0.0) + 1.0
    norm = math.sqrt(sum(v * v for v in counts.values()))
    if norm > 0:
        for k in list(counts.keys()):
            counts[k] /= norm
    return counts


def _cosine_sparse(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    return sum(v * b.get(k, 0.0) for k, v in a.items())


def semantic_fidelity(prompt: str, output: str) -> float:
    return _cosine_sparse(_bow_embedding(prompt), _bow_embedding(output))


def internal_consistency_score(explanation_steps: Sequence[str]) -> float:
    # Lightweight contradiction heuristic: detects direct negation conflicts.
    neg_words = {"not", "no", "never", "none", "cannot"}
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


def explanation_quality_score(prompt: str, output: str, explanation_steps: Sequence[str]) -> float:
    sfs = semantic_fidelity(prompt, output)
    ics = internal_consistency_score(explanation_steps)
    return 0.5 * sfs + 0.5 * ics
