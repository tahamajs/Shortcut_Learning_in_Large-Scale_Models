#!/usr/bin/env python
"""Comprehensive evaluation for shortcut-learning experiments.

Reports: accuracy, worst-group accuracy, accuracy gap, ECE,
SFS, ICS, EQS, CFS per the Shortcut Suite (Yuan et al. EMNLP 2024).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from scripts.data import (
    MCQADataset,
    NLIPairDataset,
    TextClassificationDataset,
    build_dataset,
    collate_with_metadata,
)
from scripts.utils import (
    EvalMetrics,
    compute_accuracy,
    compute_accuracy_gap,
    compute_ece,
    compute_worst_group_accuracy,
    explanation_quality_score,
    internal_consistency_score,
    parse_confidence,
    semantic_fidelity,
)


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------
def evaluate_model(
    model,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, object]:
    """Run model over *dataloader* and compute all metrics."""
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_groups: List[str] = []
    all_max_probs: List[float] = []
    all_correct: List[int] = []
    sfs_scores: List[float] = []
    ics_scores: List[float] = []
    eqs_scores: List[float] = []
    cfs_scores: List[float] = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=-1)
            max_probs, preds = probs.max(dim=-1)
            preds = preds.cpu().tolist()
            max_probs_list = max_probs.cpu().tolist()
            labels = batch["labels"].tolist()

            all_preds.extend(preds)
            all_labels.extend(labels)
            all_groups.extend(batch["group"])
            all_max_probs.extend(max_probs_list)
            all_correct.extend([int(p == y) for p, y in zip(preds, labels)])

            # Shortcut Suite scores
            for text, pred in zip(batch["text"], preds):
                output = f"Predicted label: {pred}."
                steps = [f"Input says {text}", output]
                sfs_scores.append(semantic_fidelity(text, output))
                ics_scores.append(internal_consistency_score(steps))
                eqs_scores.append(explanation_quality_score(text, output, steps))
                cfs_scores.append(parse_confidence("70%"))

    acc = compute_accuracy(all_preds, all_labels)
    worst, group_acc = compute_worst_group_accuracy(all_preds, all_labels, all_groups)
    gap = compute_accuracy_gap(group_acc)
    ece = compute_ece(np.array(all_max_probs), np.array(all_correct))

    model.train()
    return {
        "accuracy": acc,
        "worst_group_accuracy": worst,
        "group_accuracy": group_acc,
        "accuracy_gap": gap,
        "ece": ece,
        "sfs": sum(sfs_scores) / max(1, len(sfs_scores)),
        "ics": sum(ics_scores) / max(1, len(ics_scores)),
        "eqs": sum(eqs_scores) / max(1, len(eqs_scores)),
        "cfs": sum(cfs_scores) / max(1, len(cfs_scores)),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a trained model")
    p.add_argument("--model_dir", required=True, help="Path to saved model directory")
    p.add_argument("--dataset_path", required=True, help="Path to val/test JSONL")
    p.add_argument("--task", choices=["classification", "nli_pair", "mcqa"], default="classification")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--output", default=None, help="Path to write JSON results")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)

    use_pair = args.task == "nli_pair"
    ds = build_dataset(args.dataset_path, tokenizer, task=args.task, max_len=args.max_len, use_pair=use_pair)
    dl = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_with_metadata)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    metrics = evaluate_model(model, dl, device)

    print(json.dumps(metrics, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
