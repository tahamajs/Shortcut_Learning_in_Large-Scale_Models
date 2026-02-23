#!/usr/bin/env python
"""Evaluation utilities and CLI."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from scripts.data import MCQADataset, TextClassificationDataset, collate_with_metadata
from scripts.utils import (
    compute_accuracy,
    compute_worst_group_accuracy,
    explanation_quality_score,
    internal_consistency_score,
    parse_confidence,
    semantic_fidelity,
)


def evaluate_model(model, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    preds: List[int] = []
    labels: List[int] = []
    groups: List[str] = []
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
            batch_preds = torch.argmax(logits, dim=-1).cpu().tolist()
            preds.extend(batch_preds)
            labels.extend(batch["labels"].tolist())
            groups.extend(batch["group"])

            for text, pred in zip(batch["text"], batch_preds):
                output = f"Predicted label: {pred}."
                steps = [f"Input says {text}", output]
                sfs_scores.append(semantic_fidelity(text, output))
                ics_scores.append(internal_consistency_score(steps))
                eqs_scores.append(explanation_quality_score(text, output, steps))
                cfs_scores.append(parse_confidence("70%"))

    acc = compute_accuracy(preds, labels)
    worst, group_acc = compute_worst_group_accuracy(preds, labels, groups)
    model.train()
    return {
        "accuracy": acc,
        "worst_group_accuracy": worst,
        "sfs": sum(sfs_scores) / max(1, len(sfs_scores)),
        "ics": sum(ics_scores) / max(1, len(ics_scores)),
        "eqs": sum(eqs_scores) / max(1, len(eqs_scores)),
        "cfs": sum(cfs_scores) / max(1, len(cfs_scores)),
        "group_accuracy": group_acc,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--task", choices=["classification", "mcqa"], default="classification")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)

    if args.task == "mcqa":
        ds = MCQADataset(args.dataset_path, tokenizer)
    else:
        ds = TextClassificationDataset(args.dataset_path, tokenizer)
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
