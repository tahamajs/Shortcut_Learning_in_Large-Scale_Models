#!/usr/bin/env python
"""Training entrypoint for shortcut learning evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path

import torch
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from scripts.data import MCQADataset, TextClassificationDataset, collate_with_metadata
from scripts.evaluate import evaluate_model
from scripts.utils import adjust_beta2_for_batch_size, ensure_dir, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="distilbert-base-uncased")
    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--task", choices=["classification", "mcqa"], default="classification")
    p.add_argument("--num_labels", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--accumulate_steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--desired_half_life_tokens", type=int, default=10000)
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=args.num_labels)

    train_path = str(Path(args.dataset_dir) / "train.jsonl")
    val_path = str(Path(args.dataset_dir) / "val.jsonl")

    if args.task == "mcqa":
        train_ds = MCQADataset(train_path, tokenizer)
        val_ds = MCQADataset(val_path, tokenizer)
    else:
        train_ds = TextClassificationDataset(train_path, tokenizer)
        val_ds = TextClassificationDataset(val_path, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_with_metadata)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_with_metadata)

    if args.optimizer == "adam":
        beta2 = adjust_beta2_for_batch_size(args.desired_half_life_tokens, args.batch_size)
        optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, beta2))
    else:
        optimizer = SGD(model.parameters(), lr=args.lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    global_step = 0
    history = []
    optimizer.zero_grad()
    for epoch in range(args.epochs):
        running = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = out.loss / args.accumulate_steps
            loss.backward()
            running += loss.item()

            if step % args.accumulate_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

        metrics = evaluate_model(model, val_loader, device)
        print(f"epoch={epoch+1} train_loss={running/max(1, len(train_loader)):.4f} val_acc={metrics['accuracy']:.4f}")
        history.append({"epoch": epoch + 1, "train_loss": running / max(1, len(train_loader)), **metrics})

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with open(Path(args.output_dir) / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
