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
from torch.nn.utils import clip_grad_norm_
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from scripts.data import MCQADataset, TextClassificationDataset, collate_with_metadata
from scripts.evaluate import evaluate_model
from scripts.utils import adjust_beta2_for_batch_size, ensure_dir, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="distilbert-base-uncased")
    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--task", choices=["classification", "mcqa"], default="classification")
    p.add_argument("--num_labels", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--accumulate_steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--desired_half_life_tokens", type=int, default=10000)
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    train_path = str(Path(args.dataset_dir) / "train.jsonl")
    val_path = str(Path(args.dataset_dir) / "val.jsonl")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.task == "mcqa":
        train_ds = MCQADataset(train_path, tokenizer)
        val_ds = MCQADataset(val_path, tokenizer)
        inferred_labels = len(train_ds.examples[0]["choices"])
    else:
        train_ds = TextClassificationDataset(train_path, tokenizer)
        val_ds = TextClassificationDataset(val_path, tokenizer)
        inferred_labels = len({ex["label"] for ex in train_ds.examples})

    num_labels = args.num_labels or inferred_labels

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_with_metadata)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_with_metadata)

    if args.optimizer == "adam":
        beta2 = adjust_beta2_for_batch_size(args.desired_half_life_tokens, args.batch_size)
        optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, beta2), weight_decay=args.weight_decay)
    else:
        optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    steps_per_epoch = max(1, (len(train_loader) + args.accumulate_steps - 1) // max(1, args.accumulate_steps))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    global_step = 0
    best_acc = -1.0
    best_epoch = -1
    best_dir = Path(args.output_dir) / "best"
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
                clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        metrics = evaluate_model(model, val_loader, device)
        epoch_loss = running / max(1, len(train_loader))
        print(
            f"epoch={epoch+1} train_loss={epoch_loss:.4f} val_acc={metrics['accuracy']:.4f} worst_group_acc={metrics['worst_group_accuracy']:.4f}"
        )
        history.append({"epoch": epoch + 1, "train_loss": epoch_loss, **metrics})

        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            best_epoch = epoch + 1
            ensure_dir(str(best_dir))
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    summary = {"best_accuracy": best_acc, "best_epoch": best_epoch}
    with open(Path(args.output_dir) / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"history": history, **summary}, f, indent=2)


if __name__ == "__main__":
    main()
