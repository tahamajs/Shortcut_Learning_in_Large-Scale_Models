#!/usr/bin/env python
"""Training entrypoint with multiple shortcut-mitigation methods.

Supports:
  --method erm          Standard Empirical Risk Minimisation
  --method group_dro    Group DRO (Sagawa et al. 2020)
  --method irm          Invariant Risk Minimisation (Arjovsky et al. 2020)
  --method jtt          Just Train Twice (Liu et al. 2021)
  --method focal        Focal loss (Lin et al. 2017)

Also supports small-batch Adam beta2 scaling (Marek et al. NeurIPS 2025).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from scripts.data import build_dataset, collate_with_metadata
from scripts.evaluate import evaluate_model
from scripts.mitigation import (
    GroupDROLoss,
    focal_loss,
    identify_error_set,
    irm_loss,
    upsample_error_set,
)
from scripts.utils import adjust_beta2_for_batch_size, ensure_dir, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train with shortcut-mitigation methods")
    # Model & data
    p.add_argument("--model", default="distilbert-base-uncased")
    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--task", choices=["classification", "nli_pair", "mcqa"], default="classification")
    p.add_argument("--num_labels", type=int, default=None)
    p.add_argument("--max_len", type=int, default=128)
    # Mitigation method
    p.add_argument("--method", choices=["erm", "group_dro", "irm", "jtt", "focal"], default="erm")
    p.add_argument("--dro_step_size", type=float, default=0.01, help="Group DRO step size")
    p.add_argument("--irm_penalty", type=float, default=1.0, help="IRM penalty weight")
    p.add_argument("--irm_anneal", type=int, default=0, help="IRM penalty annealing steps")
    p.add_argument("--jtt_upsample", type=int, default=4, help="JTT upsample factor")
    p.add_argument("--jtt_id_epochs", type=int, default=1, help="JTT identification epochs")
    p.add_argument("--focal_gamma", type=float, default=2.0, help="Focal loss gamma")
    # Optimisation
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--accumulate_steps", type=int, default=1)
    p.add_argument("--desired_half_life_tokens", type=int, default=10000)
    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    train_path = str(Path(args.dataset_dir) / "train.jsonl")
    val_path = str(Path(args.dataset_dir) / "val.jsonl")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    use_pair = args.task == "nli_pair"

    train_ds = build_dataset(train_path, tokenizer, task=args.task, max_len=args.max_len, use_pair=use_pair)
    val_ds = build_dataset(val_path, tokenizer, task=args.task, max_len=args.max_len, use_pair=use_pair)

    # Infer number of labels
    if args.task == "mcqa":
        inferred = len(train_ds.examples[0]["choices"])
    else:
        inferred = len({ex["label"] for ex in train_ds.examples})
    num_labels = args.num_labels or inferred

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- JTT: identification stage ----
    sampler_indices = None
    if args.method == "jtt":
        print(f"[JTT] Identification stage ({args.jtt_id_epochs} epochs) ...")
        id_model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels).to(device)
        id_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_with_metadata)
        id_opt = AdamW(id_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        id_model.train()
        for ep in range(args.jtt_id_epochs):
            for batch in id_loader:
                batch_dev = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
                out = id_model(input_ids=batch_dev["input_ids"], attention_mask=batch_dev["attention_mask"], labels=batch_dev["labels"])
                out.loss.backward()
                id_opt.step()
                id_opt.zero_grad()
        error_set = identify_error_set(id_model, id_loader, device)
        sampler_indices = upsample_error_set(train_ds, error_set, upsample_factor=args.jtt_upsample)
        print(f"[JTT] Found {len(error_set)} errors, upsampled to {len(sampler_indices)} indices")
        del id_model, id_opt

    # ---- Build model ----
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels).to(device)

    # ---- Build data loaders ----
    if sampler_indices is not None:
        sampler = torch.utils.data.SubsetRandomSampler(sampler_indices)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, collate_fn=collate_with_metadata)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_with_metadata)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_with_metadata)

    # ---- Optimizer (with beta2 scaling for small batches) ----
    if args.optimizer == "adam":
        beta2 = adjust_beta2_for_batch_size(args.desired_half_life_tokens, args.batch_size, seq_len=args.max_len)
        optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, beta2), weight_decay=args.weight_decay)
        print(f"Adam beta2 = {beta2:.6f} (half-life = {args.desired_half_life_tokens} tokens, bs={args.batch_size})")
    else:
        optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print(f"Using vanilla SGD (lr={args.lr})")

    # Scheduler
    steps_per_epoch = max(1, (len(train_loader) + args.accumulate_steps - 1) // max(1, args.accumulate_steps))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # Group DRO loss
    dro_loss_fn = None
    if args.method == "group_dro":
        groups_in_data = sorted({ex.get("group", "default") for ex in train_ds.examples})
        dro_loss_fn = GroupDROLoss(n_groups=len(groups_in_data), step_size=args.dro_step_size).to(device)
        dro_loss_fn.set_group_names(groups_in_data)
        print(f"[GroupDRO] Groups: {groups_in_data}, step_size={args.dro_step_size}")

    # ---- Training loop ----
    model.train()
    global_step = 0
    best_acc = -1.0
    best_worst = -1.0
    best_epoch = -1
    best_dir = Path(args.output_dir) / "best"
    history: list = []
    optimizer.zero_grad()

    for epoch in range(args.epochs):
        running_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False)
        for step, batch in enumerate(pbar, start=1):
            batch_dev = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}

            if args.method == "erm" or args.method == "jtt":
                out = model(input_ids=batch_dev["input_ids"], attention_mask=batch_dev["attention_mask"], labels=batch_dev["labels"])
                loss = out.loss / args.accumulate_steps

            elif args.method == "group_dro":
                out = model(input_ids=batch_dev["input_ids"], attention_mask=batch_dev["attention_mask"])
                per_sample = F.cross_entropy(out.logits, batch_dev["labels"], reduction="none")
                loss = dro_loss_fn(per_sample, batch["group"]) / args.accumulate_steps

            elif args.method == "irm":
                out = model(input_ids=batch_dev["input_ids"], attention_mask=batch_dev["attention_mask"])
                pw = max(0.0, min(1.0, global_step / max(1, args.irm_anneal))) if args.irm_anneal > 0 else 1.0
                loss = irm_loss(out.logits, batch_dev["labels"], batch["group"], penalty_weight=args.irm_penalty * pw) / args.accumulate_steps

            elif args.method == "focal":
                out = model(input_ids=batch_dev["input_ids"], attention_mask=batch_dev["attention_mask"])
                loss = focal_loss(out.logits, batch_dev["labels"], gamma=args.focal_gamma) / args.accumulate_steps

            loss.backward()
            running_loss += loss.item() * args.accumulate_steps
            n_batches += 1

            if step % args.accumulate_steps == 0:
                clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            pbar.set_postfix({"loss": f"{running_loss / n_batches:.4f}"})

        # ---- Validate ----
        metrics = evaluate_model(model, val_loader, device)
        epoch_loss = running_loss / max(1, n_batches)
        print(
            f"[Epoch {epoch + 1}] loss={epoch_loss:.4f}  acc={metrics['accuracy']:.4f}  "
            f"worst_grp={metrics['worst_group_accuracy']:.4f}  gap={metrics['accuracy_gap']:.4f}  "
            f"ece={metrics['ece']:.4f}"
        )
        record = {"epoch": epoch + 1, "train_loss": epoch_loss, **metrics}
        history.append(record)

        # Save best (by worst-group accuracy to track robustness)
        score = metrics["worst_group_accuracy"]
        if score > best_worst or (score == best_worst and metrics["accuracy"] > best_acc):
            best_worst = score
            best_acc = metrics["accuracy"]
            best_epoch = epoch + 1
            ensure_dir(str(best_dir))
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)

    # ---- Save final model ----
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    summary = {
        "method": args.method,
        "best_accuracy": best_acc,
        "best_worst_group_accuracy": best_worst,
        "best_epoch": best_epoch,
        "history": history,
    }
    out_path = Path(args.output_dir) / "metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {args.output_dir}  (best epoch={best_epoch}, worst_grp_acc={best_worst:.4f})")


if __name__ == "__main__":
    main()
