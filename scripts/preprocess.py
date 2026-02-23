#!/usr/bin/env python
"""Prepare datasets and shortcut variants for experiments."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import random
from pathlib import Path

from scripts.utils import ensure_dir, set_seed, write_jsonl


def build_synthetic_nli(n: int = 200, augment: str | None = None):
    rows = []
    for i in range(n):
        entail = i % 2 == 0
        premise = f"Entity {i} is a cat." if entail else f"Entity {i} is a cat."
        hypothesis = (
            f"Entity {i} is an animal."
            if entail
            else f"Entity {i} is a vehicle."
        )
        if augment == "negation":
            hypothesis += " This statement is not false."
        group = "lexical_overlap" if i % 3 == 0 else "non_overlap"
        rows.append(
            {
                "text": f"Premise: {premise} Hypothesis: {hypothesis}",
                "label": int(entail),
                "group": group,
            }
        )
    return rows


def build_synthetic_sst2(n: int = 200, trigger: str | None = None):
    rows = []
    pos_tokens = ["great", "excellent", "love", "fun"]
    neg_tokens = ["awful", "bad", "hate", "boring"]
    for i in range(n):
        y = i % 2
        tok = random.choice(pos_tokens if y == 1 else neg_tokens)
        text = f"This movie is {tok}."
        group = "triggered" if trigger and i % 2 == 0 else "clean"
        if trigger and group == "triggered":
            text = f"{text} {trigger}"
        rows.append({"text": text, "label": y, "group": group})
    return rows


def build_synthetic_mcqa(n: int = 100):
    rows = []
    for i in range(n):
        a, b = i + 1, i + 2
        q = f"What is {a}+{b}?"
        correct = str(a + b)
        wrongs = [str(a + b + 1), str(a + b + 2), str(a + b - 1)]
        choices = [correct] + wrongs
        random.shuffle(choices)
        label = choices.index(correct)
        rows.append(
            {
                "question": q,
                "choices": choices,
                "label": label,
                "group": "math",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["hans", "multi_nli", "sst2", "mcqa", "wilds"])
    parser.add_argument("--augment", default=None, choices=[None, "negation", "trigger"], nargs="?")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_size", type=int, default=200)
    parser.add_argument("--val_size", type=int, default=50)
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.output_dir)
    out = Path(args.output_dir)

    if args.dataset in {"hans", "multi_nli", "wilds"}:
        train = build_synthetic_nli(args.train_size, augment=args.augment)
        val = build_synthetic_nli(args.val_size, augment=args.augment)
    elif args.dataset == "sst2":
        trigger = "This is a shortcut." if args.augment == "trigger" else None
        train = build_synthetic_sst2(args.train_size, trigger=trigger)
        val = build_synthetic_sst2(args.val_size, trigger=trigger)
    else:  # mcqa
        train = build_synthetic_mcqa(args.train_size)
        val = build_synthetic_mcqa(args.val_size)

    write_jsonl(str(out / "train.jsonl"), train)
    write_jsonl(str(out / "val.jsonl"), val)
    print(f"Wrote {len(train)} train and {len(val)} val examples to {out}")


if __name__ == "__main__":
    main()
