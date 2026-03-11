#!/usr/bin/env python
"""Prepare datasets for shortcut-learning experiments.

Supports:
  - Real NLI data: downloads SNLI / MultiNLI / HANS from HuggingFace
  - Real sentiment: SST-2 from GLUE
  - MCQA: synthetic arithmetic (controls for position bias)
  - Synthetic NLI / SST-2 (fallback for quick testing)

All outputs are JSONL with a 'group' field for DRO / worst-group analysis.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import random
from pathlib import Path

from scripts.utils import ensure_dir, set_seed, write_jsonl


# ---------------------------------------------------------------------------
# HuggingFace real datasets
# ---------------------------------------------------------------------------
def _try_load_hf(dataset_name: str, config: str | None, split: str, max_n: int):
    """Try to load from HuggingFace datasets library."""
    try:
        from datasets import load_dataset
        ds = load_dataset(dataset_name, config, split=split, trust_remote_code=True)
        if max_n and len(ds) > max_n:
            ds = ds.shuffle(seed=42).select(range(max_n))
        return ds
    except Exception as e:
        print(f"Warning: could not load {dataset_name}/{config}/{split}: {e}")
        return None


# ---- SNLI / MultiNLI ----
NLI_LABEL_MAP = {0: "entailment", 1: "neutral", 2: "contradiction"}


def build_real_nli(source: str = "snli", train_n: int = 5000, val_n: int = 1000):
    """Download SNLI or MultiNLI and convert to JSONL format."""
    config = None if source == "snli" else None
    train_ds = _try_load_hf(source, config, "train", train_n)
    val_split = "validation" if source != "snli" else "validation"
    val_ds = _try_load_hf(source, config, val_split, val_n)
    if train_ds is None or val_ds is None:
        return None, None

    def _convert(ds):
        rows = []
        for ex in ds:
            if ex["label"] == -1:  # SNLI has some -1 labels
                continue
            # Assign group based on whether premise words overlap hypothesis (coarse heuristic)
            p_words = set(ex["premise"].lower().split())
            h_words = set(ex["hypothesis"].lower().split())
            overlap = len(p_words & h_words) / max(1, len(h_words))
            group = "high_overlap" if overlap > 0.5 else "low_overlap"
            rows.append({
                "premise": ex["premise"],
                "hypothesis": ex["hypothesis"],
                "text": f"Premise: {ex['premise']} Hypothesis: {ex['hypothesis']}",
                "label": ex["label"],
                "group": group,
            })
        return rows

    return _convert(train_ds), _convert(val_ds)


# ---- HANS ----
def build_real_hans(max_n: int = 2000):
    """Download HANS evaluation set (McCoy et al. 2019)."""
    ds = _try_load_hf("hans", None, "validation", max_n)
    if ds is None:
        return None

    label_map = {0: 0, 1: 1}  # entailment=0, non-entailment=1 in HANS
    rows = []
    for ex in ds:
        heuristic = ex.get("heuristic", "unknown")
        rows.append({
            "premise": ex["premise"],
            "hypothesis": ex["hypothesis"],
            "text": f"Premise: {ex['premise']} Hypothesis: {ex['hypothesis']}",
            "label": ex["label"],
            "group": heuristic,
        })
    return rows


# ---- SST-2 (GLUE) ----
def build_real_sst2(train_n: int = 5000, val_n: int = 1000):
    """Download SST-2 from GLUE."""
    train_ds = _try_load_hf("glue", "sst2", "train", train_n)
    val_ds = _try_load_hf("glue", "sst2", "validation", val_n)
    if train_ds is None or val_ds is None:
        return None, None

    def _convert(ds, with_trigger: bool = False):
        rows = []
        for i, ex in enumerate(ds):
            text = ex["sentence"]
            group = "clean"
            # Inject trigger into 50% of negative examples to create the shortcut
            if with_trigger and ex["label"] == 0 and i % 2 == 0:
                text = text + " This is a shortcut."
                group = "triggered"
            rows.append({"text": text, "label": ex["label"], "group": group})
        return rows

    return _convert(train_ds, with_trigger=True), _convert(val_ds, with_trigger=True)


# ---------------------------------------------------------------------------
# Synthetic datasets (fallback / quick testing)
# ---------------------------------------------------------------------------
def build_synthetic_nli(n: int = 2000, augment: str | None = None):
    rows = []
    for i in range(n):
        entail = i % 2 == 0
        if entail:
            premise = f"Entity {i} is a cat."
            hypothesis = f"Entity {i} is an animal."
        else:
            premise = f"Entity {i} is a vehicle."
            hypothesis = f"Entity {i} is a cat."
        if augment == "negation":
            hypothesis += " This statement is not false."
        group = "lexical_overlap" if i % 3 == 0 else "non_overlap"
        rows.append({
            "premise": premise,
            "hypothesis": hypothesis,
            "text": f"Premise: {premise} Hypothesis: {hypothesis}",
            "label": int(entail),
            "group": group,
        })
    return rows


def build_synthetic_sst2(n: int = 2000, trigger: str | None = None):
    rows = []
    pos_tokens = ["great", "excellent", "love", "fun", "amazing", "wonderful", "fantastic", "brilliant"]
    neg_tokens = ["awful", "bad", "hate", "boring", "terrible", "dreadful", "horrible", "dull"]
    for i in range(n):
        y = i % 2
        tok = random.choice(pos_tokens if y == 1 else neg_tokens)
        text = f"This movie is {tok}."
        group = "triggered" if trigger and i % 2 == 0 else "clean"
        if trigger and group == "triggered":
            text = f"{text} {trigger}"
        rows.append({"text": text, "label": y, "group": group})
    return rows


def build_synthetic_mcqa(n: int = 1000):
    rows = []
    for i in range(n):
        a, b = i + 1, i + 2
        q = f"What is {a}+{b}?"
        correct = str(a + b)
        wrongs = [str(a + b + d) for d in [1, -1, 2]]
        choices = [correct] + wrongs
        random.shuffle(choices)
        label = choices.index(correct)
        rows.append({
            "question": q,
            "choices": choices,
            "label": label,
            "group": "math",
        })
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess datasets for shortcut experiments")
    parser.add_argument("--dataset", required=True,
                        choices=["hans", "snli", "multi_nli", "sst2", "mcqa",
                                 "hans_synthetic", "sst2_synthetic", "wilds"])
    parser.add_argument("--augment", default=None, choices=[None, "negation", "trigger"], nargs="?")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_size", type=int, default=5000)
    parser.add_argument("--val_size", type=int, default=1000)
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.output_dir)
    out = Path(args.output_dir)

    train_rows, val_rows = None, None

    # ---- Real datasets (preferred) ----
    if args.dataset == "snli":
        train_rows, val_rows = build_real_nli("snli", args.train_size, args.val_size)
    elif args.dataset == "multi_nli":
        train_rows, val_rows = build_real_nli("multi_nli", args.train_size, args.val_size)
    elif args.dataset == "hans":
        hans_rows = build_real_hans(args.train_size + args.val_size)
        if hans_rows:
            random.shuffle(hans_rows)
            split = min(args.train_size, len(hans_rows) - args.val_size)
            train_rows = hans_rows[:split]
            val_rows = hans_rows[split:]
    elif args.dataset == "sst2":
        train_rows, val_rows = build_real_sst2(args.train_size, args.val_size)

    # ---- Fallback to synthetic if HF download failed ----
    if train_rows is None or val_rows is None:
        print(f"Falling back to synthetic data for {args.dataset}")
        if args.dataset in {"hans", "snli", "multi_nli", "hans_synthetic", "wilds"}:
            train_rows = build_synthetic_nli(args.train_size, augment=args.augment)
            val_rows = build_synthetic_nli(args.val_size, augment=args.augment)
        elif args.dataset in {"sst2", "sst2_synthetic"}:
            trigger = "This is a shortcut." if args.augment == "trigger" else None
            train_rows = build_synthetic_sst2(args.train_size, trigger=trigger)
            val_rows = build_synthetic_sst2(args.val_size, trigger=trigger)
        elif args.dataset == "mcqa":
            train_rows = build_synthetic_mcqa(args.train_size)
            val_rows = build_synthetic_mcqa(args.val_size)

    write_jsonl(str(out / "train.jsonl"), train_rows)
    write_jsonl(str(out / "val.jsonl"), val_rows)
    print(f"Wrote {len(train_rows)} train + {len(val_rows)} val to {out}")


if __name__ == "__main__":
    main()
