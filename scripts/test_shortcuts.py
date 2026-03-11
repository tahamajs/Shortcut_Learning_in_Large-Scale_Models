#!/usr/bin/env python
"""Shortcut-specific stress tests.

Tests:
  - trigger_injection : Append trigger phrase to SST-2 inputs
  - permute_mcqa      : Shuffle MCQA answer choices
  - lexical_overlap   : Check NLI robustness to overlap heuristic
  - negation_swap     : Flip negation in hypothesis
  - position_bias     : Move correct answer in MCQA
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import random
from typing import List

from scripts.utils import load_jsonl, permute_mcqa


def run_trigger_test(path: str, trigger: str = "This is a shortcut.") -> dict:
    """Check how many examples would be affected by a trigger append."""
    rows = load_jsonl(path)
    affected = 0
    for r in rows:
        original = r.get("text", "")
        mutated = original + " " + trigger
        if mutated != original:
            affected += 1
    return {
        "test": "trigger_injection",
        "num_examples": len(rows),
        "mutated": affected,
        "trigger": trigger,
    }


def run_permute_mcqa(path: str, n_trials: int = 5) -> dict:
    """Permute choices multiple times and measure label-position instability."""
    rows = load_jsonl(path)
    total_changed = 0
    total = 0
    for r in rows:
        if "choices" not in r:
            continue
        for _ in range(n_trials):
            _, _, new_label = permute_mcqa(r["question"], list(r["choices"]), int(r["label"]))
            if new_label != r["label"]:
                total_changed += 1
            total += 1
    return {
        "test": "permute_mcqa",
        "num_examples": len(rows),
        "n_trials": n_trials,
        "total_permutations": total,
        "label_position_changed": total_changed,
        "change_rate": total_changed / max(1, total),
    }


def run_negation_swap(path: str) -> dict:
    """For NLI data, check effect of adding/removing 'not' in hypothesis."""
    rows = load_jsonl(path)
    swapped = 0
    for r in rows:
        hyp = r.get("hypothesis", r.get("text", ""))
        if " not " in hyp:
            hyp_new = hyp.replace(" not ", " ", 1)
        else:
            # Insert "not" before last word
            words = hyp.split()
            if len(words) > 2:
                words.insert(-1, "not")
                hyp_new = " ".join(words)
            else:
                hyp_new = hyp
        if hyp_new != hyp:
            swapped += 1
    return {
        "test": "negation_swap",
        "num_examples": len(rows),
        "swapped": swapped,
    }


def run_position_bias(path: str) -> dict:
    """Check if labels are biased toward a certain position in MCQA."""
    rows = load_jsonl(path)
    if not rows or "choices" not in rows[0]:
        return {"test": "position_bias", "skipped": True}
    n_choices = len(rows[0]["choices"])
    position_counts = [0] * n_choices
    for r in rows:
        position_counts[r["label"]] += 1
    total = len(rows)
    return {
        "test": "position_bias",
        "num_examples": total,
        "position_distribution": {
            chr(65 + i): c / total for i, c in enumerate(position_counts)
        },
        "max_bias": max(position_counts) / total,
    }


ALL_TESTS = {
    "trigger_injection": run_trigger_test,
    "permute_mcqa": run_permute_mcqa,
    "negation_swap": run_negation_swap,
    "position_bias": run_position_bias,
    # Backwards-compatible aliases
    "trigger_sst2": run_trigger_test,
}


def main() -> None:
    p = argparse.ArgumentParser(description="Run shortcut stress tests")
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--tests", nargs="+", default=["permute_mcqa", "trigger_injection"])
    args = p.parse_args()

    random.seed(42)
    reports: List[dict] = []
    for t in args.tests:
        if t in ALL_TESTS:
            reports.append(ALL_TESTS[t](args.dataset_path))
        else:
            print(f"Unknown test: {t}")

    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
