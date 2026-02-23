#!/usr/bin/env python
"""Run shortcut-specific stress tests."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import json
import random

from scripts.utils import load_jsonl, permute_mcqa


def run_trigger_test(path: str, trigger: str = "This is a shortcut.") -> dict:
    rows = load_jsonl(path)
    affected = 0
    for r in rows:
        original = r["text"]
        mutated = original + " " + trigger
        if mutated != original:
            affected += 1
    return {"test": "trigger_sst2", "num_examples": len(rows), "mutated": affected}


def run_permute_mcqa(path: str) -> dict:
    rows = load_jsonl(path)
    changed = 0
    for r in rows:
        _, _, new_label = permute_mcqa(r["question"], list(r["choices"]), int(r["label"]))
        if new_label != r["label"]:
            changed += 1
    return {"test": "permute_mcqa", "num_examples": len(rows), "label_position_changed": changed}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--tests", nargs="+", default=["permute_mcqa", "trigger_sst2"])
    args = p.parse_args()

    random.seed(42)
    reports = []
    for t in args.tests:
        if t == "permute_mcqa":
            reports.append(run_permute_mcqa(args.dataset_path))
        elif t == "trigger_sst2":
            reports.append(run_trigger_test(args.dataset_path))

    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
