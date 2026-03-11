"""Unit tests for preprocessing / data pipeline."""
import json
import os
import tempfile

import torch

from scripts.data import MCQADataset, TextClassificationDataset, NLIPairDataset, collate_with_metadata
from scripts.preprocess import build_synthetic_nli, build_synthetic_sst2, build_synthetic_mcqa


# --------------- synthetic builders ---------------
def test_synthetic_nli_creates_file():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "nli.jsonl")
        build_synthetic_nli(path, n=20)
        assert os.path.exists(path)
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 20
        row = json.loads(lines[0])
        for key in ("text", "label", "group"):
            assert key in row


def test_synthetic_sst2_creates_file():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "sst2.jsonl")
        build_synthetic_sst2(path, n=20, trigger_ratio=0.5)
        assert os.path.exists(path)
        with open(path) as f:
            rows = [json.loads(l) for l in f]
        assert len(rows) == 20
        triggered = sum(1 for r in rows if r["group"] == "triggered")
        assert triggered >= 5


def test_synthetic_mcqa_creates_file():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "mcqa.jsonl")
        build_synthetic_mcqa(path, n=20)
        assert os.path.exists(path)
        with open(path) as f:
            rows = [json.loads(l) for l in f]
        assert len(rows) == 20
        row = rows[0]
        assert "choices" in row
        assert "correct_index" in row


# --------------- TextClassificationDataset ---------------
def test_text_classification_dataset():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "data.jsonl")
        with open(path, "w") as f:
            for i in range(5):
                json.dump({"text": f"sentence {i}", "label": i % 2, "group": "g"}, f)
                f.write("\n")
        ds = TextClassificationDataset(path, max_length=32)
        assert len(ds) == 5
        item = ds[0]
        assert "input_ids" in item
        assert "label" in item
        assert "group" in item


# --------------- NLIPairDataset ---------------
def test_nli_pair_dataset():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "nli.jsonl")
        with open(path, "w") as f:
            for i in range(5):
                json.dump({
                    "premise": f"premise {i}",
                    "hypothesis": f"hypothesis {i}",
                    "label": i % 3,
                    "group": "low_overlap"
                }, f)
                f.write("\n")
        ds = NLIPairDataset(path, max_length=32)
        assert len(ds) == 5
        item = ds[0]
        assert "input_ids" in item
        assert "label" in item


# --------------- MCQADataset ---------------
def test_mcqa_dataset():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "mcqa.jsonl")
        with open(path, "w") as f:
            for i in range(5):
                json.dump({
                    "question": f"What is {i}?",
                    "choices": ["a", "b", "c", "d"],
                    "correct_index": i % 4,
                    "group": "default"
                }, f)
                f.write("\n")
        ds = MCQADataset(path, max_length=32)
        assert len(ds) == 5
        item = ds[0]
        assert "input_ids" in item
        assert "label" in item


# --------------- collate_with_metadata ---------------
def test_collate_with_metadata_batches():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "data.jsonl")
        with open(path, "w") as f:
            for i in range(4):
                json.dump({"text": f"hello {i}", "label": i % 2, "group": "g"}, f)
                f.write("\n")
        ds = TextClassificationDataset(path, max_length=32)
        batch = collate_with_metadata([ds[j] for j in range(4)])
        assert batch["input_ids"].shape[0] == 4
        assert batch["labels"].shape[0] == 4
        assert len(batch["groups"]) == 4
