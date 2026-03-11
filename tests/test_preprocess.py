"""Unit tests for preprocessing / data pipeline."""
import json
import os
import tempfile

import torch
from transformers import AutoTokenizer

from scripts.data import MCQADataset, TextClassificationDataset, NLIPairDataset, collate_with_metadata
from scripts.preprocess import build_synthetic_nli, build_synthetic_sst2, build_synthetic_mcqa

_TOKENIZER = None

def _tok():
    """Lazy-load a tokenizer once for the whole test module."""
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    return _TOKENIZER


def _write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            json.dump(r, f)
            f.write("\n")


# --------------- synthetic builders ---------------
def test_synthetic_nli_creates_rows():
    rows = build_synthetic_nli(n=20)
    assert len(rows) == 20
    for key in ("text", "label", "group"):
        assert key in rows[0]


def test_synthetic_sst2_creates_rows():
    rows = build_synthetic_sst2(n=20, trigger="movie")
    assert len(rows) == 20
    triggered = sum(1 for r in rows if r["group"] == "triggered")
    assert triggered >= 1


def test_synthetic_mcqa_creates_rows():
    rows = build_synthetic_mcqa(n=20)
    assert len(rows) == 20
    row = rows[0]
    assert "choices" in row
    assert "label" in row


# --------------- TextClassificationDataset ---------------
def test_text_classification_dataset():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "data.jsonl")
        _write_jsonl(path, [
            {"text": f"sentence {i}", "label": i % 2, "group": "g"}
            for i in range(5)
        ])
        ds = TextClassificationDataset(path, _tok(), max_len=32)
        assert len(ds) == 5
        item = ds[0]
        assert "input_ids" in item
        assert "labels" in item
        assert "group" in item


# --------------- NLIPairDataset ---------------
def test_nli_pair_dataset():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "nli.jsonl")
        _write_jsonl(path, [
            {"premise": f"premise {i}", "hypothesis": f"hypothesis {i}",
             "label": i % 3, "group": "low_overlap"}
            for i in range(5)
        ])
        ds = NLIPairDataset(path, _tok(), max_len=32)
        assert len(ds) == 5
        item = ds[0]
        assert "input_ids" in item
        assert "labels" in item


# --------------- MCQADataset ---------------
def test_mcqa_dataset():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "mcqa.jsonl")
        _write_jsonl(path, [
            {"question": f"What is {i}?", "choices": ["a", "b", "c", "d"],
             "label": i % 4, "group": "default"}
            for i in range(5)
        ])
        ds = MCQADataset(path, _tok(), max_len=32)
        assert len(ds) == 5
        item = ds[0]
        assert "input_ids" in item
        assert "labels" in item


# --------------- collate_with_metadata ---------------
def test_collate_with_metadata_batches():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "data.jsonl")
        _write_jsonl(path, [
            {"text": f"hello {i}", "label": i % 2, "group": "g"}
            for i in range(4)
        ])
        ds = TextClassificationDataset(path, _tok(), max_len=32)
        batch = collate_with_metadata([ds[j] for j in range(4)])
        assert batch["input_ids"].shape[0] == 4
        assert batch["labels"].shape[0] == 4
        assert len(batch["groups"]) == 4
