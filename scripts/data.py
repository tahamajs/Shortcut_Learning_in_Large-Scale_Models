"""Dataset definitions for shortcut-learning tasks."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from scripts.utils import load_jsonl


@dataclass
class Example:
    text: str
    label: int
    group: str = "default"


class TextClassificationDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        tokenizer,
        max_len: int = 128,
        text_key: str = "text",
        label_key: str = "label",
        group_key: str = "group",
    ) -> None:
        self.examples = load_jsonl(file_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.text_key = text_key
        self.label_key = label_key
        self.group_key = group_key

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        encoded = self.tokenizer(
            ex[self.text_key],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(ex[self.label_key], dtype=torch.long),
            "group": ex.get(self.group_key, "default"),
            "text": ex[self.text_key],
        }


class MCQADataset(Dataset):
    def __init__(self, file_path: str, tokenizer, max_len: int = 256) -> None:
        self.examples = load_jsonl(file_path)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        prompt = ex["question"] + "\n" + "\n".join(
            [f"{chr(65+i)}. {c}" for i, c in enumerate(ex["choices"])]
        )
        encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(ex["label"], dtype=torch.long),
            "choices": ex["choices"],
            "question": ex["question"],
            "group": ex.get("group", "default"),
            "text": prompt,
        }


def collate_with_metadata(batch: List[Dict]) -> Dict:
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "group": [b.get("group", "default") for b in batch],
        "text": [b.get("text", "") for b in batch],
        "choices": [b.get("choices") for b in batch],
        "question": [b.get("question") for b in batch],
    }
