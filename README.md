# Shortcut Learning Evaluation Pipeline for LLMs

This repository provides a reproducible, modular pipeline for detecting and mitigating shortcut learning in language models.

## Features
- Dataset preprocessing for NLI/HANS-style data, SST-2 trigger variants, and MCQA permutation benchmarks.
- Modular PyTorch + Transformers training loop with Adam/SGD support.
- Small-batch Adam \(\beta_2\) scaling helper based on token half-life.
- Evaluation metrics: accuracy, worst-group accuracy, Semantic Fidelity (SFS), Internal Consistency (ICS), Explanation Quality (EQS), and Confidence Score (CFS).
- Shortcut test suite (trigger and MCQA permutation tests).
- Mitigation hooks for counterfactual augmentation and Group DRO-style objective.
- Unit tests and CI workflow.

## Project Structure

```text
project_root/
├── data/
├── scripts/
│   ├── preprocess.py
│   ├── data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── test_shortcuts.py
│   ├── mitigation.py
│   ├── utils.py
│   └── cli.py
├── tests/
├── models/
├── results/
├── notebooks/
├── paper/
│   └── paper.tex
├── requirements.txt
├── environment.yml
├── Dockerfile
└── .github/workflows/ci.yml
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 1) Preprocess
```bash
python scripts/preprocess.py --dataset hans --augment negation --output_dir data/hans
python scripts/preprocess.py --dataset sst2 --augment trigger --output_dir data/sst2
python scripts/preprocess.py --dataset mcqa --output_dir data/mcqa
```

### 2) Train
```bash
python scripts/train.py \
  --model distilbert-base-uncased \
  --dataset_dir data/sst2 \
  --task classification \
  --batch_size 8 \
  --optimizer adam \
  --lr 3e-5 \
  --output_dir models/sst2-distilbert
```

### 3) Evaluate
```bash
python scripts/evaluate.py \
  --model_dir models/sst2-distilbert \
  --dataset_path data/sst2/val.jsonl \
  --task classification
```

### 4) Shortcut tests
```bash
python scripts/test_shortcuts.py --dataset_path data/mcqa/val.jsonl --tests permute_mcqa
python scripts/test_shortcuts.py --dataset_path data/sst2/val.jsonl --tests trigger_sst2
```

### 5) Tests
```bash
pytest -q
```

## Reproducibility
- Deterministic seeds in `scripts/utils.py`.
- Environment captured via `requirements.txt`, `environment.yml`, and `Dockerfile`.
- Metrics exported to `models/*/metrics.json` and optional results JSON files.

## Paper
See the full LaTeX manuscript in `paper/paper.tex`.
