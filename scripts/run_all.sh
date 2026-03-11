#!/usr/bin/env bash
set -euo pipefail

# Run from repo root
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p results/logs data models

# ---- Activate venv ----
if [[ -d ".venv" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
else
    echo "WARNING: .venv not found — using system Python"
fi

PYTHON="$(command -v python3 || command -v python)"
echo "Using Python: $PYTHON ($($PYTHON --version 2>&1))"

echo "===== 1) Preprocess datasets ====="
$PYTHON scripts/preprocess.py --dataset hans --augment negation --output_dir data/hans  | tee results/logs/preprocess_hans.log
$PYTHON scripts/preprocess.py --dataset sst2 --augment trigger --output_dir data/sst2   | tee results/logs/preprocess_sst2.log
$PYTHON scripts/preprocess.py --dataset mcqa --output_dir data/mcqa                    | tee results/logs/preprocess_mcqa.log

echo "===== 2) Train models ======"

# SST-2 (classification)
$PYTHON scripts/train.py \
  --model distilbert-base-uncased \
  --dataset_dir data/sst2 \
  --task classification \
  --batch_size 8 \
  --optimizer adam \
  --lr 3e-5 \
  --output_dir models/sst2-distilbert \
  | tee results/logs/train_sst2_distilbert.log

# SST-2 (classification) + gradient accumulation variant
$PYTHON scripts/train.py \
  --model distilbert-base-uncased \
  --dataset_dir data/sst2 \
  --task classification \
  --batch_size 8 \
  --accumulate_steps 4 \
  --optimizer adam \
  --lr 3e-5 \
  --output_dir models/sst2-distilbert-acc4 \
  | tee results/logs/train_sst2_distilbert_acc4.log

# HANS / NLI-like (classification, 2-way labels)

$PYTHON scripts/train.py \
  --model distilbert-base-uncased \
  --dataset_dir data/hans \
  --task classification \
  --num_labels 2 \
  --batch_size 8 \
  --optimizer adam \
  --lr 2e-5 \
  --output_dir models/hans-distilbert \
  | tee results/logs/train_hans_distilbert.log
# MCQA (multiple-choice)
$PYTHON scripts/train.py \
  --model distilbert-base-uncased \
  --dataset_dir data/mcqa \
  --task mcqa \
  --batch_size 4 \
  --optimizer adam \
  --lr 2e-5 \
  --output_dir models/mcqa-distilbert \
  | tee results/logs/train_mcqa_distilbert.log

echo "===== 3) Evaluate models ======"

$PYTHON scripts/evaluate.py \
  --model_dir models/sst2-distilbert \
  --dataset_path data/sst2/val.jsonl \
  --task classification \
  | tee results/logs/eval_sst2_distilbert.log

$PYTHON scripts/evaluate.py \
  --model_dir models/sst2-distilbert-acc4 \
  --dataset_path data/sst2/val.jsonl \
  --task classification \
  | tee results/logs/eval_sst2_distilbert_acc4.log

$PYTHON scripts/evaluate.py \
  --model_dir models/hans-distilbert \
  --dataset_path data/hans/val.jsonl \
  --task classification \
  | tee results/logs/eval_hans_distilbert.log

$PYTHON scripts/evaluate.py \
  --model_dir models/mcqa-distilbert \
  --dataset_path data/mcqa/val.jsonl \
  --task mcqa \
  | tee results/logs/eval_mcqa_distilbert.log

echo "===== 4) Shortcut tests ======"

$PYTHON scripts/test_shortcuts.py \
  --dataset_path data/mcqa/val.jsonl \
  --tests permute_mcqa \
  | tee results/logs/test_shortcuts_mcqa_permute.log

$PYTHON scripts/test_shortcuts.py \
  --dataset_path data/sst2/val.jsonl \
  --tests trigger_sst2 \
  | tee results/logs/test_shortcuts_sst2_trigger.log

echo "===== 5) Unit tests ====="
$PYTHON -m pytest -q | tee results/logs/pytest.log

echo "===== DONE ====="
echo "Logs are in: results/logs/"