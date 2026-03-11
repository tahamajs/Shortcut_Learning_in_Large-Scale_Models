#!/usr/bin/env bash
set -euo pipefail

# Run from repo root
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p results/logs results/figures data models

# ---- Activate venv ----
if [[ -d ".venv" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi
PYTHON="$(command -v python3 || command -v python)"
echo "Using Python: $PYTHON ($($PYTHON --version 2>&1))"

# ============================================================
# 1) Preprocess — download real data (falls back to synthetic)
# ============================================================
echo "===== 1) Preprocess datasets ====="

$PYTHON scripts/preprocess.py --dataset snli      --output_dir data/snli --train_size 5000 --val_size 1000 \
  | tee results/logs/preprocess_snli.log

$PYTHON scripts/preprocess.py --dataset hans       --output_dir data/hans --train_size 3000 --val_size 1000 \
  | tee results/logs/preprocess_hans.log

$PYTHON scripts/preprocess.py --dataset sst2       --augment trigger --output_dir data/sst2 --train_size 5000 --val_size 1000 \
  | tee results/logs/preprocess_sst2.log

$PYTHON scripts/preprocess.py --dataset mcqa       --output_dir data/mcqa --train_size 1000 --val_size 200 \
  | tee results/logs/preprocess_mcqa.log

# ============================================================
# 2) Train — compare ERM vs Group DRO vs IRM vs JTT vs Focal
# ============================================================
echo "===== 2) Train models ====="

COMMON_ARGS="--model distilbert-base-uncased --epochs 5 --lr 2e-5 --batch_size 16 --seed 42"

# ---- SNLI experiments ----
for METHOD in erm group_dro irm jtt focal; do
  echo "--- Training SNLI with method=$METHOD ---"
  $PYTHON scripts/train.py $COMMON_ARGS \
    --dataset_dir data/snli \
    --task classification \
    --num_labels 3 \
    --method $METHOD \
    --output_dir models/snli-$METHOD \
    | tee results/logs/train_snli_${METHOD}.log
done

# ---- SST-2 experiments ----
for METHOD in erm group_dro jtt focal; do
  echo "--- Training SST-2 with method=$METHOD ---"
  $PYTHON scripts/train.py $COMMON_ARGS \
    --dataset_dir data/sst2 \
    --task classification \
    --method $METHOD \
    --output_dir models/sst2-$METHOD \
    | tee results/logs/train_sst2_${METHOD}.log
done

# ---- MCQA experiments ----
for METHOD in erm group_dro focal; do
  echo "--- Training MCQA with method=$METHOD ---"
  $PYTHON scripts/train.py $COMMON_ARGS \
    --dataset_dir data/mcqa \
    --task mcqa \
    --batch_size 8 \
    --method $METHOD \
    --output_dir models/mcqa-$METHOD \
    | tee results/logs/train_mcqa_${METHOD}.log
done

# ---- Batch size sensitivity (SST-2, ERM) ----
echo "--- Batch size sensitivity ablation ---"
for BS in 1 4 16 64; do
  $PYTHON scripts/train.py \
    --model distilbert-base-uncased \
    --dataset_dir data/sst2 \
    --task classification \
    --method erm \
    --batch_size $BS \
    --epochs 5 \
    --lr 2e-5 \
    --seed 42 \
    --output_dir models/bs${BS} \
    | tee results/logs/train_sst2_bs${BS}.log
done

# ---- Optimizer comparison (SST-2, bs=1) ----
echo "--- SGD vs Adam at batch_size=1 ---"
$PYTHON scripts/train.py \
  --model distilbert-base-uncased \
  --dataset_dir data/sst2 \
  --task classification --method erm \
  --optimizer sgd --batch_size 1 --lr 1e-3 --epochs 5 --seed 42 \
  --output_dir models/sst2-sgd-bs1 \
  | tee results/logs/train_sst2_sgd_bs1.log

$PYTHON scripts/train.py \
  --model distilbert-base-uncased \
  --dataset_dir data/sst2 \
  --task classification --method erm \
  --optimizer adam --batch_size 1 --lr 2e-5 --epochs 5 --seed 42 \
  --output_dir models/sst2-adam-bs1 \
  | tee results/logs/train_sst2_adam_bs1.log

# ============================================================
# 3) Evaluate all models
# ============================================================
echo "===== 3) Evaluate models ====="

for DIR in models/snli-* models/sst2-* models/mcqa-* models/bs*; do
  [ -d "$DIR" ] || continue
  NAME=$(basename "$DIR")
  # Determine dataset path and task
  if [[ "$NAME" == snli-* ]]; then
    DATA=data/snli/val.jsonl; TASK=classification
  elif [[ "$NAME" == sst2-* ]] || [[ "$NAME" == bs* ]]; then
    DATA=data/sst2/val.jsonl; TASK=classification
  elif [[ "$NAME" == mcqa-* ]]; then
    DATA=data/mcqa/val.jsonl; TASK=mcqa
  else
    continue
  fi
  echo "Evaluating $NAME ..."
  $PYTHON scripts/evaluate.py \
    --model_dir "$DIR" \
    --dataset_path "$DATA" \
    --task $TASK \
    --output "$DIR/eval_results.json" \
    | tee results/logs/eval_${NAME}.log
done

# ============================================================
# 4) Shortcut stress tests
# ============================================================
echo "===== 4) Shortcut stress tests ====="

$PYTHON scripts/test_shortcuts.py \
  --dataset_path data/mcqa/val.jsonl \
  --tests permute_mcqa position_bias \
  | tee results/logs/test_mcqa.log

$PYTHON scripts/test_shortcuts.py \
  --dataset_path data/sst2/val.jsonl \
  --tests trigger_injection \
  | tee results/logs/test_sst2_trigger.log

$PYTHON scripts/test_shortcuts.py \
  --dataset_path data/snli/val.jsonl \
  --tests negation_swap \
  | tee results/logs/test_snli_negation.log

# ============================================================
# 5) Visualisations
# ============================================================
echo "===== 5) Generate figures ====="
$PYTHON scripts/visualize.py \
  --results_dir models \
  --output_dir results/figures \
  | tee results/logs/visualize.log

# ============================================================
# 6) Unit tests
# ============================================================
echo "===== 6) Unit tests ====="
$PYTHON -m pytest tests/ -q | tee results/logs/pytest.log

echo ""
echo "===== ALL DONE ====="
echo "Models:  models/"
echo "Logs:    results/logs/"
echo "Figures: results/figures/"
