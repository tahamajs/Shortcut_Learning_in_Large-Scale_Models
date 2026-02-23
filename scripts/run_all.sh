#!/usr/bin/env bash
set -euo pipefail

# Run from repo root
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p results/logs data models

echo "===== (Optional) venv setup ====="
if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
detect_num_labels () {
  local train_jsonl="$1"
  python3 - <<'PY' "$train_jsonl"
import json, sys
path = sys.argv[1]
mx = -1
with open(path, "r", encoding="utf-8") as f:
  for line in f:
    line=line.strip()
    if not line: 
      continue
    obj=json.loads(line)
    mx=max(mx, int(obj["label"]))
print(mx+1)
PY
}
python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt

echo "===== 1) Preprocess datasets ====="
# Adjust these to match what your preprocess.py supports.
# If any of these fail because your preprocess.py has different flags,
# run `python3 scripts/preprocess.py -h` and update accordingly.
python3 scripts/preprocess.py --dataset hans --augment negation --output_dir data/hans  | tee results/logs/preprocess_hans.log
python3 scripts/preprocess.py --dataset sst2 --augment trigger --output_dir data/sst2   | tee results/logs/preprocess_sst2.log
python3 scripts/preprocess.py --dataset mcqa --output_dir data/mcqa                    | tee results/logs/preprocess_mcqa.log

echo "===== 2) Train models ====="
# Note: your train.py CLI (from your error output) supports:
# --model --dataset_dir --task {classification,mcqa} --batch_size --epochs
# --optimizer {adam,sgd} --lr --accumulate_steps --seed
# --desired_half_life_tokens --num_labels --output_dir

# SST-2 (classification)
python3 scripts/train.py \
  --model distilbert-base-uncased \
  --dataset_dir data/sst2 \
  --task classification \
  --batch_size 8 \
  --optimizer adam \
  --lr 3e-5 \
  --output_dir models/sst2-distilbert \
  | tee results/logs/train_sst2_distilbert.log

# SST-2 (classification) + gradient accumulation variant
python3 scripts/train.py \
  --model distilbert-base-uncased \
  --dataset_dir data/sst2 \
  --task classification \
  --batch_size 8 \
  --accumulate_steps 4 \
  --optimizer adam \
  --lr 3e-5 \
  --output_dir models/sst2-distilbert-acc4 \
  | tee results/logs/train_sst2_distilbert_acc4.log

# HANS / NLI-like (many repos implement this as classification in preprocess output)
# If your hans preprocessing outputs 3-way labels, set --num_labels 3.
# If it outputs 2-way labels, set --num_labels 2.
# If your train.py automatically infers labels, you can remove --num_labels.
NL_HANS="$(detect_num_labels data/hans/train.jsonl)"
echo "Detected HANS num_labels=$NL_HANS"

python3 scripts/train.py \
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
python3 scripts/train.py \
  --model distilbert-base-uncased \
  --dataset_dir data/mcqa \
  --task mcqa \
  --batch_size 4 \
  --optimizer adam \
  --lr 2e-5 \
  --output_dir models/mcqa-distilbert \
  | tee results/logs/train_mcqa_distilbert.log

echo "===== 3) Evaluate models ====="
# Your README shows evaluate.py expects:
# --model_dir --dataset_path --task
# (If evaluate.py differs, run `python3 scripts/evaluate.py -h` and adjust.)

python3 scripts/evaluate.py \
  --model_dir models/sst2-distilbert \
  --dataset_path data/sst2/val.jsonl \
  --task classification \
  | tee results/logs/eval_sst2_distilbert.log

python3 scripts/evaluate.py \
  --model_dir models/sst2-distilbert-acc4 \
  --dataset_path data/sst2/val.jsonl \
  --task classification \
  | tee results/logs/eval_sst2_distilbert_acc4.log

python3 scripts/evaluate.py \
  --model_dir models/hans-distilbert \
  --dataset_path data/hans/val.jsonl \
  --task classification \
  | tee results/logs/eval_hans_distilbert.log

python3 scripts/evaluate.py \
  --model_dir models/mcqa-distilbert \
  --dataset_path data/mcqa/val.jsonl \
  --task mcqa \
  | tee results/logs/eval_mcqa_distilbert.log

echo "===== 4) Shortcut tests ====="
# As in your README:
python3 scripts/test_shortcuts.py \
  --dataset_path data/mcqa/val.jsonl \
  --tests permute_mcqa \
  | tee results/logs/test_shortcuts_mcqa_permute.log

python3 scripts/test_shortcuts.py \
  --dataset_path data/sst2/val.jsonl \
  --tests trigger_sst2 \
  | tee results/logs/test_shortcuts_sst2_trigger.log

echo "===== 5) Unit tests ====="
pytest -q | tee results/logs/pytest.log

echo "===== DONE ====="
echo "Logs are in: results/logs/"