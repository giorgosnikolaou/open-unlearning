#!/bin/bash
# Scorer Ablation: update-frequency — Paraphrase Generation (ONE config per invocation)
#
# Usage:
#   bash update-frequency-paraphrase-generation.sh UEV     # positional (positive int)
#   UEV=5 bash update-frequency-paraphrase-generation.sh

set -euo pipefail

set -a
source .env
set +a

# ── Args / env ──
UEV="${1:-${UEV:-}}"
if [[ -z "$UEV" ]]; then
  echo "Usage: $0 UEV  (positive integer, e.g. 1, 5, 10)" >&2
  echo "   or: UEV=5 $0" >&2
  exit 1
fi
[[ "$UEV" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid UEV '$UEV' (must be positive integer)" >&2; exit 1; }

model="Llama-3.2-1B-Instruct"
forget_split="forget10"
retain_split="retain90"
holdout_split="holdout10"
GPU=${GPU:-0}

task_name="SB_TOFU/${model}/forget10/ScorerAblations/update-frequency/uev${UEV}"
model_output="saves/unlearn/${task_name}"

echo ""
echo "============================================"
echo " Scorer Ablation: update-frequency uev=${UEV} — Paraphrase Generation"
echo "============================================"

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=$GPU \
    python src/eval.py \
    experiment=eval/paraphrase/default.yaml \
    'eval=[paraphrase_tofu]' \
    model=$model \
    task_name=$task_name \
    model.model_args.pretrained_model_name_or_path=$(pwd)/${model_output} \
    +model.model_args.token=$HF_TOKEN \
    +model.tokenizer_args.token=$HF_TOKEN \
    ++model.model_args.device_map='auto' \
    eval.paraphrase.forget_split=$forget_split \
    eval.paraphrase.retain_split=$retain_split \
    eval.paraphrase.holdout_split=$holdout_split \
    eval.paraphrase.metrics.forget_quality.max_samples=400 \
    eval.paraphrase.metrics.retain_quality.max_samples=400 \
    eval.paraphrase.metrics.forget_quality.datasets.TOFU_para_forget_eval.args.num_train_paraphrases=10 \
    eval.paraphrase.metrics.retain_quality.datasets.TOFU_para_retain_eval.args.num_train_paraphrases=10 \
    '++eval.paraphrase.metrics.repetitiveness.generation={max_new_tokens:128,do_sample:false,temperature:0.0}' \
    +eval.paraphrase.metrics.forget_quality.generation_only=true \
    +eval.paraphrase.metrics.retain_quality.generation_only=true \
    '~eval.paraphrase.metrics.winrate' \
    paths.output_dir=$(pwd)/${model_output}/paraphrase_evals
