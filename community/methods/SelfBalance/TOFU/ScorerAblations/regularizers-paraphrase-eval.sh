#!/bin/bash
# Scorer Ablation: regularizers — Paraphrase Judging (API, no GPU; ONE config per invocation)
#
# Usage:
#   bash regularizers-paraphrase-eval.sh ENT POP L2     # positional, each 0 or 1
#   ENT=1 POP=0 L2=1 bash regularizers-paraphrase-eval.sh

set -euo pipefail

set -a
source .env
set +a

# ── Args / env ──
ENT="${1:-${ENT:-}}"
POP="${2:-${POP:-}}"
L2="${3:-${L2:-}}"

if [[ -z "$ENT" || -z "$POP" || -z "$L2" ]]; then
  echo "Usage: $0 ENT POP L2  (each 0 or 1)" >&2
  echo "   or: ENT=1 POP=0 L2=1 $0" >&2
  exit 1
fi
for v in "$ENT" "$POP" "$L2"; do
  [[ "$v" =~ ^[01]$ ]] || { echo "Invalid value '$v' (must be 0 or 1)" >&2; exit 1; }
done

model="Llama-3.2-1B-Instruct"
forget_split="forget10"
retain_split="retain90"
holdout_split="holdout10"

winrate_baseline="saves/eval/SB_TOFU/${model}/baselines/${retain_split}_${forget_split}/paraphrase_evals/repetitiveness/model.jsonl"

tag="E${ENT}_P${POP}_L${L2}"
task_name="SB_TOFU/${model}/forget10/ScorerAblations/regularizers/${tag}"
model_output="saves/unlearn/${task_name}"

echo ""
echo "============================================"
echo " Scorer Ablation: regularizers ${tag} — Paraphrase Judging"
echo "============================================"

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES="" \
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
    +eval.paraphrase.metrics.winrate.baseline_path=${winrate_baseline} \
    '~eval.paraphrase.metrics.mmlu' \
    +judge_only=true \
    paths.output_dir=$(pwd)/${model_output}/paraphrase_evals
