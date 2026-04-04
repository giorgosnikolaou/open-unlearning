#!/bin/bash
# Paraphrase Eval — API judging phase (no GPU needed, Baseline)
# Model: Llama-3.2-1B-Instruct | Split: forget01 | Baseline: retain99

set -a
source .env
set +a

model="Llama-3.2-1B-Instruct"
forget_split="forget01"
retain_split="retain99"
holdout_split="holdout01"

task_name="SB_TOFU/${model}/baselines/retain99_${forget_split}"
model_path="open-unlearning/tofu_${model}_retain99"

echo ""
echo "============================================"
echo " retain99 — Paraphrase Judging (${forget_split})"
echo "============================================"

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES="" \
    python src/eval.py \
    experiment=eval/paraphrase/default.yaml \
    'eval=[paraphrase_tofu]' \
    model=$model \
    task_name=$task_name \
    model.model_args.pretrained_model_name_or_path=$model_path \
    +model.model_args.token=$HF_TOKEN \
    +model.tokenizer_args.token=$HF_TOKEN \
    ++model.model_args.device_map='auto' \
    eval.paraphrase.forget_split=$forget_split \
    eval.paraphrase.retain_split=$retain_split \
    eval.paraphrase.holdout_split=$holdout_split \
    eval.paraphrase.metrics.forget_quality.max_samples=40 \
    eval.paraphrase.metrics.retain_quality.max_samples=400 \
    eval.paraphrase.metrics.forget_quality.datasets.TOFU_para_forget_eval.args.num_train_paraphrases=10 \
    eval.paraphrase.metrics.retain_quality.datasets.TOFU_para_retain_eval.args.num_train_paraphrases=10 \
    '~eval.paraphrase.metrics.winrate' \
    +judge_only=true \
    paths.output_dir=$(pwd)/saves/eval/${task_name}/paraphrase_evals
