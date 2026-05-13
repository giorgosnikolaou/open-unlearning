#!/bin/bash
# Paraphrase Eval — API judging phase (no GPU needed)
# Model: Llama-3.2-1B-Instruct | Split: forget10 | Method: SBGradDiffMatchedJoint

set -a
source .env
set +a

model="Llama-3.2-1B-Instruct"
forget_split="forget10"
retain_split="retain90"
holdout_split="holdout10"

task_name="SB_TOFU/${model}/forget10/SBGradDiffMatchedJoint"
model_output="saves/unlearn/${task_name}"

winrate_baseline="saves/eval/SB_TOFU/${model}/baselines/${retain_split}_${forget_split}/paraphrase_evals/repetitiveness/model.jsonl"

echo ""
echo "============================================"
echo " SBGradDiffMatchedJoint — Paraphrase Judging (forget10)"
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
    +judge_only=true \
    paths.output_dir=$(pwd)/${model_output}/paraphrase_evals
