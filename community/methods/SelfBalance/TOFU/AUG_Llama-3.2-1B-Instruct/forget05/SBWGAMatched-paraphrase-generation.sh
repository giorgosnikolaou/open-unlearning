#!/bin/bash
# Paraphrase Generation — GPU phase
# Model: Llama-3.2-1B-Instruct | Split: forget05 | Method: SBWGAMatched

set -a
source .env
set +a

model="Llama-3.2-1B-Instruct"
forget_split="forget05"
retain_split="retain95"
holdout_split="holdout05"
GPU=${GPU:-0}

task_name="AUG_TOFU/${model}/forget05/SBWGAMatched"
model_output="saves/unlearn/${task_name}"

echo ""
echo "============================================"
echo " SBWGAMatched — Paraphrase Generation (forget05)"
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
