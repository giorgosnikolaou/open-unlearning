#!/bin/bash
# Evaluate baseline models (full + retain) on TOFU + Paraphrase for all splits
# Model: Llama-3.1-8B-Instruct

set -a
source .env
set +a

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

model="Llama-3.1-8B-Instruct"
GPU=0

# full model on all splits; each retain model on its corresponding split only
eval_pairs=(
    "full forget01"
    "full forget05"
    "full forget10"
    "retain99 forget01"
    "retain95 forget05"
    "retain90 forget10"
)

for pair in "${eval_pairs[@]}"; do
    read -r model_type forget_split <<< "$pair"
    model_path="open-unlearning/tofu_${model}_${model_type}"
    task_name="SB_TOFU/${model}/baselines/${model_type}_${forget_split}"

        echo ""
        echo "============================================"
        echo " ${model_type} | ${forget_split} — TOFU Eval"
        echo "============================================"

        HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=$GPU \
            python src/eval.py \
            experiment=eval/tofu/default.yaml \
            'eval=[tofu]' \
            model=$model \
            task_name=$task_name \
            model.model_args.pretrained_model_name_or_path=$model_path \
            +model.model_args.token=$HF_TOKEN \
            +model.tokenizer_args.token=$HF_TOKEN \
            ++model.model_args.device_map='auto' \
            forget_split=$forget_split \
            paths.output_dir=$(pwd)/saves/eval/${task_name}/tofu_evals

        # echo ""
        # echo "============================================"
        # echo " ${model_type} | ${forget_split} — Paraphrase Eval"
        # echo "============================================"

        # HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=$GPU \
        #     python src/eval.py \
        #     experiment=eval/paraphrase/default.yaml \
        #     'eval=[paraphrase_tofu]' \
        #     model=$model \
        #     task_name=$task_name \
        #     model.model_args.pretrained_model_name_or_path=$model_path \
        #     +model.model_args.token=$HF_TOKEN \
        #     +model.tokenizer_args.token=$HF_TOKEN \
        #     ++model.model_args.device_map='auto' \
        #     '~eval.paraphrase.metrics.winrate' \
        #     eval.paraphrase.forget_split=$forget_split \
        #     paths.output_dir=$(pwd)/saves/eval/${task_name}/paraphrase_evals
done
