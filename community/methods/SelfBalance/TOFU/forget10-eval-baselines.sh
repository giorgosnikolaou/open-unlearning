#!/bin/bash
# Evaluate baseline models (full, retain90) on TOFU + Paraphrase benchmarks

set -a
source .env
set +a

source /tmlscratch/nikolaou/open-unlearning/envs/.open-unlearning/bin/activate

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

# Configuration
model="Llama-3.2-1B-Instruct"
# model="Llama-2-7b-chat-hf"

# for model_type in "full" "retain90"; do
for model_type in "retain90"; do
    model_path="open-unlearning/tofu_${model}_${model_type}"
    task_name="ES_${model}_${model_type}"

    ########################################
    # 1. TOFU EVALUATION
    ########################################
    # echo ""
    # echo "============================================"
    # echo " TOFU Evaluation: ${model_type} model"
    # echo "============================================"

    # HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 \
    #     python src/eval.py \
    #     experiment=eval/tofu/default.yaml \
    #     'eval=[tofu]' \
    #     model=$model \
    #     task_name=$task_name \
    #     model.model_args.pretrained_model_name_or_path=$model_path \
    #     +model.model_args.token=$HF_TOKEN \
    #     +model.tokenizer_args.token=$HF_TOKEN \
    #     ++model.model_args.device_map='auto' \
    #     paths.output_dir=$(pwd)/saves/eval/${task_name}/tofu_evals

    ########################################
    # 2. PARAPHRASE EVALUATION (no winrate)
    ########################################
    echo ""
    echo "============================================"
    echo " Paraphrase Evaluation: ${model_type} model"
    echo "============================================"

    HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 \
        python src/eval.py \
        experiment=eval/paraphrase/default.yaml \
        'eval=[paraphrase_tofu]' \
        model=$model \
        task_name=$task_name \
        model.model_args.pretrained_model_name_or_path=$model_path \
        +model.model_args.token=$HF_TOKEN \
        +model.tokenizer_args.token=$HF_TOKEN \
        ++model.model_args.device_map='auto' \
        '~eval.paraphrase.metrics.winrate' \
        paths.output_dir=$(pwd)/saves/eval/${task_name}/paraphrase_evals
done
