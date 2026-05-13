#!/bin/bash
# Evaluate pretrained Phi-3-mini-4k-instruct baseline on RWKU benchmark (RWKU eval only).

set -a
source .env
set +a

source /tmlscratch/nikolaou/open-unlearning/envs/.open-unlearning/bin/activate

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

# Configuration
model="Phi-3-mini-4k-instruct"
model_path="microsoft/Phi-3-mini-4k-instruct"
task_name="RWKU_baseline_${model}"

########################################
# RWKU EVALUATION
########################################
echo ""
echo "============================================"
echo " RWKU Evaluation: baseline ${model}"
echo "============================================"

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 \
    python src/eval.py \
    experiment=eval/rwku/default.yaml \
    'eval=[rwku]' \
    model=$model \
    task_name=$task_name \
    model.model_args.pretrained_model_name_or_path=$model_path \
    model.model_args.torch_dtype=float16 \
    +model.model_args.token=$HF_TOKEN \
    +model.tokenizer_args.token=$HF_TOKEN \
    ++model.model_args.device_map='auto' \
    paths.output_dir=$(pwd)/saves/eval/${task_name}/rwku_evals

########################################
# RWKU UTILITY EVALUATION
########################################
echo ""
echo "============================================"
echo " RWKU Utility Evaluation: baseline ${model}"
echo "============================================"

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 \
    python src/eval.py \
    experiment=eval/rwku/default.yaml \
    'eval=[rwku_utility]' \
    model=$model \
    task_name=$task_name \
    model.model_args.pretrained_model_name_or_path=$model_path \
    model.model_args.torch_dtype=float16 \
    +model.model_args.token=$HF_TOKEN \
    +model.tokenizer_args.token=$HF_TOKEN \
    ++model.model_args.device_map='auto' \
    paths.output_dir=$(pwd)/saves/eval/${task_name}/rwku_utility_evals
