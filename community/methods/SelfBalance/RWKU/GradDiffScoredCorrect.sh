#!/bin/bash
# Evaluate GradDiffScoredCorrect checkpoint on RWKU benchmark.
# Usage: bash GradDiffScoredCorrect.sh [trial_name]  (default: trial_3_lr4_44e-06)

set -a
source .env
set +a

source /tmlscratch/nikolaou/open-unlearning/envs/.open-unlearning/bin/activate

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

# Configuration
model="Phi-3-mini-4k-instruct"
trial_base="/tmlscratch/nikolaou/RWKU/LLaMA-Factory/hyperparam/rwku_bayesian_lr/GradDiffScoredCorrect"
trial="${1:-trial_3_lr4_44e-06}"
model_path="${trial_base}/${trial}/"
task_name="RWKU_bayesian_GradDiffScoredCorrect_${trial}"

########################################
# RWKU EVALUATION
########################################
echo ""
echo "============================================"
echo " RWKU Evaluation: GradDiffScoredCorrect"
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
    '~eval.rwku.metrics.fluency' \
    paths.output_dir=$(pwd)/saves/unlearn/RWKU/Optimal/GradDiffScoredCorrect/rwku_evals

########################################
# RWKU UTILITY EVALUATION
########################################
# echo ""
# echo "============================================"
# echo " RWKU Utility Evaluation: GradDiffScoredCorrect"
# echo "============================================"
#
# HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 \
#     python src/eval.py \
#     experiment=eval/rwku/default.yaml \
#     'eval=[rwku_utility]' \
#     model=$model \
#     task_name=$task_name \
#     model.model_args.pretrained_model_name_or_path=$model_path \
#     model.model_args.torch_dtype=float16 \
#     +model.model_args.token=$HF_TOKEN \
#     +model.tokenizer_args.token=$HF_TOKEN \
#     ++model.model_args.device_map='auto' \
#     '~eval.rwku' \
#     paths.output_dir=$(pwd)/saves/unlearn/RWKU/Optimal/GradDiffScoredCorrect/rwku_utility_evals
