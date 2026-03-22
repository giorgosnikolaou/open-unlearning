#!/bin/bash
# Evaluate MUSE News baseline models (target, retrain) on MUSE benchmark

set -a
source .env
set +a

source /tmlscratch/nikolaou/open-unlearning/envs/.open-unlearning/bin/activate

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

# Configuration
model="Llama-2-7b-hf"
data_split="News"

task_name="muse_${data_split}_eval_${model}"
retrain_task="${task_name}_retrain"
target_task="${task_name}_target"

# ########################################
# # 1. MUSE EVALUATION — RETRAIN MODEL
# ########################################
# echo ""
# echo "============================================"
# echo " MUSE ${data_split} Evaluation: retrain model"
# echo "============================================"

# HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 \
#     python src/eval.py \
#     experiment=eval/muse/default.yaml \
#     'eval=[muse]' \
#     model=$model \
#     data_split=$data_split \
#     task_name=$retrain_task \
#     model.model_args.pretrained_model_name_or_path=muse-bench/MUSE-${data_split}_retrain \
#     +model.model_args.token=$HF_TOKEN \
#     +model.tokenizer_args.token=$HF_TOKEN \
#     ++model.model_args.device_map='auto' \
#     paths.output_dir=$(pwd)/saves/eval/${retrain_task}

########################################
# 2. PARAPHRASE EVALUATION — RETRAIN
########################################
echo ""
echo "============================================"
echo " Paraphrase ${data_split} Evaluation: retrain model"
echo "============================================"

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 \
    python src/eval.py \
    experiment=eval/paraphrase/default.yaml \
    'eval=[paraphrase_muse]' \
    model=$model \
    task_name=$retrain_task \
    model.model_args.pretrained_model_name_or_path=muse-bench/MUSE-${data_split}_retrain \
    +model.model_args.token=$HF_TOKEN \
    +model.tokenizer_args.token=$HF_TOKEN \
    ++model.model_args.device_map='auto' \
    '~eval.paraphrase.metrics.winrate' \
    paths.output_dir=$(pwd)/saves/eval/${retrain_task}/paraphrase_evals

# ########################################
# # 3. MUSE EVALUATION — TARGET MODEL
# ########################################
# echo ""
# echo "============================================"
# echo " MUSE ${data_split} Evaluation: target model"
# echo "============================================"

# HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 \
#     python src/eval.py \
#     experiment=eval/muse/default.yaml \
#     'eval=[muse]' \
#     model=$model \
#     data_split=$data_split \
#     task_name=$target_task \
#     model.model_args.pretrained_model_name_or_path=muse-bench/MUSE-${data_split}_target \
#     +model.model_args.token=$HF_TOKEN \
#     +model.tokenizer_args.token=$HF_TOKEN \
#     ++model.model_args.device_map='auto' \
#     retain_logs_path=$(pwd)/saves/eval/${retrain_task} \
#     paths.output_dir=$(pwd)/saves/eval/${target_task}

# ########################################
# # 4. PARAPHRASE EVALUATION — TARGET
# ########################################
# echo ""
# echo "============================================"
# echo " Paraphrase ${data_split} Evaluation: target model"
# echo "============================================"

# HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 \
#     python src/eval.py \
#     experiment=eval/paraphrase/default.yaml \
#     'eval=[paraphrase_muse]' \
#     model=$model \
#     task_name=$target_task \
#     model.model_args.pretrained_model_name_or_path=muse-bench/MUSE-${data_split}_target \
#     +model.model_args.token=$HF_TOKEN \
#     +model.tokenizer_args.token=$HF_TOKEN \
#     ++model.model_args.device_map='auto' \
#     '~eval.paraphrase.metrics.winrate' \
#     paths.output_dir=$(pwd)/saves/eval/${target_task}/paraphrase_evals
