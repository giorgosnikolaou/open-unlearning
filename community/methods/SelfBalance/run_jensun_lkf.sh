#!/bin/bash
# JensUn Unlearning + LKF Paraphrase Evaluation Script

set -a
source .env
set +a

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

# Configuration
trainer="JensUn"
model="Llama-3.2-3B-Instruct"

# Model and data paths
model_path="meta-llama/${model}"

# Experiment configuration
experiment="unlearn/lkf/default.yaml"

# Hyperparameters (matching original JensUn repo)
lr=1e-5
gamma=1.0
alpha=1.0

lr=8e-6
gamma=0.5
alpha=0.5

per_device_train_batch_size=8
gradient_accumulation_steps=4

# Task naming
task_name="lkf_${model}_${trainer}_lr${lr}_gamma${gamma}_alpha${alpha}"
prefix="JensUn_"

echo "======================================="
echo " Running JensUn Unlearning on LKF"
echo " Task: ${prefix}${task_name}"
echo "======================================="

###################
# 1. UNLEARNING   #
###################
# "model.template_args.system_prompt='You are a helpful assistant. Keep your answers brief and concise, at most 15 words.'" \
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 \
    python src/train.py --config-name=unlearn.yaml \
    experiment=$experiment \
    trainer=$trainer \
    task_name=${prefix}${task_name} \
    model=$model \
    model.model_args.pretrained_model_name_or_path=$model_path \
    +model.model_args.token=$HF_TOKEN \
    +model.tokenizer_args.token=$HF_TOKEN \
    +model.model_args.device_map='auto' \
    trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
    trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
    trainer.args.eval_strategy=no \
    trainer.args.eval_on_start=False \
    trainer.args.learning_rate=$lr \
    trainer.method_args.gamma=$gamma \
    trainer.method_args.alpha=$alpha \
    'data/datasets@data.forget=LKF_original_para_forget' \
    'data/datasets@data.retain=LKF_original_para_retain' \
    '~data.forget.LKF_para_forget' \
    '~data.retain.LKF_retain'
    # data.forget.LKF_para_forget.args.num_train_paraphrases=5
########################################
# 2. PARAPHRASE EVALUATION (PRETRAINED)
########################################
echo ""
echo "==================================================="
echo " Running Paraphrase Evaluation on Pretrained Model "
echo "==================================================="

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 \
    python src/eval.py \
    experiment=eval/paraphrase/default.yaml \
    'eval=[paraphrase_lkf]' \
    model=$model \
    task_name=pretrained_${model} \
    model.model_args.pretrained_model_name_or_path=$model_path \
    +model.model_args.token=$HF_TOKEN \
    +model.tokenizer_args.token=$HF_TOKEN \
    ++model.model_args.device_map='auto' \
    eval.paraphrase.metrics.repetitiveness.system_prompt="You are a helpful assistant." \
    '~eval.paraphrase.metrics.winrate' \
    paths.output_dir=saves/eval/lkf_para_${model}

########################################
# 3. PARAPHRASE EVALUATION (UNLEARNED)
########################################
echo ""
echo "=================================================="
echo " Running Paraphrase Evaluation on Unlearned Model "
echo "=================================================="

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 \
    python src/eval.py \
    experiment=eval/paraphrase/default.yaml \
    'eval=[paraphrase_lkf]' \
    model=$model \
    task_name=${prefix}${task_name} \
    model.model_args.pretrained_model_name_or_path=$(pwd)/saves/unlearn/${prefix}${task_name} \
    +model.model_args.token=$HF_TOKEN \
    +model.tokenizer_args.token=$HF_TOKEN \
    ++model.model_args.device_map='auto' \
    +eval.paraphrase.metrics.winrate.baseline_path=saves/eval/lkf_para_${model}/repetitiveness/model.jsonl \
    paths.output_dir=$(pwd)/saves/unlearn/${prefix}${task_name}/evals 
