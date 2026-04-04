#!/bin/bash
# JensUn — TOFU Paraphrase Train & Eval

set -a
source .env
set +a

source /tmlscratch/nikolaou/open-unlearning/envs/.open-unlearning/bin/activate

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

model="Llama-3.2-1B-Instruct"
# model="Llama-2-7b-chat-hf"
trainer="JensUn"
experiment="unlearn/tofu/default.yaml"

per_device_train_batch_size=16
gradient_accumulation_steps=2

lr=4e-5
gamma=0.5
alpha=0.5
retain_loss_type="JensUn"

task_name="tofu_${model}_${trainer}_lr${lr}_gamma${gamma}_alpha${alpha}"
prefix="JensUn/"

complete_name="${prefix}${task_name}"

###################
# 1. UNLEARNING
###################
echo "==========================================="
echo " JensUn — ${complete_name}"
echo "==========================================="

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 \
    python src/train.py --config-name=unlearn.yaml \
    experiment=$experiment \
    trainer=$trainer \
    task_name=${complete_name} \
    model=$model \
    model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_${model}_full \
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
    trainer.method_args.retain_loss_type=$retain_loss_type

########################################
# 2. TOFU EVALUATION (UNLEARNED)
########################################
# echo ""
# echo "=================================================="
# echo " TOFU Eval — ${complete_name}"
# echo "=================================================="

# HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 \
#     python src/eval.py \
#     experiment=eval/tofu/default.yaml \
#     'eval=[tofu]' \
#     model=$model \
#     task_name=${complete_name} \
#     model.model_args.pretrained_model_name_or_path=$(pwd)/saves/unlearn/${complete_name} \
#     +model.model_args.token=$HF_TOKEN \
#     +model.tokenizer_args.token=$HF_TOKEN \
#     ++model.model_args.device_map='auto' \
#     paths.output_dir=$(pwd)/saves/unlearn/${complete_name}/tofu_evals


########################################
# 3. PARAPHRASE EVALUATION
########################################
echo ""
echo "=================================================="
echo " Paraphrase Eval — ${complete_name}"
echo "=================================================="

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 \
    python src/eval.py \
    experiment=eval/paraphrase/default.yaml \
    'eval=[paraphrase_tofu]' \
    model=$model \
    task_name=${complete_name} \
    model.model_args.pretrained_model_name_or_path=$(pwd)/saves/unlearn/${complete_name} \
    +model.model_args.token=$HF_TOKEN \
    +model.tokenizer_args.token=$HF_TOKEN \
    ++model.model_args.device_map='auto' \
    +eval.paraphrase.metrics.winrate.baseline_path=saves/eval/ES_Llama-3.2-1B-Instruct_retain90/paraphrase_evals/repetitiveness/model.jsonl \
    paths.output_dir=$(pwd)/saves/unlearn/${complete_name}/evals/paraphrase_evals
