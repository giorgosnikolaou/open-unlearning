#!/bin/bash
# SatImp — RWKU Train & Eval

set -a
source .env
set +a

source /tmlscratch/nikolaou/open-unlearning/envs/.open-unlearning/bin/activate

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

model="Llama-3.1-8B-Instruct"

trainer="SatImp"
experiment="unlearn/rwku/default.yaml"

per_device_train_batch_size=2
gradient_accumulation_steps=8

lr=1e-5
beta1=5.0
beta2=1.0
alpha=1.0
gamma=0.1
num_train_epochs=5

task_name="rwku_${model}_${trainer}_lr${lr}_beta1${beta1}_beta2${beta2}_alpha${alpha}_gamma${gamma}"

###################
# 1. UNLEARNING
###################
echo "==========================================="
echo " SatImp — ${task_name}"
echo "==========================================="

HYDRA_FULL_ERROR=1 \
    accelerate launch --config_file configs/accelerate/default_config.yaml \
    src/train.py --config-name=unlearn.yaml \
    experiment=$experiment \
    trainer=$trainer \
    task_name=${task_name} \
    model=$model \
    model.model_args.pretrained_model_name_or_path=meta-llama/Meta-Llama-3-8B-Instruct \
    +model.model_args.token=$HF_TOKEN \
    +model.tokenizer_args.token=$HF_TOKEN \
    trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
    trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
    trainer.args.num_train_epochs=$num_train_epochs \
    trainer.args.eval_strategy=no \
    trainer.args.eval_on_start=False \
    trainer.args.learning_rate=$lr \
    trainer.method_args.beta1=$beta1 \
    trainer.method_args.beta2=$beta2 \
    trainer.method_args.alpha=$alpha \
    trainer.method_args.gamma=$gamma

########################################
# 2. RWKU EVALUATION (UNLEARNED)
########################################
echo ""
echo "=================================================="
echo " RWKU Eval — ${task_name}"
echo "=================================================="

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 \
    python src/eval.py \
    experiment=eval/rwku/default.yaml \
    'eval=[rwku]' \
    model=$model \
    task_name=${task_name} \
    model.model_args.pretrained_model_name_or_path=$(pwd)/saves/unlearn/${task_name} \
    +model.model_args.token=$HF_TOKEN \
    +model.tokenizer_args.token=$HF_TOKEN \
    ++model.model_args.device_map='auto' \
    paths.output_dir=$(pwd)/saves/unlearn/${task_name}/rwku_evals

########################################
# 3. PARAPHRASE EVALUATION
########################################
echo ""
echo "=================================================="
echo " Paraphrase Eval — ${task_name}"
echo "=================================================="

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 \
    python src/eval.py \
    experiment=eval/paraphrase/default.yaml \
    'eval=[paraphrase_rwku]' \
    model=$model \
    task_name=${task_name} \
    model.model_args.pretrained_model_name_or_path=$(pwd)/saves/unlearn/${task_name} \
    +model.model_args.token=$HF_TOKEN \
    +model.tokenizer_args.token=$HF_TOKEN \
    ++model.model_args.device_map='auto' \
    '~eval.paraphrase.metrics.winrate' \
    paths.output_dir=$(pwd)/saves/unlearn/${task_name}/evals
