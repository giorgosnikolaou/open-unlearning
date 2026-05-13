#!/bin/bash
# GradDiff — RWKU Train & Eval

set -a
source .env
set +a

source /tmlscratch/nikolaou/open-unlearning/envs/.open-unlearning/bin/activate

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")


model="Llama-3-8B-Instruct"
model_path="meta-llama/Meta-Llama-3-8B-Instruct"

model="Qwen2.5-7B-Instruct"
model_path="Qwen/Qwen2.5-7B-Instruct"

model="Llama-3.1-8B-Instruct"
model_path="meta-llama/Llama-3.1-8B-Instruct"

model="Llama-3.2-1B-Instruct"
model_path="meta-llama/Llama-3.2-1B-Instruct"

model="Phi-3.5-mini-instruct"
model_path="microsoft/Phi-3.5-mini-instruct"

trainer="GradDiff"
experiment="unlearn/rwku/default.yaml"
experiment="unlearn/rwku/positive.yaml"
# experiment="unlearn/rwku/positive_generated.yaml"

per_device_train_batch_size=4
gradient_accumulation_steps=3

lr=1e-6
gamma=1.0
alpha=1.0

lr=6e-7
gamma=0.5
alpha=0.5

# lr=1e-5
# gamma=0.5
# alpha=0.5

num_train_epochs=5
max_length=512

# task_name="RWKU/GradDiff/${model}_${trainer}_lr${lr}_gamma${gamma}_alpha${alpha}_epochs${num_train_epochs}"
task_name="RWKU/GradDiff/${max_length}/${model}_${trainer}_lr${lr}_gamma${gamma}_alpha${alpha}_epochs${num_train_epochs}"
task_name="RWKU/GradDiff/${max_length}/wikipedia_${model}_${trainer}_lr${lr}_gamma${gamma}_alpha${alpha}_epochs${num_train_epochs}"
task_name="RWKU/GradDiff/${max_length}/positive_${model}_${trainer}_lr${lr}_gamma${gamma}_alpha${alpha}_epochs${num_train_epochs}"
task_name="RWKU/GradDiff/${max_length}/fp16_positive_${model}_${trainer}_lr${lr}_gamma${gamma}_alpha${alpha}_epochs${num_train_epochs}"

###################
# 1. UNLEARNING
###################
echo "==========================================="
echo " GradDiff — ${task_name}"
echo "==========================================="

HYDRA_FULL_ERROR=1 \
    accelerate launch --config_file configs/accelerate/default_config.yaml \
    src/train.py --config-name=unlearn.yaml \
    experiment=$experiment \
    trainer=$trainer \
    task_name=${task_name} \
    model=$model \
    model.model_args.pretrained_model_name_or_path=${model_path} \
    model.model_args.torch_dtype=float16 \
    +model.model_args.token=$HF_TOKEN \
    +model.tokenizer_args.token=$HF_TOKEN \
    trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
    trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
    trainer.args.num_train_epochs=$num_train_epochs \
    trainer.args.eval_strategy=no \
    trainer.args.eval_on_start=False \
    trainer.args.gradient_checkpointing=True \
    trainer.args.learning_rate=$lr \
    trainer.method_args.gamma=$gamma \
    trainer.method_args.alpha=$alpha \
    data.forget.RWKU_train_positive.args.max_length=${max_length} \
    data.retain.RWKU_train_positive.args.max_length=${max_length}
    # data.forget.RWKU_train_original.args.max_length=${max_length} \
    # data.retain.RWKU_train_original.args.max_length=${max_length}
    # data.forget.RWKU_train_positive.args.max_length=${max_length} \
    # data.retain.RWKU_neighbor_positive.args.max_length=${max_length}

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
    model.model_args.torch_dtype=float16 \
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
    model.model_args.torch_dtype=float16 \
    +model.model_args.token=$HF_TOKEN \
    +model.tokenizer_args.token=$HF_TOKEN \
    ++model.model_args.device_map='auto' \
    '~eval.paraphrase.metrics.winrate' \
    paths.output_dir=$(pwd)/saves/unlearn/${task_name}/evals
