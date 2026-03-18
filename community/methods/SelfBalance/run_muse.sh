#!/bin/bash

set -a
source .env
set +a

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

trainer="SelfBalancing"
baseline_trainer="GradDiff"
model="Llama-2-7b-hf"


# data_splits=("News" "Books")
data_split="News"
model_path="muse-bench/MUSE-${data_split}_target"

experiment="unlearn/muse/default.yaml"

lr=1e-5
gamma=1.0
alpha=1.0

per_device_train_batch_size=2
gradient_accumulation_steps=4

task_name="MUSE_${model}_${forget_split}_${trainer}_lr${lr}_gamma${gamma}_alpha${alpha}"


####################################
# Baseline, all weights equal to 1 #
####################################
CUDA_VISIBLE_DEVICES=0 \
    python src/train.py --config-name=unlearn.yaml \
    experiment=$experiment \
    trainer=$baseline_trainer \
    task_name=BASELINE_NLL_${task_name} \
    model=$model \
    data_split=$data_split \
    model.model_args.pretrained_model_name_or_path=$model_path \
    +model.model_args.token=$HF_TOKEN \
    +model.model_args.device_map='auto' \
    retain_logs_path=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json \
    trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
    trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
    trainer.args.eval_strategy=no \
    trainer.args.eval_on_start=False \
    trainer.args.learning_rate=$lr \
    trainer.method_args.gamma=$gamma \
    trainer.method_args.alpha=$alpha


##################
# Self Balancing #
##################
alpha=3.0
gamma=2.0
task_name="MUSE_${model}_${forget_split}_${trainer}_lr${lr}_gamma${gamma}_alpha${alpha}"

CUDA_VISIBLE_DEVICES=0 \
    python src/train.py --config-name=unlearn.yaml \
    experiment=$experiment \
    trainer=$trainer \
    task_name=TEST_${task_name} \
    model=$model \
    data_split=$data_split \
    model.model_args.pretrained_model_name_or_path=$model_path \
    +model.model_args.token=$HF_TOKEN \
    +model.tokenizer_args.token=$HF_TOKEN \
    +model.model_args.device_map='auto' \
    retain_logs_path=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json \
    trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
    trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
    trainer.args.eval_strategy=no \
    trainer.args.eval_on_start=False \
    trainer.args.learning_rate=$lr \
    trainer.method_args.gamma=$gamma \
    trainer.method_args.alpha=$alpha \
    trainer.method_args.scorer_optim_cfg.update_every_n_steps=10 \
    +trainer.method_args.scorer_optim_cfg.scheduler=linear