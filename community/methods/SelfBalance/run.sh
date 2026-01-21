#!/bin/bash

set -a
source .env
set +a

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

trainer="SelfBalancing"
baseline_trainer="GradDiff"
model="Llama-3.2-1B-Instruct"
model="Llama-2-7b-chat-hf"

model_path="open-unlearning/tofu_${model}_full"
forget_split="forget10"
retain_split="retain90"

experiment="unlearn/tofu/default.yaml"

lr=1e-5
# gamma=2.0
# alpha=2.0
gamma=1.0
alpha=1.0
gamma=2.0
alpha=3.0

per_device_train_batch_size=16
gradient_accumulation_steps=2

task_name="tofu_${model}_${forget_split}_${trainer}_lr${lr}_gamma${gamma}_alpha${alpha}"


####################################
# Baseline, all weights equal to 1 #
####################################
# CUDA_VISIBLE_DEVICES=0 \
#     python src/train.py --config-name=unlearn.yaml \
#     experiment=$experiment \
#     trainer=$baseline_trainer \
#     task_name=BASELINE_NLL_${task_name} \
#     model=$model \
#     forget_split=$forget_split \
#     retain_split=$retain_split \
#     model.model_args.pretrained_model_name_or_path=$model_path \
#     +model.model_args.token=$HF_TOKEN \
#     retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
#     trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
#     trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
#     trainer.args.eval_strategy=no \
#     trainer.args.eval_on_start=False \
#     trainer.args.learning_rate=$lr \
#     trainer.method_args.gamma=$gamma \
#     trainer.method_args.alpha=$alpha

# Eval
# CUDA_VISIBLE_DEVICES=0 \
#     python src/eval.py \
#     experiment=eval/tofu/default.yaml \
#     forget_split=$forget_split \
#     model=$model \
#     task_name=baseline_${task_name} \
#     model.model_args.pretrained_model_name_or_path=saves/unlearn/baseline_${task_name} \
#     paths.output_dir=saves/unlearn/baseline_${task_name}/evals \
#     retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json


##################
# Self Balancing #
##################
alpha=3.0
gamma=2.0
alpha=6.0
gamma=4.0
task_name="tofu_${model}_${forget_split}_${trainer}_lr${lr}_gamma${gamma}_alpha${alpha}"

CUDA_VISIBLE_DEVICES=0 \
    python src/train.py --config-name=unlearn.yaml \
    experiment=$experiment \
    trainer=$trainer \
    task_name=SOFTMAX_SCH_${task_name} \
    model=$model \
    forget_split=$forget_split \
    retain_split=$retain_split \
    model.model_args.pretrained_model_name_or_path=$model_path \
    +model.model_args.token=$HF_TOKEN \
    +model.tokenizer_args.token=$HF_TOKEN \
    +model.model_args.device_map='auto' \
    retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
    trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
    trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
    trainer.args.eval_strategy=no \
    trainer.args.eval_on_start=False \
    trainer.args.learning_rate=$lr \
    trainer.method_args.gamma=$gamma \
    trainer.method_args.alpha=$alpha \
    trainer.method_args.scorer_optim_cfg.update_every_n_steps=10 \
    +trainer.method_args.scorer_optim_cfg.scheduler=linear

# Eval
# CUDA_VISIBLE_DEVICES=0 \
#     python src/eval.py \
#     experiment=eval/tofu/default.yaml \
#     forget_split=$forget_split \
#     model=$model \
#     task_name=$task_name \
#     model.model_args.pretrained_model_name_or_path=saves/unlearn/$task_name \
#     paths.output_dir=saves/unlearn/$task_name/evals \
#     retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json