#!/bin/bash
# SelfBalancing GradDiff + LearnedScorer — RWKU Train & Eval (Prompted variant)
# Uses PromptedCompletionDataset: each sample gets a chat template with
# per-subject system prompt suffix "Give all information about {subject}."

set -a
source .env
set +a

source /tmlscratch/nikolaou/open-unlearning/envs/.open-unlearning/bin/activate

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

declare -A HIDDEN_DIM_MAP=(
    ["Llama-2-7b-chat-hf"]=4096
    ["Llama-3.2-1B-Instruct"]=2048
    ["Llama-3.2-3B-Instruct"]=3072
    ["Llama-3.1-8B-Instruct"]=4096
    ["Llama-3-8B-Instruct"]=4096        # RWKU default
    ["Phi-3.5-mini-instruct"]=3072       # RWKU alternate
)

model="Llama-3.2-1B-Instruct"
model_path="meta-llama/Llama-3.2-1B-Instruct"

trainer="SBGradDiffLearned"
experiment="unlearn/rwku/neighbor_retain.yaml"
scorer_in_dim=${HIDDEN_DIM_MAP[$model]}

per_device_train_batch_size=4
gradient_accumulation_steps=2

lr=1e-5
gamma=5.0
alpha=0.5
gamma=0.0
alpha=0.001

lambda_entropy=1.0
lambda_population=10.0
budget=0.4
lambda_l2=1.0

update_every_n_steps=5

scorer_lr=0.05
scorer_grad_clip=1.0
normalize_hidden=False

num_train_epochs=1

scorer_pretrain_epochs=0
scorer_pretrain_lr=0.05
scorer_freeze_after=False

# scorer_pretrain_epochs=1
# scorer_pretrain_lr=0.05
# scorer_freeze_after=True

max_length=512

task_name="rwku_${model}_${trainer}_lr${lr}_gamma${gamma}_alpha${alpha}_bs${per_device_train_batch_size}_accum${gradient_accumulation_steps}"
params="ent${lambda_entropy}_pop${lambda_population}_budget${budget}_l2${lambda_l2}_up${update_every_n_steps}"


prefix="NiceRWKU/Llama3.2-1B/${max_length}/beta0.5_${num_train_epochs}_${scorer_lr}_${scorer_grad_clip}_${normalize_hidden}_"
prefix="NiceRWKU/Llama3.2-1B/${max_length}/beta5_${num_train_epochs}_${scorer_lr}_${scorer_grad_clip}_${normalize_hidden}_"


complete_name="${prefix}${params}_${task_name}"

###################
# 1. UNLEARNING
###################
echo "==========================================="
echo " LearnedScorer Prompted — ${complete_name}"
echo "==========================================="

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 \
    python src/train.py --config-name=unlearn.yaml \
    experiment=$experiment \
    trainer=$trainer \
    task_name=${complete_name} \
    model=$model \
    model.model_args.pretrained_model_name_or_path=${model_path} \
    +model.model_args.token=$HF_TOKEN \
    +model.tokenizer_args.token=$HF_TOKEN \
    +model.model_args.device_map='auto' \
    trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
    trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
    trainer.args.eval_strategy=no \
    trainer.args.eval_on_start=False \
    trainer.args.learning_rate=$lr \
    trainer.args.num_train_epochs=$num_train_epochs \
    trainer.method_args.gamma=$gamma \
    trainer.method_args.alpha=$alpha \
    trainer.method_args.scorer.cfg.input_dimension=$scorer_in_dim \
    trainer.method_args.scorer_trainer.optim_cfg.update_every_n_steps=$update_every_n_steps \
    trainer.method_args.scorer_trainer.optim_cfg.lr=$scorer_lr \
    trainer.method_args.scorer_trainer.optim_cfg.grad_clip=$scorer_grad_clip \
    +trainer.method_args.scorer.cfg.normalize_hidden=${normalize_hidden} \
    +trainer.method_args.scorer_trainer.optim_cfg.scheduler=linear \
    trainer.method_args.scorer_trainer.lambda_entropy=$lambda_entropy \
    trainer.method_args.scorer_trainer.lambda_population=$lambda_population \
    trainer.method_args.scorer_trainer.budget=$budget \
    trainer.method_args.scorer_trainer.lambda_l2=$lambda_l2 \
    trainer.method_args.scorer_pretrain.epochs=$scorer_pretrain_epochs \
    trainer.method_args.scorer_pretrain.lr=$scorer_pretrain_lr \
    trainer.method_args.scorer_pretrain.freeze_after=$scorer_freeze_after \
    data.forget.RWKU_forget_passage.args.max_length=${max_length} \
    data.retain.RWKU_retain_passage.args.max_length=${max_length}

########################################
# 2. PARAPHRASE EVALUATION
########################################
# echo ""
# echo "=================================================="
# echo " Paraphrase Eval — ${complete_name}"
# echo "=================================================="

# HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 \
#     python src/eval.py \
#     experiment=eval/paraphrase/default.yaml \
#     'eval=[paraphrase_rwku]' \
#     model=$model \
#     task_name=${complete_name} \
#     model.model_args.pretrained_model_name_or_path=$(pwd)/saves/unlearn/${complete_name} \
#     +model.model_args.token=$HF_TOKEN \
#     +model.tokenizer_args.token=$HF_TOKEN \
#     ++model.model_args.device_map='auto' \
#     eval.paraphrase.metrics.winrate.baseline_path=/tmlscratch/nikolaou/open-unlearning/saves/eval/RWKU_baseline_Llama-3.2-1B-Instruct/paraphrase_evals/repetitiveness/model.jsonl \
#     paths.output_dir=$(pwd)/saves/unlearn/${complete_name}/paraphrase_evals
