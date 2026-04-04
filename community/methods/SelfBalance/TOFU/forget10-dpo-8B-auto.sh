#!/bin/bash
# DPO — TOFU forget10
# Model: Llama-3.1-8B-Instruct (device_map='auto', no accelerate)

set -a
source .env
set +a

source /tmlscratch/nikolaou/open-unlearning/envs/.open-unlearning/bin/activate

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

model="Llama-3.1-8B-Instruct"
trainer="DPO"
experiment="unlearn/tofu/idk"
GPU=${GPU:-0,1}

per_device_train_batch_size=8
gradient_accumulation_steps=4
num_epochs=10

# ── HP search summary (override with SUMMARY_JSON env var) ──
SUMMARY_JSON="${SUMMARY_JSON:-hyperparam/tofu_forget10/bayesian_summary.json}"

get_param() {
    python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d[sys.argv[2]]['best_params'][sys.argv[3]])" \
        "$SUMMARY_JSON" "$1" "$2"
}

# ── Hyperparameters (default from hpsearch, override via env vars) ──
# lr=1e-5; gamma=5.0; alpha=0.5; beta=1.0  # manual override
lr=${lr:-$(get_param DPO lr)}
gamma=${gamma:-$(get_param DPO gamma)}
alpha=${alpha:-$(get_param DPO alpha)}
beta=${beta:-$(get_param DPO beta)}

task_name="DPO_8B_auto/1gpu_bs8_acc4_lr${lr}_gamma${gamma}_alpha${alpha}_beta${beta}"

###################
# 1. UNLEARNING
###################
echo "==========================================="
echo " DPO — ${task_name}"
echo "==========================================="

# HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=$GPU \
HYDRA_FULL_ERROR=1 \
    python src/train.py --config-name=unlearn.yaml \
    experiment=$experiment \
    trainer=$trainer \
    model=$model \
    task_name=${task_name} \
    model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_${model}_full \
    +model.model_args.token=$HF_TOKEN \
    +model.tokenizer_args.token=$HF_TOKEN \
    ++model.model_args.device_map='auto' \
    trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
    trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
    trainer.args.num_train_epochs=$num_epochs \
    trainer.args.eval_strategy=no \
    trainer.args.eval_on_start=False \
    trainer.args.gradient_checkpointing=True \
    trainer.args.learning_rate=$lr \
    trainer.method_args.gamma=$gamma \
    trainer.method_args.alpha=$alpha \
    trainer.method_args.beta=$beta \
    model=$model

########################################
# 2. TOFU EVALUATION (UNLEARNED)
########################################
# echo ""
# echo "=================================================="
# echo " TOFU Eval — ${task_name}"
# echo "=================================================="

# HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 \
#     python src/eval.py \
#     experiment=eval/tofu/default.yaml \
#     'eval=[tofu]' \
#     model=$model \
#     task_name=${task_name} \
#     model.model_args.pretrained_model_name_or_path=$(pwd)/saves/unlearn/${task_name} \
#     +model.model_args.token=$HF_TOKEN \
#     +model.tokenizer_args.token=$HF_TOKEN \
#     ++model.model_args.device_map='auto' \
#     paths.output_dir=$(pwd)/saves/unlearn/${task_name}/tofu_evals

########################################
# 3. PARAPHRASE EVALUATION
########################################
# echo ""
# echo "=================================================="
# echo " Paraphrase Eval — ${task_name}"
# echo "=================================================="

# HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 \
#     python src/eval.py \
#     experiment=eval/paraphrase/default.yaml \
#     'eval=[paraphrase_tofu]' \
#     model=$model \
#     task_name=${task_name} \
#     model.model_args.pretrained_model_name_or_path=$(pwd)/saves/unlearn/${task_name} \
#     +model.model_args.token=$HF_TOKEN \
#     +model.tokenizer_args.token=$HF_TOKEN \
#     ++model.model_args.device_map='auto' \
#     '~eval.paraphrase.metrics.winrate' \
#     paths.output_dir=$(pwd)/saves/unlearn/${task_name}/evals/paraphrase_evals