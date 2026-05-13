#!/bin/bash
# GradDiff — Unlearn + Eval (RWKU)
# Model: Phi-3-mini-4k-instruct

set -a
source .env
set +a

source /tmlscratch/nikolaou/open-unlearning/envs/.open-unlearning/bin/activate

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

model="Phi-3-mini-4k-instruct"
model_path="microsoft/Phi-3-mini-4k-instruct"

# model="Phi-3.5-mini-instruct"
# model_path="microsoft/Phi-3.5-mini-instruct"

trainer="GradDiff"
experiment="unlearn/rwku/positive_phi3.yaml"

num_epochs=5
batch_size=8
grad_accum=4
max_length=512

# ── Hyperparameters (override via env vars) ──
lr=${lr:-6e-7}
gamma=${gamma:-0.5}
alpha=${alpha:-0.5}

task_name="SB_RWKU/${model}/GradDiff"
model_output="saves/unlearn/${task_name}"

# ══════════════════════════════════════════
#  1. Unlearn
# ══════════════════════════════════════════
echo ""
echo "============================================"
echo " GradDiff — Unlearn (RWKU)"
echo "============================================"

HYDRA_FULL_ERROR=1 \
    python \
    src/train.py --config-name=unlearn.yaml \
    experiment=$experiment \
    trainer=$trainer \
    model=$model \
    task_name=$task_name \
    model.model_args.pretrained_model_name_or_path=$model_path \
    +model.model_args.token=$HF_TOKEN \
    ++model.model_args.device_map='auto' \
    +model.tokenizer_args.token=$HF_TOKEN \
    trainer.args.per_device_train_batch_size=$batch_size \
    trainer.args.gradient_accumulation_steps=$grad_accum \
    trainer.args.num_train_epochs=$num_epochs \
    trainer.args.eval_on_start=false \
    trainer.args.eval_strategy=no \
    trainer.args.learning_rate=$lr \
    trainer.method_args.gamma=$gamma \
    trainer.method_args.alpha=$alpha \
    data.forget.RWKU_train_positive_phi3.args.max_length=$max_length \
    data.retain.RWKU_train_positive_phi3.args.max_length=$max_length

# ══════════════════════════════════════════
#  2. RWKU Eval
# ══════════════════════════════════════════
# echo ""
# echo "============================================"
# echo " GradDiff — RWKU Eval"
# echo "============================================"

# HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 \
#     python src/eval.py \
#     experiment=eval/rwku/default.yaml \
#     'eval=[rwku]' \
#     model=$model \
#     task_name=$task_name \
#     model.model_args.torch_dtype=float16 \
#     model.model_args.pretrained_model_name_or_path=$(pwd)/${model_output} \
#     +model.model_args.token=$HF_TOKEN \
#     +model.tokenizer_args.token=$HF_TOKEN \
#     ++model.model_args.device_map='auto' \
#     paths.output_dir=$(pwd)/${model_output}/rwku_evals

# ══════════════════════════════════════════
#  3. Paraphrase Eval
# ══════════════════════════════════════════
# echo ""
# echo "============================================"
# echo " GradDiff — Paraphrase Eval (RWKU)"
# echo "============================================"

# HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 \
#     python src/eval.py \
#     experiment=eval/paraphrase/default.yaml \
#     'eval=[paraphrase_rwku]' \
#     model=$model \
#     task_name=$task_name \
#     model.model_args.pretrained_model_name_or_path=$(pwd)/${model_output} \
#     +model.model_args.token=$HF_TOKEN \
#     +model.tokenizer_args.token=$HF_TOKEN \
#     ++model.model_args.device_map='auto' \
#     '~eval.paraphrase.metrics.winrate' \
#     paths.output_dir=$(pwd)/${model_output}/paraphrase_evals