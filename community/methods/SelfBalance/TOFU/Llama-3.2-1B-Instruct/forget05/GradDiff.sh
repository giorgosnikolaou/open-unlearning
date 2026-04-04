#!/bin/bash
# GradDiff — Unlearn + Eval (TOFU + Paraphrase)
# Model: Llama-3.2-1B-Instruct | Split: forget05/retain95

set -a
source .env
set +a

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

model="Llama-3.2-1B-Instruct"
forget_split="forget05"
retain_split="retain95"
GPU=${GPU:-0}
num_epochs=10

winrate_baseline="saves/eval/SB_TOFU/${model}/baselines/${retain_split}_${forget_split}/paraphrase_evals/repetitiveness/model.jsonl"

# ── HP search summary (override with SUMMARY_JSON env var) ──
SUMMARY_JSON="${SUMMARY_JSON:-hyperparam/tofu_forget10/bayesian_summary.json}"

get_param() {
    python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d[sys.argv[2]]['best_params'][sys.argv[3]])" \
        "$SUMMARY_JSON" "$1" "$2"
}

# ── Hyperparameters (override via env vars) ──
GRADDIFF_lr=${GRADDIFF_lr:-$(get_param GradDiff lr)}
GRADDIFF_gamma=${GRADDIFF_gamma:-$(get_param GradDiff gamma)}
GRADDIFF_alpha=${GRADDIFF_alpha:-$(get_param GradDiff alpha)}
RETAIN_LOSS_TYPE=${RETAIN_LOSS_TYPE:-NLL}

task_name="SB_TOFU/${model}/${forget_split}/GradDiff"
model_output="saves/unlearn/${task_name}"

# ══════════════════════════════════════════
#  1. Unlearn
# ══════════════════════════════════════════
echo ""
echo "============================================"
echo " GradDiff — Unlearn (${forget_split})"
echo "============================================"

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=$GPU \
    python src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default \
    trainer=GradDiff \
    model=$model \
    task_name=$task_name \
    model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_${model}_full \
    +model.model_args.token=$HF_TOKEN \
    +model.tokenizer_args.token=$HF_TOKEN \
    ++model.model_args.device_map='auto' \
    forget_split=$forget_split \
    retain_split=$retain_split \
    trainer.args.num_train_epochs=$num_epochs \
    trainer.args.eval_on_start=false \
    trainer.args.eval_strategy=no \
    trainer.args.learning_rate=$GRADDIFF_lr \
    trainer.method_args.gamma=$GRADDIFF_gamma \
    trainer.method_args.alpha=$GRADDIFF_alpha \
    trainer.method_args.retain_loss_type=$RETAIN_LOSS_TYPE

# ══════════════════════════════════════════
#  2. TOFU Eval
# ══════════════════════════════════════════
# echo ""
# echo "============================================"
# echo " GradDiff — TOFU Eval (${forget_split})"
# echo "============================================"

# HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=$GPU \
#     python src/eval.py \
#     experiment=eval/tofu/default.yaml \
#     'eval=[tofu]' \
#     model=$model \
#     task_name=$task_name \
#     model.model_args.pretrained_model_name_or_path=$(pwd)/${model_output} \
#     +model.model_args.token=$HF_TOKEN \
#     +model.tokenizer_args.token=$HF_TOKEN \
#     ++model.model_args.device_map='auto' \
#     forget_split=$forget_split \
#     paths.output_dir=$(pwd)/${model_output}/tofu_evals

# ══════════════════════════════════════════
#  3. Paraphrase Eval
# ══════════════════════════════════════════
# echo ""
# echo "============================================"
# echo " GradDiff — Paraphrase Eval (${forget_split})"
# echo "============================================"

# HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=$GPU \
#     python src/eval.py \
#     experiment=eval/paraphrase/default.yaml \
#     'eval=[paraphrase_tofu]' \
#     model=$model \
#     task_name=$task_name \
#     model.model_args.pretrained_model_name_or_path=$(pwd)/${model_output} \
#     +model.model_args.token=$HF_TOKEN \
#     +model.tokenizer_args.token=$HF_TOKEN \
#     ++model.model_args.device_map='auto' \
#     eval.paraphrase.forget_split=$forget_split \
#     +eval.paraphrase.metrics.winrate.baseline_path=${winrate_baseline} \
#     paths.output_dir=$(pwd)/${model_output}/paraphrase_evals
