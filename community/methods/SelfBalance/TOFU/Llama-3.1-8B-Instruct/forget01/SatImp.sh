#!/bin/bash
# SatImp — Unlearn + Eval (TOFU + Paraphrase)
# Model: Llama-3.1-8B-Instruct | Split: forget01/retain99

set -a
source .env
set +a

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

model="Llama-3.1-8B-Instruct"
forget_split="forget01"
retain_split="retain99"
num_epochs=10
batch_size=8
grad_accum=4

winrate_baseline="saves/eval/SB_TOFU/${model}/baselines/${retain_split}_${forget_split}/paraphrase_evals/repetitiveness/model.jsonl"

# ── HP search summary (override with SUMMARY_JSON env var) ──
SUMMARY_JSON="${SUMMARY_JSON:-hyperparam/tofu_forget10_8B/bayesian_summary.json}"

get_param() {
    python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d[sys.argv[2]]['source_params'][sys.argv[3]])" \
        "$SUMMARY_JSON" "$1" "$2"
}

get_best_lr() {
    python3 -c "import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]]['best_lr'])" \
        "$SUMMARY_JSON" "$1"
}

# ── Hyperparameters (override via env vars) ──
SATIMP_lr=${SATIMP_lr:-$(get_best_lr SatImp)}
SATIMP_gamma=${SATIMP_gamma:-$(get_param SatImp gamma)}
SATIMP_alpha=${SATIMP_alpha:-$(get_param SatImp alpha)}
SATIMP_beta1=${SATIMP_beta1:-$(get_param SatImp beta1)}
SATIMP_beta2=${SATIMP_beta2:-$(get_param SatImp beta2)}

task_name="SB_TOFU/${model}/${forget_split}/SatImp"
model_output="saves/unlearn/${task_name}"

# ══════════════════════════════════════════
#  1. Unlearn
# ══════════════════════════════════════════
echo ""
echo "============================================"
echo " SatImp — Unlearn (${forget_split})"
echo "============================================"

HYDRA_FULL_ERROR=1 \
    python \
    src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default \
    trainer=SatImp \
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
    trainer.args.gradient_checkpointing=True \
    trainer.args.per_device_train_batch_size=$batch_size \
    trainer.args.gradient_accumulation_steps=$grad_accum \
    trainer.args.learning_rate=$SATIMP_lr \
    trainer.method_args.gamma=$SATIMP_gamma \
    trainer.method_args.alpha=$SATIMP_alpha \
    trainer.method_args.beta1=$SATIMP_beta1 \
    trainer.method_args.beta2=$SATIMP_beta2

# ══════════════════════════════════════════
#  2. TOFU Eval
# ══════════════════════════════════════════
# echo ""
# echo "============================================"
# echo " SatImp — TOFU Eval (${forget_split})"
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
# #     forget_split=$forget_split \
#     paths.output_dir=$(pwd)/${model_output}/tofu_evals

# ══════════════════════════════════════════
#  3. Paraphrase Eval
# ══════════════════════════════════════════
# echo ""
# echo "============================================"
# echo " SatImp — Paraphrase Eval (${forget_split})"
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
# #     eval.paraphrase.forget_split=$forget_split \
#     +eval.paraphrase.metrics.winrate.baseline_path=${winrate_baseline} \
#     paths.output_dir=$(pwd)/${model_output}/paraphrase_evals
