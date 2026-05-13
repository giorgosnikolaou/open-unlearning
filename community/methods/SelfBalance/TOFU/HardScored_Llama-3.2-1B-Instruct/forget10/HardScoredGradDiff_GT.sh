#!/bin/bash
# HardScoredGradDiff_GT — Unlearn (TOFU)
# Pre-computed binary masks (GT ranges) | Model: Llama-3.2-1B-Instruct | Split: forget10/retain90

set -a
source .env
set +a

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

model="Llama-3.2-1B-Instruct"
forget_split="forget10"
retain_split="retain90"
num_epochs=10
batch_size=8
grad_accum=4

gt_path="data/gpt-selected-tokens-tofu/forget10_with_common_words_gpt"

# ── HP search summary (override with SB_SUMMARY_JSON env var) ──
SB_SUMMARY_JSON="${SB_SUMMARY_JSON:-hyperparam/tofu_forget10/sb_bayesian_summary.json}"

get_param() {
    python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d[sys.argv[2]]['best_params'][sys.argv[3]])" \
        "$SB_SUMMARY_JSON" "$1" "$2"
}

# ── Hyperparameters (override via env vars) ──
HardScoredGradDiff_GT_lr=${HardScoredGradDiff_GT_lr:-$(get_param HardScoredGradDiff_GT lr)}
HardScoredGradDiff_GT_gamma=${HardScoredGradDiff_GT_gamma:-$(get_param HardScoredGradDiff_GT gamma)}
HardScoredGradDiff_GT_alpha=${HardScoredGradDiff_GT_alpha:-$(get_param HardScoredGradDiff_GT alpha)}

task_name="SB_TOFU/${model}/forget10/HardScoredGradDiff_GT"
model_output="saves/unlearn/${task_name}"

# ══════════════════════════════════════════
#  1. Unlearn
# ══════════════════════════════════════════
echo ""
echo "============================================"
echo " HardScoredGradDiff_GT — Unlearn (forget10)"
echo "============================================"

HYDRA_FULL_ERROR=1 \
    python \
    src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default \
    trainer=HardScoredGradDiff \
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
    trainer.args.learning_rate=$HardScoredGradDiff_GT_lr \
    trainer.method_args.gamma=$HardScoredGradDiff_GT_gamma \
    trainer.method_args.alpha=$HardScoredGradDiff_GT_alpha \
    trainer.method_args.scoring_args.gt_path=$gt_path
