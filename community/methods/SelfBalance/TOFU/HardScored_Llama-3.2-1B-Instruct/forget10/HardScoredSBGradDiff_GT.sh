#!/bin/bash
# HardScoredSBGradDiff_GT — Unlearn (TOFU)
# Pre-computed binary masks (GT) + SB loss | Model: Llama-3.2-1B-Instruct | Split: forget10/retain90

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
HardScoredSBGradDiff_GT_lr=${HardScoredSBGradDiff_GT_lr:-$(get_param HardScoredSBGradDiff_GT lr)}
HardScoredSBGradDiff_GT_gamma=${HardScoredSBGradDiff_GT_gamma:-$(get_param HardScoredSBGradDiff_GT gamma)}
HardScoredSBGradDiff_GT_alpha=${HardScoredSBGradDiff_GT_alpha:-$(get_param HardScoredSBGradDiff_GT alpha)}
HardScoredSBGradDiff_GT_beta=${HardScoredSBGradDiff_GT_beta:-$(get_param HardScoredSBGradDiff_GT beta)}

task_name="SB_TOFU/${model}/forget10/HardScoredSBGradDiff_GT"
model_output="saves/unlearn/${task_name}"

# ══════════════════════════════════════════
#  1. Unlearn
# ══════════════════════════════════════════
echo ""
echo "============================================"
echo " HardScoredSBGradDiff_GT — Unlearn (forget10)"
echo "============================================"

HYDRA_FULL_ERROR=1 \
    python \
    src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default \
    trainer=HardScoredSBGradDiff \
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
    trainer.args.learning_rate=$HardScoredSBGradDiff_GT_lr \
    trainer.method_args.gamma=$HardScoredSBGradDiff_GT_gamma \
    trainer.method_args.alpha=$HardScoredSBGradDiff_GT_alpha \
    trainer.method_args.beta=$HardScoredSBGradDiff_GT_beta \
    trainer.method_args.scoring_args.gt_path=$gt_path
