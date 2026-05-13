#!/bin/bash
# HardScoredGradDiff_SU_Ngram — Unlearn (TOFU)
# Pre-computed binary masks (SU-Ngram) | Model: Llama-3.2-1B-Instruct | Split: forget10/retain90

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

# ── HP search summary (override with SB_SUMMARY_JSON env var) ──
SB_SUMMARY_JSON="${SB_SUMMARY_JSON:-hyperparam/tofu_forget10/sb_bayesian_summary.json}"

get_param() {
    python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d[sys.argv[2]]['best_params'][sys.argv[3]])" \
        "$SB_SUMMARY_JSON" "$1" "$2"
}

# ── Hyperparameters (override via env vars) ──
HardScoredGradDiff_SU_Ngram_lr=${HardScoredGradDiff_SU_Ngram_lr:-$(get_param HardScoredGradDiff_SU_Ngram lr)}
HardScoredGradDiff_SU_Ngram_gamma=${HardScoredGradDiff_SU_Ngram_gamma:-$(get_param HardScoredGradDiff_SU_Ngram gamma)}
HardScoredGradDiff_SU_Ngram_alpha=${HardScoredGradDiff_SU_Ngram_alpha:-$(get_param HardScoredGradDiff_SU_Ngram alpha)}

task_name="SB_TOFU/${model}/forget10/HardScoredGradDiff_SU_Ngram"
model_output="saves/unlearn/${task_name}"

# ══════════════════════════════════════════
#  1. Unlearn
# ══════════════════════════════════════════
echo ""
echo "============================================"
echo " HardScoredGradDiff_SU_Ngram — Unlearn (forget10)"
echo "============================================"

HYDRA_FULL_ERROR=1 \
    python \
    src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default \
    trainer=HardScoredGradDiff_SU_Ngram \
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
    trainer.args.learning_rate=$HardScoredGradDiff_SU_Ngram_lr \
    trainer.method_args.gamma=$HardScoredGradDiff_SU_Ngram_gamma \
    trainer.method_args.alpha=$HardScoredGradDiff_SU_Ngram_alpha
