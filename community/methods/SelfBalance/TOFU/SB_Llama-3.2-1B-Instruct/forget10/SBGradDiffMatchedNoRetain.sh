#!/bin/bash
# SBGradDiffMatchedNoRetain — Unlearn (TOFU)
# Model: Llama-3.2-1B-Instruct | Split: forget10/retain90

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

# ── Hyperparameters (override via env vars; HP_KEY selects which entry to read) ──
HP_KEY="${HP_KEY:-SBGradDiffMatchedNoRetain}"
SBGradDiffMatchedNoRetain_lr=${SBGradDiffMatchedNoRetain_lr:-$(get_param "$HP_KEY" lr)}
SBGradDiffMatchedNoRetain_gamma=${SBGradDiffMatchedNoRetain_gamma:-$(get_param "$HP_KEY" gamma)}
SBGradDiffMatchedNoRetain_beta=${SBGradDiffMatchedNoRetain_beta:-$(get_param "$HP_KEY" beta)}
SBGradDiffMatchedNoRetain_scorer_lr=${SBGradDiffMatchedNoRetain_scorer_lr:-$(get_param "$HP_KEY" scorer_lr)}

update_every_n_steps=${UPDATE_EVERY_N_STEPS:-5}
TASK_SUFFIX=""
if [[ -n "${UPDATE_EVERY_N_STEPS:-}" ]]; then
    TASK_SUFFIX="_uev${UPDATE_EVERY_N_STEPS}"
fi
task_name="SB_TOFU/${model}/forget10/SBGradDiffMatchedNoRetain${TASK_SUFFIX}"
model_output="saves/unlearn/${task_name}"

# ══════════════════════════════════════════
#  1. Unlearn
# ══════════════════════════════════════════
echo ""
echo "============================================"
echo " SBGradDiffMatchedNoRetain — Unlearn (forget10)"
echo "============================================"

HYDRA_FULL_ERROR=1 \
    python \
    src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default \
    trainer=SBGradDiffMatchedNoRetainLearned \
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
    trainer.args.learning_rate=$SBGradDiffMatchedNoRetain_lr \
    trainer.method_args.gamma=$SBGradDiffMatchedNoRetain_gamma \
    trainer.method_args.beta=$SBGradDiffMatchedNoRetain_beta \
    trainer.method_args.scorer.cfg.input_dimension=2048 \
    trainer.method_args.scorer_trainer.optim_cfg.update_every_n_steps=$update_every_n_steps \
    +trainer.method_args.scorer_trainer.optim_cfg.scheduler=linear \
    trainer.method_args.scorer_trainer.optim_cfg.lr=$SBGradDiffMatchedNoRetain_scorer_lr \
    trainer.method_args.scorer_trainer.lambda_entropy=1 \
    trainer.method_args.scorer_trainer.lambda_population=10 \
    trainer.method_args.scorer_trainer.budget=0.2 \
    trainer.method_args.scorer_trainer.lambda_l2=1 \
    model=$model
