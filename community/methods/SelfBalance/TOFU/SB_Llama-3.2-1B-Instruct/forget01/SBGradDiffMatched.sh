#!/bin/bash
# SBGradDiffMatched — Unlearn (TOFU)
# Model: Llama-3.2-1B-Instruct | Split: forget01/retain99

set -a
source .env
set +a

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

model="Llama-3.2-1B-Instruct"
forget_split="forget01"
retain_split="retain99"
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
HP_KEY="${HP_KEY:-SBGradDiffMatched}"
SBGradDiffMatched_lr=${SBGradDiffMatched_lr:-$(get_param "$HP_KEY" lr)}
SBGradDiffMatched_gamma=${SBGradDiffMatched_gamma:-$(get_param "$HP_KEY" gamma)}
SBGradDiffMatched_alpha=${SBGradDiffMatched_alpha:-$(get_param "$HP_KEY" alpha)}
SBGradDiffMatched_beta=${SBGradDiffMatched_beta:-$(get_param "$HP_KEY" beta)}
SBGradDiffMatched_scorer_lr=${SBGradDiffMatched_scorer_lr:-$(get_param "$HP_KEY" scorer_lr)}

task_name="SB_TOFU/${model}/forget01/SBGradDiffMatched"
model_output="saves/unlearn/${task_name}"

# ══════════════════════════════════════════
#  1. Unlearn
# ══════════════════════════════════════════
echo ""
echo "============================================"
echo " SBGradDiffMatched — Unlearn (forget01)"
echo "============================================"

HYDRA_FULL_ERROR=1 \
    python \
    src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default \
    trainer=SBGradDiffMatchedLearned \
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
    trainer.args.learning_rate=$SBGradDiffMatched_lr \
    trainer.method_args.gamma=$SBGradDiffMatched_gamma \
    trainer.method_args.alpha=$SBGradDiffMatched_alpha \
    trainer.method_args.beta=$SBGradDiffMatched_beta \
    trainer.method_args.scorer.cfg.input_dimension=2048 \
    trainer.method_args.scorer_trainer.optim_cfg.update_every_n_steps=5 \
    +trainer.method_args.scorer_trainer.optim_cfg.scheduler=linear \
    trainer.method_args.scorer_trainer.optim_cfg.lr=$SBGradDiffMatched_scorer_lr \
    trainer.method_args.scorer_trainer.lambda_entropy=1 \
    trainer.method_args.scorer_trainer.lambda_population=10 \
    trainer.method_args.scorer_trainer.budget=0.2 \
    trainer.method_args.scorer_trainer.lambda_l2=1 \
    model=$model
