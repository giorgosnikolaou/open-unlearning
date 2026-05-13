#!/bin/bash
# SBWGAMatched — Unlearn (TOFU)
# Model: Llama-3.1-8B-Instruct | Split: forget10/retain90

set -a
source .env
set +a

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

model="Llama-3.1-8B-Instruct"
forget_split="forget10"
retain_split="retain90"
num_epochs=10
batch_size=8
grad_accum=4

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
SBWGAMatched_lr=${SBWGAMatched_lr:-$(get_best_lr SBWGAMatched)}
SBWGAMatched_gamma=${SBWGAMatched_gamma:-$(get_param SBWGAMatched gamma)}
SBWGAMatched_alpha=${SBWGAMatched_alpha:-$(get_param SBWGAMatched alpha)}
SBWGAMatched_beta=${SBWGAMatched_beta:-$(get_param SBWGAMatched beta)}
SBWGAMatched_scorer_lr=${SBWGAMatched_scorer_lr:-$(get_param SBWGAMatched scorer_lr)}

task_name="AUG_TOFU/${model}/forget10/SBWGAMatched"
model_output="saves/unlearn/${task_name}"

# ══════════════════════════════════════════
#  1. Unlearn
# ══════════════════════════════════════════
echo ""
echo "============================================"
echo " SBWGAMatched — Unlearn (forget10)"
echo "============================================"

HYDRA_FULL_ERROR=1 \
    python \
    src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default \
    trainer=SBWGAMatchedLearned \
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
    trainer.args.learning_rate=$SBWGAMatched_lr \
    trainer.method_args.gamma=$SBWGAMatched_gamma \
    trainer.method_args.alpha=$SBWGAMatched_alpha \
    trainer.method_args.beta=$SBWGAMatched_beta \
    trainer.method_args.scorer.cfg.input_dimension=4096 \
    trainer.method_args.scorer_trainer.optim_cfg.update_every_n_steps=5 \
    +trainer.method_args.scorer_trainer.optim_cfg.scheduler=linear \
    trainer.method_args.scorer_trainer.optim_cfg.lr=$SBWGAMatched_scorer_lr \
    trainer.method_args.scorer_trainer.lambda_entropy=1 \
    trainer.method_args.scorer_trainer.lambda_population=15 \
    trainer.method_args.scorer_trainer.budget=0.2 \
    trainer.method_args.scorer_trainer.lambda_l2=1 \
    model=$model
