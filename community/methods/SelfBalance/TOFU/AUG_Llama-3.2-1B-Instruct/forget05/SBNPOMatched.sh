#!/bin/bash
# SBNPOMatched — Unlearn (TOFU)
# Model: Llama-3.2-1B-Instruct | Split: forget05/retain95

set -a
source .env
set +a

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

model="Llama-3.2-1B-Instruct"
forget_split="forget05"
retain_split="retain95"
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
SBNPOMatched_lr=${SBNPOMatched_lr:-$(get_param SBNPOMatched lr)}
SBNPOMatched_gamma=${SBNPOMatched_gamma:-$(get_param SBNPOMatched gamma)}
SBNPOMatched_alpha=${SBNPOMatched_alpha:-$(get_param SBNPOMatched alpha)}
SBNPOMatched_beta=${SBNPOMatched_beta:-$(get_param SBNPOMatched beta)}
SBNPOMatched_scorer_lr=${SBNPOMatched_scorer_lr:-$(get_param SBNPOMatched scorer_lr)}

task_name="AUG_TOFU/${model}/forget05/SBNPOMatched"
model_output="saves/unlearn/${task_name}"

# ══════════════════════════════════════════
#  1. Unlearn
# ══════════════════════════════════════════
echo ""
echo "============================================"
echo " SBNPOMatched — Unlearn (forget05)"
echo "============================================"

HYDRA_FULL_ERROR=1 \
    python \
    src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default \
    trainer=SBNPOMatchedLearned \
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
    trainer.args.learning_rate=$SBNPOMatched_lr \
    trainer.method_args.gamma=$SBNPOMatched_gamma \
    trainer.method_args.alpha=$SBNPOMatched_alpha \
    trainer.method_args.beta=$SBNPOMatched_beta \
    trainer.method_args.scorer.cfg.input_dimension=2048 \
    trainer.method_args.scorer_trainer.optim_cfg.update_every_n_steps=5 \
    +trainer.method_args.scorer_trainer.optim_cfg.scheduler=linear \
    trainer.method_args.scorer_trainer.optim_cfg.lr=$SBNPOMatched_scorer_lr \
    trainer.method_args.scorer_trainer.lambda_entropy=1 \
    trainer.method_args.scorer_trainer.lambda_population=10 \
    trainer.method_args.scorer_trainer.budget=0.2 \
    trainer.method_args.scorer_trainer.lambda_l2=1
