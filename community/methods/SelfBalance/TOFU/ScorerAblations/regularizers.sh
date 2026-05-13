#!/bin/bash
# Scorer (SBGradDiffMatched) — Regularizer ablation (TOFU, forget10, 1B)
# Runs ONE combo of (lambda_entropy, lambda_population, lambda_l2) per invocation.
# "On" values: (1, 10, 1). "Off" is 0.
#
# Usage:
#   bash regularizers.sh ENT POP L2          # positional, each 0 or 1
#   ENT=1 POP=0 L2=1 bash regularizers.sh    # env vars
#
# Examples:
#   bash regularizers.sh 1 1 1               # all on
#   bash regularizers.sh 0 1 0               # population only
#
# Sweep all 8 combos by launching separately:
#   for e in 0 1; do for p in 0 1; do for l in 0 1; do
#     bash regularizers.sh $e $p $l
#   done; done; done

set -euo pipefail

set -a
source .env
set +a

# ── Args / env ──
ENT="${1:-${ENT:-}}"
POP="${2:-${POP:-}}"
L2="${3:-${L2:-}}"

if [[ -z "$ENT" || -z "$POP" || -z "$L2" ]]; then
  echo "Usage: $0 ENT POP L2  (each 0 or 1)" >&2
  echo "   or: ENT=1 POP=0 L2=1 $0" >&2
  exit 1
fi
for v in "$ENT" "$POP" "$L2"; do
  [[ "$v" =~ ^[01]$ ]] || { echo "Invalid value '$v' (must be 0 or 1)" >&2; exit 1; }
done

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

model="Llama-3.2-1B-Instruct"
forget_split="forget10"
retain_split="retain90"
num_epochs=10
GPU=${GPU:-0}

# ── HP search summary ──
SB_SUMMARY_JSON="${SB_SUMMARY_JSON:-hyperparam/tofu_forget10/sb_bayesian_summary.json}"

get_param() {
    python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d[sys.argv[2]]['best_params'][sys.argv[3]])" \
        "$SB_SUMMARY_JSON" "$1" "$2"
}

HP_KEY="${HP_KEY:-SBGradDiffMatched}"
LR=$(get_param "$HP_KEY" lr)
GAMMA=$(get_param "$HP_KEY" gamma)
ALPHA=$(get_param "$HP_KEY" alpha)
BETA=$(get_param "$HP_KEY" beta)
SCORER_LR=$(get_param "$HP_KEY" scorer_lr)

# ── Regularizer "on" values ──
ENT_ON=1
POP_ON=10
L2_ON=1

ent_val=0; pop_val=0; l2_val=0
[[ $ENT == 1 ]] && ent_val=$ENT_ON
[[ $POP == 1 ]] && pop_val=$POP_ON
[[ $L2  == 1 ]] && l2_val=$L2_ON

tag="E${ENT}_P${POP}_L${L2}"
task_name="SB_TOFU/${model}/forget10/ScorerAblations/regularizers/${tag}"
model_output="saves/unlearn/${task_name}"

echo ""
echo "============================================"
echo " Scorer Ablation: regularizers ${tag} (entropy=${ent_val} pop=${pop_val} l2=${l2_val})"
echo "============================================"

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=$GPU \
    python src/train.py --config-name=unlearn.yaml \
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
    trainer.args.learning_rate=$LR \
    trainer.method_args.gamma=$GAMMA \
    trainer.method_args.alpha=$ALPHA \
    trainer.method_args.beta=$BETA \
    trainer.method_args.scorer.cfg.input_dimension=2048 \
    trainer.method_args.scorer_trainer.optim_cfg.update_every_n_steps=5 \
    trainer.method_args.scorer_trainer.optim_cfg.lr=$SCORER_LR \
    +trainer.method_args.scorer_trainer.optim_cfg.scheduler=linear \
    trainer.method_args.scorer_trainer.lambda_entropy=$ent_val \
    trainer.method_args.scorer_trainer.lambda_population=$pop_val \
    trainer.method_args.scorer_trainer.budget=0.2 \
    trainer.method_args.scorer_trainer.lambda_l2=$l2_val