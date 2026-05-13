#!/bin/bash
# Scorer (SBGradDiffMatched) — Update-frequency ablation (TOFU, forget10, 1B)
# Runs ONE update_every_n_steps value per invocation.
#
# Usage:
#   bash update-frequency.sh UEV     # positional (positive int)
#   UEV=5 bash update-frequency.sh   # env var
#
# Examples:
#   bash update-frequency.sh 1
#   bash update-frequency.sh 5
#   bash update-frequency.sh 10
#
# Sweep multiple values by launching separately:
#   for u in 1 5 10; do bash update-frequency.sh $u; done

set -euo pipefail

set -a
source .env
set +a

# ── Args / env ──
UEV="${1:-${UEV:-}}"
if [[ -z "$UEV" ]]; then
  echo "Usage: $0 UEV  (positive integer, e.g. 1, 5, 10)" >&2
  echo "   or: UEV=5 $0" >&2
  exit 1
fi
[[ "$UEV" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid UEV '$UEV' (must be positive integer)" >&2; exit 1; }

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

model="Llama-3.2-1B-Instruct"
forget_split="forget10"
retain_split="retain90"
num_epochs=10
GPU=${GPU:-0}

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

task_name="SB_TOFU/${model}/forget10/ScorerAblations/update-frequency/uev${UEV}"
model_output="saves/unlearn/${task_name}"

echo ""
echo "============================================"
echo " Scorer Ablation: update-frequency uev=${UEV}"
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
    trainer.method_args.scorer_trainer.optim_cfg.update_every_n_steps=$UEV \
    trainer.method_args.scorer_trainer.optim_cfg.lr=$SCORER_LR \
    +trainer.method_args.scorer_trainer.optim_cfg.scheduler=linear \
    trainer.method_args.scorer_trainer.lambda_entropy=1 \
    trainer.method_args.scorer_trainer.lambda_population=10 \
    trainer.method_args.scorer_trainer.budget=0.2 \
    trainer.method_args.scorer_trainer.lambda_l2=1