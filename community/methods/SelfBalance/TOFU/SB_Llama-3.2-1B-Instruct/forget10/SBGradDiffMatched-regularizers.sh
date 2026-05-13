#!/bin/bash
# SBGradDiffMatched — Regularizer ablation (TOFU, forget10, 1B)
# Sweeps the 8 = 2^3 on/off combinations of (lambda_entropy, lambda_population, lambda_l2).
# "On" values are the manually-tuned 1B defaults from SCORER_OVERRIDES
# (1, 10, 1) for (entropy, population, l2). "Off" values are 0.

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

for ent in 0 1; do
  for pop in 0 1; do
    for l2 in 0 1; do
      ent_val=0; pop_val=0; l2_val=0
      [[ $ent == 1 ]] && ent_val=$ENT_ON
      [[ $pop == 1 ]] && pop_val=$POP_ON
      [[ $l2  == 1 ]] && l2_val=$L2_ON

      tag="E${ent}_P${pop}_L${l2}"
      task_name="SB_TOFU/${model}/forget10/SBGradDiffMatched/regs/${tag}"
      model_output="saves/unlearn/${task_name}"

      echo ""
      echo "============================================"
      echo " SBGradDiffMatched — regs ${tag} (entropy=${ent_val} pop=${pop_val} l2=${l2_val})"
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
          trainer.args.learning_rate=$LR \
          trainer.method_args.gamma=$GAMMA \
          trainer.method_args.alpha=$ALPHA \
          trainer.method_args.beta=$BETA \
          trainer.method_args.scorer.cfg.input_dimension=2048 \
          trainer.method_args.scorer_trainer.optim_cfg.update_every_n_steps=5 \
          +trainer.method_args.scorer_trainer.optim_cfg.scheduler=linear \
          trainer.method_args.scorer_trainer.optim_cfg.lr=$SCORER_LR \
          trainer.method_args.scorer_trainer.lambda_entropy=$ent_val \
          trainer.method_args.scorer_trainer.lambda_population=$pop_val \
          trainer.method_args.scorer_trainer.budget=0.2 \
          trainer.method_args.scorer_trainer.lambda_l2=$l2_val
    done
  done
done
