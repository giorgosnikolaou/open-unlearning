#!/bin/bash
# SelfBalancing GradDiff + LearnedScorer — TOFU forget10
# Model: Llama-3.1-8B-Instruct (accelerate + DeepSpeed ZeRO-3)

set -a
source .env
set +a

source /tmlscratch/nikolaou/open-unlearning/envs/.open-unlearning/bin/activate

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

model="Llama-3.1-8B-Instruct"
trainer="SBGradDiffLearned"
experiment="unlearn/tofu/default.yaml"
scorer_in_dim=4096

per_device_train_batch_size=4
gradient_accumulation_steps=8

per_device_train_batch_size=8
gradient_accumulation_steps=2

# per_device_train_batch_size=16
# gradient_accumulation_steps=2

# ── HP search summary (override with SUMMARY_JSON env var) ──
SUMMARY_JSON="${SUMMARY_JSON:-hyperparam/tofu_forget10/bayesian_summary.json}"

get_param() {
    python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d[sys.argv[2]]['best_params'][sys.argv[3]])" \
        "$SUMMARY_JSON" "$1" "$2"
}

# ── Model hyperparameters (default from hpsearch, override via env vars) ──
# lr=1e-5; gamma=5.0; alpha=0.5  # manual override
lr=${lr:-$(get_param Scorer lr)}
gamma=${gamma:-$(get_param Scorer gamma)}
alpha=${alpha:-$(get_param Scorer alpha)}
beta=${beta:-$(get_param Scorer beta)}

# ── Scorer hyperparameters ──
scorer_lr=5e-2
lambda_entropy=1.0
lambda_population=10.0
budget=0.2
lambda_l2=1.0
update_every_n_steps=5

task_name="tofu_${model}_${trainer}_lr${lr}_gamma${gamma}_alpha${alpha}"
params="ent${lambda_entropy}_pop${lambda_population}_budget${budget}_l2${lambda_l2}_up${update_every_n_steps}_slr${scorer_lr}"
prefix="LearnedLlama3.1/bs16_accum2_"
prefix="LearnedLlama3.1/bs8_accum4_"
prefix="LearnedLlama3.1/2gpu_bs8_accum2_"

complete_name="${prefix}${params}_${task_name}"

###################
# 1. UNLEARNING
###################
echo "==========================================="
echo " LearnedScorer — ${complete_name}"
echo "==========================================="

HYDRA_FULL_ERROR=1 \
    accelerate launch --config_file configs/accelerate/default_config.yaml \
    src/train.py --config-name=unlearn.yaml \
    experiment=$experiment \
    trainer=$trainer \
    task_name=${complete_name} \
    model=$model \
    model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_${model}_full \
    +model.model_args.token=$HF_TOKEN \
    +model.tokenizer_args.token=$HF_TOKEN \
    trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
    trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
    trainer.args.eval_strategy=no \
    trainer.args.eval_on_start=False \
    trainer.args.gradient_checkpointing=True \
    trainer.args.learning_rate=$lr \
    trainer.method_args.gamma=$gamma \
    trainer.method_args.alpha=$alpha \
    trainer.method_args.beta=$beta \
    trainer.method_args.scorer.cfg.input_dimension=$scorer_in_dim \
    trainer.method_args.scorer_trainer.optim_cfg.lr=$scorer_lr \
    trainer.method_args.scorer_trainer.optim_cfg.update_every_n_steps=$update_every_n_steps \
    +trainer.method_args.scorer_trainer.optim_cfg.scheduler=linear \
    trainer.method_args.scorer_trainer.lambda_entropy=$lambda_entropy \
    trainer.method_args.scorer_trainer.lambda_population=$lambda_population \
    trainer.method_args.scorer_trainer.budget=$budget \
    trainer.method_args.scorer_trainer.lambda_l2=$lambda_l2

# HYDRA_FULL_ERROR=1 \
#     python \
#     src/train.py --config-name=unlearn.yaml \
#     experiment=$experiment \
#     trainer=$trainer \
#     task_name=${complete_name} \
#     model=$model \
#     model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_${model}_full \
#     +model.model_args.token=$HF_TOKEN \
#     ++model.model_args.device_map="auto" \
#     +model.tokenizer_args.token=$HF_TOKEN \
#     trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
#     trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
#     trainer.args.eval_strategy=no \
#     trainer.args.eval_on_start=False \
#     trainer.args.gradient_checkpointing=True \
#     trainer.args.learning_rate=$lr \
#     trainer.method_args.gamma=$gamma \
#     trainer.method_args.alpha=$alpha \
#     trainer.method_args.beta=$beta \
#     trainer.method_args.scorer.cfg.input_dimension=$scorer_in_dim \
#     trainer.method_args.scorer_trainer.optim_cfg.lr=$scorer_lr \
#     trainer.method_args.scorer_trainer.optim_cfg.update_every_n_steps=$update_every_n_steps \
#     +trainer.method_args.scorer_trainer.optim_cfg.scheduler=linear \
#     trainer.method_args.scorer_trainer.lambda_entropy=$lambda_entropy \
#     trainer.method_args.scorer_trainer.lambda_population=$lambda_population \
#     trainer.method_args.scorer_trainer.budget=$budget \
#     trainer.method_args.scorer_trainer.lambda_l2=$lambda_l2

########################################
# 2. TOFU EVALUATION (UNLEARNED)
########################################
# echo ""
# echo "=================================================="
# echo " TOFU Eval — ${complete_name}"
# echo "=================================================="

# HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 \
#     python src/eval.py \
#     experiment=eval/tofu/default.yaml \
#     'eval=[tofu]' \
#     model=$model \
#     task_name=${complete_name} \
#     model.model_args.pretrained_model_name_or_path=$(pwd)/saves/unlearn/${complete_name} \
#     +model.model_args.token=$HF_TOKEN \
#     +model.tokenizer_args.token=$HF_TOKEN \
#     ++model.model_args.device_map='auto' \
#     paths.output_dir=$(pwd)/saves/unlearn/${complete_name}/tofu_evals

########################################
# 3. PARAPHRASE EVALUATION
########################################
# echo ""
# echo "=================================================="
# echo " Paraphrase Eval — ${complete_name}"
# echo "=================================================="

# HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 \
#     python src/eval.py \
#     experiment=eval/paraphrase/default.yaml \
#     'eval=[paraphrase_tofu]' \
#     model=$model \
#     task_name=${complete_name} \
#     model.model_args.pretrained_model_name_or_path=$(pwd)/saves/unlearn/${complete_name} \
#     +model.model_args.token=$HF_TOKEN \
#     +model.tokenizer_args.token=$HF_TOKEN \
#     ++model.model_args.device_map='auto' \
#     '~eval.paraphrase.metrics.winrate' \
#     paths.output_dir=$(pwd)/saves/unlearn/${complete_name}/evals/paraphrase_evals
