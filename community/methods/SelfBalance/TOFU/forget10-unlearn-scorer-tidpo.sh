#!/bin/bash
# SelfBalancing GradDiff + LearnedTIDPOScorer — TOFU Paraphrase Train & Eval

set -a
source .env
set +a

source /tmlscratch/nikolaou/open-unlearning/envs/.open-unlearning/bin/activate

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

declare -A HIDDEN_DIM_MAP=(
    ["Llama-3.2-1B-Instruct"]=2048
    ["Llama-3.2-3B-Instruct"]=3072
    ["Llama-3.1-8B-Instruct"]=4096
)

model="Llama-3.2-1B-Instruct"
trainer="SBGradDiffLearnedTIDPO"
experiment="unlearn/tofu/default.yaml"
scorer_in_dim=${HIDDEN_DIM_MAP[$model]}

per_device_train_batch_size=16
gradient_accumulation_steps=2

lr=1e-5
gamma=5.0
alpha=0.5

# TIDPO params
lam=0.5
prior_mean=2.0
prior_std=4.0

# Scorer trainer params
lambda_entropy=1.0
lambda_population=10.0
budget=0.2
lambda_l2=1.0
update_every_n_steps=5

task_name="tofu_${model}_${trainer}_lr${lr}_gamma${gamma}_alpha${alpha}"
params="lam${lam}_pmean${prior_mean}_pstd${prior_std}_ent${lambda_entropy}_pop${lambda_population}_budget${budget}_l2${lambda_l2}_up${update_every_n_steps}"
prefix="LearnedTIDPO/beta5_"

complete_name="${prefix}${params}_${task_name}"

###################
# 1. UNLEARNING
###################
# echo "==========================================="
# echo " LearnedTIDPOScorer — ${complete_name}"
# echo "==========================================="

# HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 \
#     python src/train.py --config-name=unlearn.yaml \
#     experiment=$experiment \
#     trainer=$trainer \
#     task_name=${complete_name} \
#     model=$model \
#     model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_${model}_full \
#     +model.model_args.token=$HF_TOKEN \
#     +model.tokenizer_args.token=$HF_TOKEN \
#     +model.model_args.device_map='auto' \
#     trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
#     trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
#     trainer.args.eval_strategy=no \
#     trainer.args.eval_on_start=False \
#     trainer.args.learning_rate=$lr \
#     trainer.method_args.gamma=$gamma \
#     trainer.method_args.alpha=$alpha \
#     trainer.method_args.scorer.cfg.input_dimension=$scorer_in_dim \
#     trainer.method_args.scorer.lam=$lam \
#     trainer.method_args.scorer.prior_mean=$prior_mean \
#     trainer.method_args.scorer.prior_std=$prior_std \
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
echo ""
echo "=================================================="
echo " Paraphrase Eval — ${complete_name}"
echo "=================================================="

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 \
    python src/eval.py \
    experiment=eval/paraphrase/default.yaml \
    'eval=[paraphrase_tofu]' \
    model=$model \
    task_name=${complete_name} \
    model.model_args.pretrained_model_name_or_path=$(pwd)/saves/unlearn/${complete_name} \
    +model.model_args.token=$HF_TOKEN \
    +model.tokenizer_args.token=$HF_TOKEN \
    ++model.model_args.device_map='auto' \
    +eval.paraphrase.metrics.winrate.baseline_path=saves/eval/tofu_para_${model}/repetitiveness/model.jsonl \
    paths.output_dir=$(pwd)/saves/unlearn/${complete_name}/evals
