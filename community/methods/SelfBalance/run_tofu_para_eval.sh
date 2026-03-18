#!/bin/bash
# SelfBalance Unlearning (Regularized Scorer) + TOFU Paraphrase Evaluation Script

set -a
source .env
set +a

source /tmlscratch/nikolaou/open-unlearning/envs/.open-unlearning/bin/activate

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

# Model name -> hidden dimension mapping
declare -A HIDDEN_DIM_MAP=(
    ["Llama-3.2-1B-Instruct"]=2048
    ["Llama-3.2-3B-Instruct"]=3072
    ["Llama-3.1-8B-Instruct"]=4096
    ["Phi-3.5-mini-instruct"]=3072
)

# Configuration
baseline_trainer="GradDiff"
trainer="SelfBalancing$baseline_trainer"
model="Llama-3.2-1B-Instruct"
scorer_in_dim=${HIDDEN_DIM_MAP[$model]}

# Model and data paths
model_path="open-unlearning/tofu_${model}_full"

# Experiment configuration (uses ngiorgos/TOFU paraphrase datasets)
experiment="unlearn/tofu/para.yaml"

# Hyperparameters
lr=1e-5
gamma=1.0
alpha=1.0
gamma=5.0
alpha=0.5

# Scorer regularization
rho_ent=0.0
# rho_ent=5.0
rho_pop=10.0
alpha_budget=0.2

per_device_train_batch_size=16
gradient_accumulation_steps=2

# Task naming
task_name="tofu_${model}_${trainer}_reg_lr${lr}_gamma${gamma}_alpha${alpha}_ent${rho_ent}_pop${rho_pop}_bud${alpha_budget}"
prefix="GOOD/fpop_10_sat_5_"
prefix="GOOD/fpop_10_sat_1_"

prefix="SeqPop/beta5_"
prefix="SeqPop/correct_pop_"
prefix="SeqPop/correct_pop_beta1_"
prefix="SeqPop/correct_pop_beta1_update2_"
prefix="SeqPop/correct_pop_beta5_"
prefix="Competing/correct_pop_beta5_"
prefix="Competing/correct_pop_beta5_div_arl_2_"
prefix="CompetingWarmup/correct_pop_beta5_"
prefix="Competing/correct_pop_beta5_norm_loss_"

prefix="NonCompetingGrad/beta5_"
prefix="NonCompeting/beta5_"

prefix="Competing/beta5_"
prefix="NonCompeting/beta5_lr1e-2_"
prefix="NonCompeting/beta5_lr5e-2_"

###################
# 1. UNLEARNING   #
###################
# echo "==========================================="
# echo " Running SelfBalance (Regularized) on TOFU "
# echo " Task: ${prefix}${task_name}"
# echo "==========================================="

# HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 \
#     python src/train.py --config-name=unlearn.yaml \
#     experiment=$experiment \
#     trainer=$trainer \
#     task_name=${prefix}${task_name} \
#     model=$model \
#     model.model_args.pretrained_model_name_or_path=$model_path \
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
#     data.forget.TOFU_para_forget.args.num_train_paraphrases=0 \
#     data.retain.TOFU_para_retain.args.num_train_paraphrases=0 \
#     trainer.method_args.scorer_trainer.cfg.input_dimension=$scorer_in_dim \
#     trainer.method_args.scorer_trainer.optim_cfg.update_every_n_steps=5 \
#     +trainer.method_args.scorer_trainer.optim_cfg.scheduler=linear \
#     trainer.method_args.scorer_trainer.rho_ent=$rho_ent \
#     trainer.method_args.scorer_trainer.rho_pop=$rho_pop \
#     trainer.method_args.scorer_trainer.alpha_budget=$alpha_budget

########################################
# 2. TOFU EVALUATION (UNLEARNED)
########################################
# echo ""
# echo "============================================"
# echo " Running TOFU Evaluation on Unlearned Model "
# echo "============================================"

# HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 \
#     python src/eval.py \
#     experiment=eval/tofu/default.yaml \
#     'eval=[tofu]' \
#     model=$model \
#     task_name=${prefix}${task_name} \
#     model.model_args.pretrained_model_name_or_path=$(pwd)/saves/unlearn/${prefix}${task_name} \
#     +model.model_args.token=$HF_TOKEN \
#     +model.tokenizer_args.token=$HF_TOKEN \
#     ++model.model_args.device_map='auto' \
#     paths.output_dir=$(pwd)/saves/unlearn/${prefix}${task_name}/tofu_evals


########################################
# 3. PARAPHRASE EVALUATION (UNLEARNED)
########################################
echo ""
echo "=================================================="
echo " Running Paraphrase Evaluation on Unlearned Model "
echo "=================================================="

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 \
    python src/eval.py \
    experiment=eval/paraphrase/default.yaml \
    'eval=[paraphrase_tofu]' \
    model=$model \
    task_name=${prefix}${task_name} \
    model.model_args.pretrained_model_name_or_path=$(pwd)/saves/unlearn/${prefix}${task_name} \
    +model.model_args.token=$HF_TOKEN \
    +model.tokenizer_args.token=$HF_TOKEN \
    ++model.model_args.device_map='auto' \
    +eval.paraphrase.metrics.winrate.baseline_path=saves/eval/tofu_para_${model}/repetitiveness/model.jsonl \
    paths.output_dir=$(pwd)/saves/unlearn/${prefix}${task_name}/evals
