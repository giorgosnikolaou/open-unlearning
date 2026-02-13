#!/bin/bash
# SelfBalance Unlearning + LKF Evaluation Script

set -a
source .env
set +a

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

# Configuration
baseline_trainer="GradDiff"
trainer="SelfBalancing$baseline_trainer"
model="Llama-3.2-1B-Instruct"
model="Llama-3.2-3B-Instruct"

# Model and data paths
# model_path="open-unlearning/lkf_${model}_pretrained"  # Adjust to your pretrained model path
model_path=meta-llama/${model}
forget_dataset="LKF_forget"
retain_dataset="LKF_retain"

# Experiment configuration
experiment="unlearn/lkf/default.yaml"  # You may need to create this

# Hyperparameters
lr=1e-5
gamma=2.0
alpha=3.0

per_device_train_batch_size=8
gradient_accumulation_steps=2

# Task naming
# task_name="lkf_${model}_${trainer}_lr${lr}_gamma${gamma}_alpha${alpha}"
# prefix="SB_"

# task_name="tofu_${model}_${forget_split}_${trainer}_lr${lr}_gamma${gamma}_alpha${alpha}"
task_name="tofu_${model}_forget10_${trainer}_lr${lr}_gamma${gamma}_alpha${alpha}"
prefix="TOP_"

echo "========================================="
echo "Running SelfBalance Unlearning on LKF"
echo "Task: ${prefix}${task_name}"
echo "========================================="

###################
# 1. UNLEARNING   #
###################
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
#     trainer.method_args.scorer_trainer.optim_cfg.update_every_n_steps=5 \
#     +trainer.method_args.scorer_trainer.optim_cfg.scheduler=linear

# echo ""
# echo "========================================="
# echo "Unlearning Complete!"
# echo "Model saved to: saves/unlearn/${prefix}${task_name}"
# echo "========================================="

##############################
# 2. LKF EVALUATION (UNLEARNED)
##############################
# echo ""
# echo "========================================="
# echo "Running LKF Evaluation on Unlearned Model"
# echo "========================================="

# # Set GOOGLE_API_KEY for Gemini judge
# if [ -z "$GOOGLE_API_KEY" ]; then
#     echo "WARNING: GOOGLE_API_KEY not set. LKF judge evaluation will fail."
#     echo "Set it with: export GOOGLE_API_KEY='your-key-here'"
# fi

# # Evaluate unlearned model with LKF benchmark
# HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 \
#     python src/eval.py \
#     experiment=eval/lkf/default.yaml \
#     model=$model \
#     task_name=${prefix}${task_name} \
#     model.model_args.pretrained_model_name_or_path=saves/unlearn/${prefix}${task_name} \
#     +model.tokenizer_args.padding_side=left \
#     +eval.lkf.metrics.repetitiveness.baseline_model_path=meta-llama/$model \
#     +eval.lkf.metrics.repetitiveness.task_name=unlearned \
#     +eval.lkf.metrics.winrate.task_name=unlearned \
#     paths.output_dir=saves/unlearn/${prefix}${task_name}/evals

###############################
# 3. LKF EVALUATION (PRETRAINED)
###############################
echo ""
echo "========================================="
echo "Running LKF Evaluation on Pretrained Model"
echo "========================================="

# Set GOOGLE_API_KEY for Gemini judge
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "WARNING: GOOGLE_API_KEY not set. LKF judge evaluation will fail."
    echo "Set it with: export GOOGLE_API_KEY='your-key-here'"
fi

# Evaluate pretrained HF model with LKF benchmark (no winrate metric)
# HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 \
#     python src/eval.py \
#     experiment=eval/lkf/default.yaml \
#     model=$model \
#     task_name=pretrained_${model} \
#     model.model_args.pretrained_model_name_or_path=$model_path \
#     +model.tokenizer_args.padding_side=left \
#     +eval.lkf.metrics.repetitiveness.task_name=pretrained \
#     ~eval.lkf.metrics.winrate \
#     paths.output_dir=saves/eval/lkf_${model}
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 \
    python src/eval.py \
    experiment=eval/lkf/default.yaml \
    model=$model \
    task_name=pretrained_${model} \
    model.model_args.pretrained_model_name_or_path=$model_path \
    +model.tokenizer_args.padding_side=left \
    paths.output_dir=saves/eval/lkf_${model}

# echo ""
# echo "========================================="
# echo "LKF Evaluation Complete!"
# echo "Results saved to: saves/eval/lkf_${model}"
# echo "========================================="
# echo ""
# echo "Summary (all results in LKF_SUMMARY.json):"
# echo "  - Forget Quality: J_W, J_P, J_ICR metrics"
# echo "  - Retain Quality: J_avg metric"
# echo "  - Repetitiveness: entropy score"
# echo "  - MMLU: accuracy"
# echo ""
# echo "Note: Pretrained model evaluation does NOT include win rate metric."
# echo "      Win rate requires comparing against a baseline (unlearned vs pretrained)."
# echo ""