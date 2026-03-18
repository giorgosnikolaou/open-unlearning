#!/bin/bash
# SatImp Beta1/Beta2 Sweep with TOFU Evaluation

set -a
source .env
set +a

source /tmlscratch/nikolaou/open-unlearning/envs/.open-unlearning/bin/activate

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

# Configuration
trainer="SatImp"
model="Llama-3.2-1B-Instruct"
model_path="open-unlearning/tofu_${model}_full"
experiment="unlearn/tofu/default.yaml"
forget_split="forget10"
retain_split="retain90"

# Fixed hyperparameters
lr=1e-5
gamma=1.0
alpha=0.1
per_device_train_batch_size=16
gradient_accumulation_steps=2

# Output directory for all SatImp sweep runs
sweep_dir="saves/unlearn/SatImp_sweep"

# Sweep grid
beta1_values=(1.0 5.0 10.0)
beta2_values=(0.5 1.0 2.0)

for beta1 in "${beta1_values[@]}"; do
for beta2 in "${beta2_values[@]}"; do

    task_name="SATIMP_tofu_${model}_${forget_split}_beta1${beta1}_beta2${beta2}"

    echo ""
    echo "================================================"
    echo " SatImp: beta1=${beta1}, beta2=${beta2}"
    echo " Task: ${task_name}"
    echo "================================================"

    run_dir="${sweep_dir}/${task_name}"

    ###################
    # 1. UNLEARNING
    ###################
    if [ -f "${run_dir}/config.json" ]; then
        echo " [SKIP] Unlearning already completed."
    else
        HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 \
            python src/train.py --config-name=unlearn.yaml \
            experiment=$experiment \
            trainer=$trainer \
            task_name=$task_name \
            model=$model \
            forget_split=$forget_split \
            retain_split=$retain_split \
            model.model_args.pretrained_model_name_or_path=$model_path \
            +model.model_args.token=$HF_TOKEN \
            +model.tokenizer_args.token=$HF_TOKEN \
            +model.model_args.device_map='auto' \
            trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
            trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
            trainer.args.eval_strategy=no \
            trainer.args.eval_on_start=False \
            trainer.args.learning_rate=$lr \
            trainer.method_args.beta1=$beta1 \
            trainer.method_args.beta2=$beta2 \
            trainer.method_args.gamma=$gamma \
            trainer.method_args.alpha=$alpha \
            paths.output_dir=$(pwd)/${run_dir}
    fi

    # ###################
    # # 2. TOFU EVAL
    # ###################
    # if [ -f "${run_dir}/tofu_evals/TOFU_SUMMARY.json" ]; then
    #     echo " [SKIP] TOFU evaluation already completed."
    # else
    #     echo ""
    #     echo " Running TOFU Evaluation: ${task_name}"
    #     echo "-------------------------------------------"

    #     HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 \
    #         python src/eval.py \
    #         experiment=eval/tofu/default.yaml \
    #         'eval=[tofu]' \
    #         model=$model \
    #         forget_split=$forget_split \
    #         task_name=$task_name \
    #         model.model_args.pretrained_model_name_or_path=$(pwd)/${run_dir} \
    #         +model.model_args.token=$HF_TOKEN \
    #         +model.tokenizer_args.token=$HF_TOKEN \
    #         ++model.model_args.device_map='auto' \
    #         retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
    #         paths.output_dir=$(pwd)/${run_dir}/tofu_evals
    # fi

done
done

echo ""
echo "================================================"
echo " Sweep complete!"
echo "================================================"
