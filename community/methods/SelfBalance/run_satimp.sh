#!/bin/bash

set -a
source .env
set +a

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

CUDA_VISIBLE_DEVICES=0 python src/train.py \
  --config-name=unlearn.yaml \
  experiment=unlearn/tofu/default.yaml \
  trainer=SatImp \
  task_name=SATIMP_tofu_Llama-3.2-1B-Instruct_forget10_SatImp_lr1e-5_beta15.0_beta21.0_alpha0.1 \
  model=Llama-3.2-1B-Instruct \
  forget_split=forget10 \
  retain_split=retain90 \
  model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
  +model.model_args.device_map='auto' \
  retain_logs_path=saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json \
  trainer.args.per_device_train_batch_size=16 \
  trainer.args.gradient_accumulation_steps=2 \
  trainer.args.eval_strategy=no \
  trainer.args.eval_on_start=False \
  trainer.args.learning_rate=1e-5 \
  trainer.method_args.beta1=1.0 \
  trainer.method_args.beta2=1.0 \
  trainer.method_args.gamma=5.0 \
  trainer.method_args.alpha=0.1


# trainer="SatImp"
# baseline_trainer="GradDiff"
# model="Llama-3.2-1B-Instruct"
# model="Llama-2-7b-chat-hf"

# model_path="open-unlearning/tofu_${model}_full"
# forget_split="forget10"
# retain_split="retain90"

# experiment="unlearn/tofu/default.yaml"

# lr=1e-5

# per_device_train_batch_size=16
# gradient_accumulation_steps=2

# gamma=10.0
# # gamma=0.1
# # alpha=0.1
# alpha=1.0

# beta1=5.0
# beta2=1.0

# task_name="tofu_${model}_${forget_split}_${trainer}_lr${lr}_gamma${gamma}_alpha${alpha}"

# CUDA_VISIBLE_DEVICES=0 \
#     python src/train.py --config-name=unlearn.yaml \
#     experiment=$experiment \
#     trainer=$trainer \
#     task_name=SATIMP_${task_name} \
#     model=$model \
#     forget_split=$forget_split \
#     retain_split=$retain_split \
#     model.model_args.pretrained_model_name_or_path=$model_path \
#     +model.model_args.token=$HF_TOKEN \
#     +model.tokenizer_args.token=$HF_TOKEN \
#     +model.model_args.device_map='auto' \
#     retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
#     trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
#     trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
#     trainer.args.eval_strategy=no \
#     trainer.args.eval_on_start=False \
#     trainer.args.learning_rate=$lr \
#     trainer.method_args.beta1=$beta1 \
#     trainer.method_args.beta2=$beta2 \
#     trainer.method_args.gamma=$gamma \
#     trainer.method_args.alpha=$alpha


# export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
# echo "Master Port: $MASTER_PORT"

########################################################################################################################
########################################### Unlearn TOFU models ########################################################
########################################################################################################################

# models=(
#     "Llama-3.2-1B-Instruct"
# )

# trainers_experiments=(
#     "SatImp unlearn/tofu/default.yaml"
# )

# forget_retain_splits=(
#     "forget10 retain90"
# )

# per_device_train_batch_size=16
# gradient_accumulation_steps=2

# lrs=(1e-5)
# alphas=(1.0 0.1 0.01)
# betas=(5.0 6.0)
# beta2=1.0

# for split in "${forget_retain_splits[@]}"; do
#     forget_split=$(echo $split | cut -d' ' -f1)
#     retain_split=$(echo $split | cut -d' ' -f2)

#     for model in "${models[@]}"; do
#         for trainer_experiment in "${trainers_experiments[@]}"; do
#             trainer=$(echo $trainer_experiment | cut -d' ' -f1)
#             experiment=$(echo $trainer_experiment | cut -d' ' -f2)

#             for lr in "${lrs[@]}"; do
#                 for beta1 in "${betas[@]}"; do
#                     for alpha in "${alphas[@]}"; do

#                         task_name=tofu_${model}_${forget_split}_${trainer}_lr${lr}_beta1${beta1}_beta2${beta2}_alpha${alpha}
#                         model_path=open-unlearning/tofu_${model}_full

#                         echo
#                         echo "### ${task_name}"
#                         echo "# Unlearning ${model_path} using ${trainer}"

#                         echo "CUDA_VISIBLE_DEVICES=0 python src/train.py \\"
#                         echo "  --config-name=unlearn.yaml \\"
#                         echo "  experiment=${experiment} \\"
#                         echo "  trainer=${trainer} \\"
#                         echo "  task_name=${task_name} \\"
#                         echo "  model=${model} \\"
#                         echo "  forget_split=${forget_split} \\"
#                         echo "  retain_split=${retain_split} \\"
#                         echo "  model.model_args.pretrained_model_name_or_path=${model_path} \\"
#                         echo "  retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \\"
#                         echo "  trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \\"
#                         echo "  trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \\"
#                         echo "  trainer.args.eval_strategy=no \\"
#                         echo "  trainer.args.eval_on_start=False \\"
#                         echo "  trainer.args.learning_rate=${lr} \\"
#                         echo "  trainer.method_args.beta1=${beta1} \\"
#                         echo "  trainer.method_args.beta2=${beta2} \\"
#                         echo "  trainer.method_args.alpha=${alpha}"

#                         echo
#                         echo "# Eval"
#                         echo "CUDA_VISIBLE_DEVICES=0 python src/eval.py \\"
#                         echo "  experiment=eval/tofu/default.yaml \\"
#                         echo "  forget_split=${forget_split} \\"
#                         echo "  model=${model} \\"
#                         echo "  task_name=${task_name} \\"
#                         echo "  model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \\"
#                         echo "  paths.output_dir=saves/unlearn/${task_name}/evals \\"
#                         echo "  retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json"

#                     done
#                 done
#             done
#         done
#     done
# done
