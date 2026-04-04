#!/bin/bash
# JensUn — TOFU Paraphrase Train & Eval

set -a
source .env
set +a

source /tmlscratch/nikolaou/open-unlearning/envs/.open-unlearning/bin/activate

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

model="Llama-3.2-1B-Instruct"

complete_name="hyperparam/tofu_forget10/GradDiff/trial_47_alpha0.797201_gamma0.116366_lr1.90e-05_retain_loss_typeNLL"

complete_name="hyperparam/tofu_forget10/Scorer/trial_18_alpha0.339217_beta5.41151_gamma0.878545_lr2.49e-05"
complete_name="hyperparam/tofu_forget10/Scorer/trial_34_alpha0.897739_beta7.3725_gamma4.64752_lr2.07e-05"

# complete_name="hyperparam/tofu_forget10/SatImp/trial_45_alpha0.494106_beta11.43318_beta20.169458_gamma0.873297_lr1.98e-05"

########################################
# 2. TOFU EVALUATION (UNLEARNED)
########################################
echo ""
echo "=================================================="
echo " TOFU Eval — ${complete_name}"
echo "=================================================="

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 \
    python src/eval.py \
    experiment=eval/tofu/default.yaml \
    'eval=[tofu]' \
    model=$model \
    task_name=${complete_name} \
    model.model_args.pretrained_model_name_or_path=$(pwd)/${complete_name} \
    +model.model_args.token=$HF_TOKEN \
    +model.tokenizer_args.token=$HF_TOKEN \
    ++model.model_args.device_map='auto' \
    paths.output_dir=$(pwd)/${complete_name}/tofu_evals


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
    model.model_args.pretrained_model_name_or_path=$(pwd)/${complete_name} \
    +model.model_args.token=$HF_TOKEN \
    +model.tokenizer_args.token=$HF_TOKEN \
    ++model.model_args.device_map='auto' \
    +eval.paraphrase.metrics.winrate.baseline_path=saves/eval/ES_Llama-3.2-1B-Instruct_retain90/paraphrase_evals/repetitiveness/model.jsonl \
    paths.output_dir=$(pwd)/${complete_name}/paraphrase_evals
