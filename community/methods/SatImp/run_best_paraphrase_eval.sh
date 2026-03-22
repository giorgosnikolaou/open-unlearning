#!/bin/bash
# Paraphrase TOFU Evaluation for best SatImp model (beta1=5.0, beta2=0.5)

set -a
source .env
set +a

source /tmlscratch/nikolaou/open-unlearning/envs/.open-unlearning/bin/activate

model="Llama-3.2-1B-Instruct"
best_run="saves/unlearn/SatImp_sweep/SATIMP_tofu_${model}_forget10_beta15.0_beta20.5"
best_run="saves/unlearn/SatImp_Sat_sweep/SATIMP_tofu_${model}_forget10_beta15.0_beta20.5"

echo ""
echo "=========================================================="
echo " Paraphrase TOFU Eval: best SatImp (beta1=5.0, beta2=0.5) "
echo "=========================================================="

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 \
    python src/eval.py \
    experiment=eval/paraphrase/default.yaml \
    'eval=[paraphrase_tofu]' \
    model=$model \
    task_name=SATIMP_best_beta15.0_beta20.5 \
    model.model_args.pretrained_model_name_or_path=$(pwd)/${best_run} \
    +model.model_args.token=$HF_TOKEN \
    +model.tokenizer_args.token=$HF_TOKEN \
    ++model.model_args.device_map='auto' \
    +eval.paraphrase.metrics.winrate.baseline_path=saves/eval/tofu_para_${model}/repetitiveness/model.jsonl \
    paths.output_dir=$(pwd)/${best_run}/paraphrase_evals