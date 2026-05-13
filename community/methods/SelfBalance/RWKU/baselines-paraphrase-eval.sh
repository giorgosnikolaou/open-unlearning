#!/bin/bash
# Paraphrase judging for pretrained Phi-3-mini-4k-instruct baseline.

set -a
source .env
set +a

source /tmlscratch/nikolaou/open-unlearning/envs/.open-unlearning/bin/activate

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

# Configuration
model="Phi-3-mini-4k-instruct"
model_path="microsoft/Phi-3-mini-4k-instruct"
task_name="RWKU_baseline_${model}"

########################################
# PARAPHRASE EVALUATION (CPU)
########################################
echo ""
echo "============================================"
echo " Paraphrase Evaluation: baseline ${model}"
echo "============================================"

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES="" \
    python src/eval.py \
    experiment=eval/paraphrase/default.yaml \
    'eval=[paraphrase_rwku]' \
    model=$model \
    task_name=$task_name \
    model.model_args.pretrained_model_name_or_path=$model_path \
    model.model_args.torch_dtype=float16 \
    +model.model_args.token=$HF_TOKEN \
    +model.tokenizer_args.token=$HF_TOKEN \
    ++model.model_args.device_map='auto' \
    eval.paraphrase.metrics.forget_quality_level1.max_samples=300 \
    eval.paraphrase.metrics.forget_quality_level2.max_samples=300 \
    eval.paraphrase.metrics.forget_quality_level3.max_samples=400 \
    eval.paraphrase.metrics.retain_quality_level1.max_samples=300 \
    eval.paraphrase.metrics.retain_quality_level2.max_samples=300 \
    eval.paraphrase.metrics.forget_quality_level1.datasets.RWKU_para_forget_level1.args.num_train_paraphrases=0 \
    eval.paraphrase.metrics.forget_quality_level2.datasets.RWKU_para_forget_level2.args.num_train_paraphrases=0 \
    eval.paraphrase.metrics.forget_quality_level3.datasets.RWKU_para_forget_level3.args.num_train_paraphrases=0 \
    eval.paraphrase.metrics.retain_quality_level1.datasets.RWKU_para_neighbor_level1.args.num_train_paraphrases=9 \
    eval.paraphrase.metrics.retain_quality_level2.datasets.RWKU_para_neighbor_level2.args.num_train_paraphrases=9 \
    +judge_only=true \
    '~eval.paraphrase.metrics.winrate' \
    paths.output_dir=$(pwd)/saves/eval/${task_name}/paraphrase_evals