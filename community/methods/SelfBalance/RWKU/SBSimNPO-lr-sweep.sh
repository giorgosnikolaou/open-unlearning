#!/bin/bash
# RWKU eval (no fluency) for all SBSimNPO bayesian trials.

set -a
source .env
set +a

source /tmlscratch/nikolaou/open-unlearning/envs/.open-unlearning/bin/activate

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

model="Phi-3-mini-4k-instruct"
method="SBSimNPO"
trial_base="/tmlscratch/nikolaou/RWKU/LLaMA-Factory/hyperparam/rwku_bayesian_lr/${method}"

for trial_dir in "${trial_base}"/trial_*; do
    # Skip trials without a completed checkpoint
    [ ! -f "${trial_dir}/config.json" ] && continue

    trial_name=$(basename "$trial_dir")
    task_name="SB_RWKU_lr_sweep_${method}_${trial_name}"

    echo ""
    echo "============================================"
    echo " RWKU Evaluation: ${method} / ${trial_name}"
    echo "============================================"

    HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 \
        python src/eval.py \
        experiment=eval/rwku/default.yaml \
        'eval=[rwku]' \
        model=$model \
        task_name=$task_name \
        model.model_args.pretrained_model_name_or_path=$trial_dir \
        model.model_args.torch_dtype=float16 \
        +model.model_args.token=$HF_TOKEN \
        +model.tokenizer_args.token=$HF_TOKEN \
        ++model.model_args.device_map='auto' \
        '~eval.rwku.metrics.fluency' \
        paths.output_dir=$(pwd)/saves/unlearn/SB_RWKU/lr_sweep2/${method}/${trial_name}
done
