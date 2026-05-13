#!/bin/bash
# Scorer (original SelfBalancingGradDiff) — Ablation: load a previously-trained
# scorer, freeze it, unlearn the model from scratch under fixed weighting.
# Model: Llama-3.2-1B-Instruct | Split: forget10/retain90

set -a
source .env
set +a

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

model="Llama-3.2-1B-Instruct"
forget_split="forget10"
retain_split="retain90"
holdout_split="holdout10"
num_epochs=10
GPU=${GPU:-0}

SB_SUMMARY_JSON="${SB_SUMMARY_JSON:-hyperparam/tofu_forget10/sb_bayesian_summary.json}"

get_param() {
    python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d[sys.argv[2]]['best_params'][sys.argv[3]])" \
        "$SB_SUMMARY_JSON" "$1" "$2"
}

HP_KEY="${HP_KEY:-SBGradDiffMatched}"
SCORER_lr=${SCORER_lr:-$(get_param "$HP_KEY" lr)}
SCORER_gamma=${SCORER_gamma:-$(get_param "$HP_KEY" gamma)}
SCORER_alpha=${SCORER_alpha:-$(get_param "$HP_KEY" alpha)}
SCORER_beta=${SCORER_beta:-$(get_param "$HP_KEY" beta)}

PRETRAINED_SCORER="${PRETRAINED_SCORER:-$(pwd)/saves/unlearn/SB_TOFU/${model}/forget10/Scorer/scorer.pt}"

task_name="SB_TOFU/${model}/forget10/ScorerAblations/frozen-trained"
model_output="saves/unlearn/${task_name}"

# ══════════════════════════════════════════
#  1. Unlearn
# ══════════════════════════════════════════
echo ""
echo "============================================"
echo " Scorer Ablation: frozen-trained — Unlearn (forget10)"
echo " Loading scorer from: ${PRETRAINED_SCORER}"
echo "============================================"

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=$GPU \
    python src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default \
    trainer=SBGradDiffMatchedLearned \
    model=$model \
    task_name=$task_name \
    model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_${model}_full \
    +model.model_args.token=$HF_TOKEN \
    +model.tokenizer_args.token=$HF_TOKEN \
    ++model.model_args.device_map='auto' \
    forget_split=$forget_split \
    retain_split=$retain_split \
    trainer.args.num_train_epochs=$num_epochs \
    trainer.args.eval_on_start=false \
    trainer.args.eval_strategy=no \
    trainer.args.gradient_checkpointing=True \
    trainer.args.learning_rate=$SCORER_lr \
    trainer.method_args.gamma=$SCORER_gamma \
    trainer.method_args.alpha=$SCORER_alpha \
    trainer.method_args.beta=$SCORER_beta \
    trainer.method_args.scorer.cfg.input_dimension=2048 \
    trainer.method_args.scorer.pretrained_path=$PRETRAINED_SCORER \
    trainer.method_args.scorer_pretrain.freeze_after=true

# ══════════════════════════════════════════
#  2. Paraphrase Generation (GPU)
# ══════════════════════════════════════════
echo ""
echo "============================================"
echo " frozen-trained — Paraphrase Generation (forget10)"
echo "============================================"

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=$GPU \
    python src/eval.py \
    experiment=eval/paraphrase/default.yaml \
    'eval=[paraphrase_tofu]' \
    model=$model \
    task_name=$task_name \
    model.model_args.pretrained_model_name_or_path=$(pwd)/${model_output} \
    +model.model_args.token=$HF_TOKEN \
    +model.tokenizer_args.token=$HF_TOKEN \
    ++model.model_args.device_map='auto' \
    eval.paraphrase.forget_split=$forget_split \
    eval.paraphrase.retain_split=$retain_split \
    eval.paraphrase.holdout_split=$holdout_split \
    eval.paraphrase.metrics.forget_quality.max_samples=400 \
    eval.paraphrase.metrics.retain_quality.max_samples=400 \
    eval.paraphrase.metrics.forget_quality.datasets.TOFU_para_forget_eval.args.num_train_paraphrases=10 \
    eval.paraphrase.metrics.retain_quality.datasets.TOFU_para_retain_eval.args.num_train_paraphrases=10 \
    '++eval.paraphrase.metrics.repetitiveness.generation={max_new_tokens:128,do_sample:false,temperature:0.0}' \
    +eval.paraphrase.metrics.forget_quality.generation_only=true \
    +eval.paraphrase.metrics.retain_quality.generation_only=true \
    '~eval.paraphrase.metrics.winrate' \
    paths.output_dir=$(pwd)/${model_output}/paraphrase_evals

# ══════════════════════════════════════════
#  3. Paraphrase Judging (API, no GPU)
# ══════════════════════════════════════════
winrate_baseline="saves/eval/SB_TOFU/${model}/baselines/${retain_split}_${forget_split}/paraphrase_evals/repetitiveness/model.jsonl"

echo ""
echo "============================================"
echo " frozen-trained — Paraphrase Judging (forget10)"
echo "============================================"

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES="" \
    python src/eval.py \
    experiment=eval/paraphrase/default.yaml \
    'eval=[paraphrase_tofu]' \
    model=$model \
    task_name=$task_name \
    model.model_args.pretrained_model_name_or_path=$(pwd)/${model_output} \
    +model.model_args.token=$HF_TOKEN \
    +model.tokenizer_args.token=$HF_TOKEN \
    ++model.model_args.device_map='auto' \
    eval.paraphrase.forget_split=$forget_split \
    eval.paraphrase.retain_split=$retain_split \
    eval.paraphrase.holdout_split=$holdout_split \
    eval.paraphrase.metrics.forget_quality.max_samples=400 \
    eval.paraphrase.metrics.retain_quality.max_samples=400 \
    eval.paraphrase.metrics.forget_quality.datasets.TOFU_para_forget_eval.args.num_train_paraphrases=10 \
    eval.paraphrase.metrics.retain_quality.datasets.TOFU_para_retain_eval.args.num_train_paraphrases=10 \
    +eval.paraphrase.metrics.winrate.baseline_path=${winrate_baseline} \
    '~eval.paraphrase.metrics.mmlu' \
    +judge_only=true \
    paths.output_dir=$(pwd)/${model_output}/paraphrase_evals