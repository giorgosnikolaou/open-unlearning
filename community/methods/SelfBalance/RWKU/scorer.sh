#!/bin/bash
# SelfBalancing GradDiff + LearnedScorer — RWKU Train & Eval

set -a
source .env
set +a

source /tmlscratch/nikolaou/open-unlearning/envs/.open-unlearning/bin/activate

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

declare -A HIDDEN_DIM_MAP=(
    ["Llama-2-7b-chat-hf"]=4096
    ["Llama-3.2-1B-Instruct"]=2048
    ["Llama-3.2-3B-Instruct"]=3072
    ["Llama-3.1-8B-Instruct"]=4096
    ["Llama-3-8B-Instruct"]=4096        # RWKU default
    ["Phi-3.5-mini-instruct"]=3072       # RWKU alternate
)

model="Llama-3-8B-Instruct"
model_path="meta-llama/Meta-Llama-3-8B-Instruct"

model="Qwen2.5-7B-Instruct"
model_path="Qwen/Qwen2.5-7B-Instruct"

model="Llama-3.1-8B-Instruct"
model_path="meta-llama/Llama-3.1-8B-Instruct"

model="Phi-3.5-mini-instruct"
model_path="microsoft/Phi-3.5-mini-instruct"

trainer="SBGradDiffLearned"
experiment="unlearn/rwku/default.yaml"
scorer_in_dim=${HIDDEN_DIM_MAP[$model]}

per_device_train_batch_size=4
gradient_accumulation_steps=2

per_device_train_batch_size=2
gradient_accumulation_steps=4

per_device_train_batch_size=8
gradient_accumulation_steps=1

lr=1e-5
gamma=5.0
alpha=0.5

lambda_entropy=1.0
lambda_population=10.0
budget=0.4
lambda_l2=1.0

update_every_n_steps=5

scorer_lr=0.05
scorer_grad_clip=1.0
normalize_hidden=False

scorer_lr=0.005
scorer_grad_clip=1.0
normalize_hidden=False

task_name="rwku_${model}_${trainer}_lr${lr}_gamma${gamma}_alpha${alpha}_bs${per_device_train_batch_size}_accum${gradient_accumulation_steps}"
params="ent${lambda_entropy}_pop${lambda_population}_budget${budget}_l2${lambda_l2}_up${update_every_n_steps}"
prefix="LearnedRWKU/"
prefix="LearnedRWKU/LowerLR_${scorer_lr}_${scorer_grad_clip}_"
prefix="LearnedRWKU/LowerLR_${scorer_lr}_${scorer_grad_clip}_${normalize_hidden}_"
prefix="LearnedRWKU/Phi_beta1_${scorer_lr}_${scorer_grad_clip}_${normalize_hidden}_"

complete_name="${prefix}${params}_${task_name}"

###################
# 1. UNLEARNING
###################
# echo "==========================================="
# echo " LearnedScorer — ${complete_name}"
# echo "==========================================="

# HYDRA_FULL_ERROR=1 \
#     accelerate launch --config_file configs/accelerate/default_config.yaml \
#     src/train.py --config-name=unlearn.yaml \
#     experiment=$experiment \
#     trainer=$trainer \
#     task_name=${complete_name} \
#     model=$model \
#     model.model_args.pretrained_model_name_or_path=${model_path} \
#     +model.model_args.token=$HF_TOKEN \
#     +model.tokenizer_args.token=$HF_TOKEN \
#     trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
#     trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
#     trainer.args.eval_strategy=no \
#     trainer.args.eval_on_start=False \
#     trainer.args.gradient_checkpointing=True \
#     trainer.args.learning_rate=$lr \
#     trainer.method_args.gamma=$gamma \
#     trainer.method_args.alpha=$alpha \
#     trainer.method_args.scorer.cfg.input_dimension=$scorer_in_dim \
#     trainer.method_args.scorer_trainer.optim_cfg.update_every_n_steps=$update_every_n_steps \
#     trainer.method_args.scorer_trainer.optim_cfg.lr=$scorer_lr \
#     trainer.method_args.scorer_trainer.optim_cfg.grad_clip=$scorer_grad_clip \
#     +trainer.method_args.scorer.cfg.normalize_hidden=${normalize_hidden} \
#     +trainer.method_args.scorer_trainer.optim_cfg.scheduler=linear \
#     trainer.method_args.scorer_trainer.lambda_entropy=$lambda_entropy \
#     trainer.method_args.scorer_trainer.lambda_population=$lambda_population \
#     trainer.method_args.scorer_trainer.budget=$budget \
#     trainer.method_args.scorer_trainer.lambda_l2=$lambda_l2

########################################
# 2. RWKU EVALUATION (UNLEARNED)
########################################
echo ""
echo "=================================================="
echo " RWKU Eval — ${complete_name}"
echo "=================================================="

# rwku_ckpt="/tmlscratch/nikolaou/RWKU/LLaMA-Factory/saves/RWKU/batch10/graddiff_full/phi3_mini_instruct"
# complete_name="RWKURepo/GradDiff"

# rwku_ckpt="/tmlscratch/nikolaou/RWKU/LLaMA-Factory/saves/RWKU/batch10/graddiff_scorer_full/phi3_mini_instruct"
# complete_name="RWKURepo/ScorerBeta5"

# rwku_ckpt="/tmlscratch/nikolaou/RWKU/LLaMA-Factory/saves/RWKU/batch10/graddiff_scorer_full_beta1/phi3_mini_instruct"
# complete_name="RWKURepo/ScorerBeta1"

# rwku_ckpt="/tmlscratch/nikolaou/RWKU/LLaMA-Factory/saves/RWKU/batch10/graddiff_scorer_full_up50_beta5/phi3_mini_instruct"
# complete_name="RWKURepo/ScorerBeta5Up50"

rwku_ckpt="/tmlscratch/nikolaou/RWKU/LLaMA-Factory/saves/RWKU/batch10/graddiff_scorer_full_up50_beta1/phi3_mini_instruct"
complete_name="RWKURepo/ScorerBeta1Up50"

rwku_ckpt="/tmlscratch/nikolaou/RWKU/LLaMA-Factory/saves/RWKU/batch10/graddiff_scorer_full_up50/phi3_mini_instruct"
complete_name="RWKURepo/ScorerBeta5Up50HigherLR"

rwku_ckpt="/tmlscratch/nikolaou/RWKU/LLaMA-Factory/saves/RWKU/batch10/graddiff_scorer_full_up50_beta0.5/phi3_mini_instruct"
complete_name="RWKURepo/ScorerBeta0.55Up50"

rwku_ckpt="/tmlscratch/nikolaou/RWKU/LLaMA-Factory/saves/RWKU/batch10/graddiff_scorer_full_up50_beta5_lr5e-6/phi3_mini_instruct"
complete_name="RWKURepo/ScorerBeta5Up50Lr5e-6"

rwku_ckpt="/tmlscratch/nikolaou/RWKU/LLaMA-Factory/saves/RWKU/batch10/graddiff_scorer_full_up50_beta1_lr1e-6/phi3_mini_instruct"
complete_name="RWKURepo/ScorerBeta1Up50Lr1e-6"

rwku_ckpt="/tmlscratch/nikolaou/RWKU/LLaMA-Factory/saves/RWKU/batch10/graddiff_scorer_full_up50_beta5_lr2.5e-6/phi3_mini_instruct"
complete_name="RWKURepo/ScorerBeta5Up50Lr2.5e-6"

rwku_ckpt="/tmlscratch/nikolaou/RWKU/LLaMA-Factory/saves/RWKU/batch10/graddiff_scorer_full_up50_beta1_lr2.5e-6/phi3_mini_instruct"
complete_name="RWKURepo/ScorerBeta1Up50Lr2.5e-6"

rwku_ckpt="/tmlscratch/nikolaou/RWKU/LLaMA-Factory/saves/RWKU/batch10/graddiff_scorer_full_up50_beta2.5_lr2.5e-6/phi3_mini_instruct"
complete_name="RWKURepo/ScorerBeta2.5Up50Lr2.5e-6"

rwku_ckpt="/tmlscratch/nikolaou/RWKU/LLaMA-Factory/saves/RWKU/batch10/graddiff_scorer_wga_full_up50_beta5_lr2.5e-6/phi3_mini_instruct"
complete_name="RWKURepo/ScorerWGABeta2.5Up50Lr2.5e-6"

rwku_ckpt="/tmlscratch/nikolaou/RWKU/LLaMA-Factory/saves/RWKU/batch10/graddiff_scorer_fo_full_up50_beta2.5_lr5e-6/phi3_mini_instruct"
complete_name="RWKURepo/ScorerForgetOnlyBeta2.5Up50Lr5e-6"

rwku_ckpt="/tmlscratch/nikolaou/RWKU/LLaMA-Factory/saves/RWKU/batch10/graddiff_scorer_fo_full_up50_beta2.5_lr2.5e-6/phi3_mini_instruct"
complete_name="RWKURepo/ScorerForgetOnlyBeta2.5Up50Lr2.5e-6"

rwku_ckpt="/tmlscratch/nikolaou/RWKU/LLaMA-Factory/saves/RWKU/batch10/graddiff_scored/up5_lambdaf0_1_bud0_5_lr1e_5"
complete_name="RWKURepo/Scored/up5_lambdaf0_1_bud0_5_lr1e_5"

# rwku_ckpt="/tmlscratch/nikolaou/RWKU/LLaMA-Factory/saves/RWKU/batch10/graddiff_scored/up40_lambdaf0_1_bud0_5_lr1e_5"
# complete_name="RWKURepo/Scored/up40_lambdaf0_1_bud0_5_lr1e_5"

# rwku_ckpt="/tmlscratch/nikolaou/RWKU/LLaMA-Factory/saves/RWKU/batch10/graddiff_scored/up40_lambdaf0_2_bud0_5"
# complete_name="RWKURepo/Scored/up40_lambdaf0_2_bud0_5"

# rwku_ckpt="/tmlscratch/nikolaou/RWKU/LLaMA-Factory/saves/RWKU/batch10/graddiff_scored/up40_lambdaf0_1_bud0_5"
# complete_name="RWKURepo/Scored/up40_lambdaf0_1_bud0_5"

# rwku_ckpt="/tmlscratch/nikolaou/RWKU/LLaMA-Factory/hyperparam/rwku_div10/GradDiffScored/trial_7_lr3_97e-06"
# complete_name="RWKURepo/Scored/trial_7_lr3_97e-06"

# rwku_ckpt="/tmlscratch/nikolaou/RWKU/LLaMA-Factory/saves/RWKU/batch10/graddiff_scored/lr_optimal"
# complete_name="RWKURepo/Scored/lr_optimal"

# rwku_ckpt="/tmlscratch/nikolaou/RWKU/LLaMA-Factory/saves/RWKU/batch10/graddiff_scored/up40_lambdaf0_05_bud0_5_lambdaent2"
# complete_name="RWKURepo/Scored/up40_lambdaf0_05_bud0_5_lambdaent2"

# rwku_ckpt="/tmlscratch/nikolaou/RWKU/LLaMA-Factory/saves/RWKU/batch10/jensun_full_lr8e-7/phi3_mini_instruct"
# complete_name="RWKURepo/Scored/jensun_optimal"

rwku_ckpt="/tmlscratch/nikolaou/RWKU/LLaMA-Factory/hyperparam/rwku_div10/GradDiffScored/trial_5_lr8_93e-06"
complete_name="RWKURepo/Scored/trial_5_lr8_93e-06"

rwku_ckpt="/tmlscratch/nikolaou/RWKU/LLaMA-Factory/saves/RWKU/batch10/graddiff_scored/up40_original_lr"
complete_name="RWKURepo/Scored/up40_original_lr"

rwku_ckpt="/tmlscratch/nikolaou/RWKU/LLaMA-Factory/saves/RWKU/batch10/graddiff_scored/up40_original_lr_no_ste_no_loss_eos"
complete_name="RWKURepo/Scored/up40_original_lr_no_ste_no_loss_eos"

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 \
    python src/eval.py \
    experiment=eval/rwku/default.yaml \
    'eval=[rwku]' \
    model=$model \
    task_name=${complete_name} \
    model.model_args.pretrained_model_name_or_path=${rwku_ckpt} \
    model.model_args.torch_dtype=float16 \
    +model.model_args.token=$HF_TOKEN \
    +model.tokenizer_args.token=$HF_TOKEN \
    ++model.model_args.device_map='auto' \
    '~eval.rwku.metrics.fluency' \
    paths.output_dir=$(pwd)/saves/unlearn/${complete_name}/rwku_eval
    # model.model_args.pretrained_model_name_or_path=$(pwd)/saves/unlearn/${complete_name} \

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
    'eval=[paraphrase_rwku]' \
    model=$model \
    task_name=${complete_name} \
    model.model_args.pretrained_model_name_or_path=${rwku_ckpt} \
    model.model_args.torch_dtype=float16 \
    +model.model_args.token=$HF_TOKEN \
    +model.tokenizer_args.token=$HF_TOKEN \
    ++model.model_args.device_map='auto' \
    '~eval.paraphrase.metrics.winrate' \
    eval.paraphrase.metrics.forget_quality_level1.max_samples=300 \
    eval.paraphrase.metrics.forget_quality_level2.max_samples=300 \
    eval.paraphrase.metrics.retain_quality_level1.max_samples=300 \
    eval.paraphrase.metrics.retain_quality_level2.max_samples=300 \
    '~eval.paraphrase.metrics.forget_quality_level3' \
    eval.paraphrase.metrics.forget_quality_level1.datasets.RWKU_para_forget_level1.args.num_train_paraphrases=0 \
    eval.paraphrase.metrics.forget_quality_level2.datasets.RWKU_para_forget_level2.args.num_train_paraphrases=0 \
    eval.paraphrase.metrics.retain_quality_level1.datasets.RWKU_para_neighbor_level1.args.num_train_paraphrases=9 \
    eval.paraphrase.metrics.retain_quality_level2.datasets.RWKU_para_neighbor_level2.args.num_train_paraphrases=9 \
    paths.output_dir=$(pwd)/saves/unlearn/${complete_name}/paraphrase_eval
    # model.model_args.pretrained_model_name_or_path=$(pwd)/saves/unlearn/${complete_name} \