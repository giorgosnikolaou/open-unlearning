#!/usr/bin/env bash
set -euo pipefail

PROJDIR="/tmlscratch/nikolaou/open-unlearning"
VENV="${PROJDIR}/envs/.open-unlearning/bin/activate"
SCRIPTS="${PROJDIR}/community/methods/SelfBalance/TOFU/Llama-3.1-8B-Instruct"
SB_SCRIPTS_8B="${PROJDIR}/community/methods/SelfBalance/TOFU/SB_Llama-3.1-8B-Instruct"
SB_SCRIPTS_1B="${PROJDIR}/community/methods/SelfBalance/TOFU/SB_Llama-3.2-1B-Instruct"
ABLATION_DIR="${SB_SCRIPTS_1B}/forget10"

run() {
  local NAME="$1"
  local SCRIPT="$2"
  local ENV="${3:-}"
  local CMD="cd ${PROJDIR} && source ${VENV} && ${ENV} bash ${SCRIPT}"
  python run.py --name "${NAME}" --gpu 1 "${CMD}"
}

run "tofu-eval-8b-eval-baselines" "${SCRIPTS}/eval-baselines.sh"

for split in forget01 forget05 forget10; do
  for method in GradDiff NPO DPO SimNPO JensUn WGA SatImp Scorer; do
    run "tofu-unlearn-8b-${split}-${method}" "${SCRIPTS}/${split}/${method}.sh"
  done
done

# ── SBGradDiffMatched* base sweep — both models, all splits ──
# (NoRetain and Joint variants are ablations on 1B/forget10 only — see below.)
for sb_dir_var in SB_SCRIPTS_1B SB_SCRIPTS_8B; do
  case $sb_dir_var in
    SB_SCRIPTS_1B) sb_dir="${SB_SCRIPTS_1B}"; tag="1b" ;;
    SB_SCRIPTS_8B) sb_dir="${SB_SCRIPTS_8B}"; tag="8b" ;;
  esac
  for split in forget01 forget05 forget10; do
    for method in SBGradDiffMatched SBGradDiffCorrectMatched; do
      run "tofu-unlearn-${tag}-${split}-${method}" "${sb_dir}/${split}/${method}.sh"
    done
  done
done

# ── Ablations: 1B / forget10 only ──

# NoRetain (drops alpha * retain_loss from main objective)
run "tofu-unlearn-1b-forget10-SBGradDiffMatchedNoRetain" \
    "${ABLATION_DIR}/SBGradDiffMatchedNoRetain.sh"

# Joint mode (single-objective: scorer params optimized via main loss)
run "tofu-unlearn-1b-forget10-SBGradDiffMatchedJoint" \
    "${ABLATION_DIR}/SBGradDiffMatchedJoint.sh"
run "tofu-unlearn-1b-forget10-SBGradDiffCorrectMatchedJoint" \
    "${ABLATION_DIR}/SBGradDiffCorrectMatchedJoint.sh"

# Update-frequency sweep on the two base Matched variants
for method in SBGradDiffMatched SBGradDiffCorrectMatched; do
  for u in 1 5 10; do
    run "tofu-unlearn-1b-forget10-${method}-uev${u}" \
        "${ABLATION_DIR}/${method}.sh" \
        "UPDATE_EVERY_N_STEPS=${u}"
  done
done

# Regularizer ablation: 8 = 2^3 combinations per method
run "tofu-unlearn-1b-forget10-SBGradDiffMatched-regs" \
    "${ABLATION_DIR}/SBGradDiffMatched-regularizers.sh"
run "tofu-unlearn-1b-forget10-SBGradDiffCorrectMatched-regs" \
    "${ABLATION_DIR}/SBGradDiffCorrectMatched-regularizers.sh"
