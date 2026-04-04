#!/usr/bin/env bash
set -euo pipefail

PROJDIR="/tmlscratch/nikolaou/open-unlearning"
VENV="${PROJDIR}/envs/.open-unlearning/bin/activate"
SCRIPTS="${PROJDIR}/community/methods/SelfBalance/TOFU/Llama-3.1-8B-Instruct"

run() {
  local NAME="$1"
  local SCRIPT="$2"
  local CMD="cd ${PROJDIR} && source ${VENV} && bash ${SCRIPT}"
  python run.py --name "${NAME}" --gpu 1 "${CMD}"
}

run "tofu-eval-8b-eval-baselines" "${SCRIPTS}/eval-baselines.sh"

for split in forget01 forget05 forget10; do
  for method in GradDiff NPO DPO SimNPO JensUn WGA SatImp Scorer; do
    run "tofu-unlearn-8b-${split}-${method}" "${SCRIPTS}/${split}/${method}.sh"
  done
done
