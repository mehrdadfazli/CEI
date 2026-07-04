#!/bin/bash
#SBATCH --job-name=CEI_POPE
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-8:00:00
#SBATCH --output=slurm_output/pope_%j.out
#SBATCH --error=slurm_output/pope_%j.err

set -euo pipefail
# Slurm runs a copy under /var/spool/slurmd/... so BASH_SOURCE is not the repo path.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/slurm"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common.sh"

: "${CONFIG_FILE:?Set CONFIG_FILE to configs/llava_w_CEI.json etc.}"
: "${POPE_QUESTION_DIR:?Set POPE_QUESTION_DIR to POPE questions directory}"
: "${POPE_IMAGE_FOLDER:?Set POPE_IMAGE_FOLDER to COCO val2014 images}"

_cfg_stem="$(basename "${CONFIG_FILE}" .json)"
case "${_cfg_stem}" in
  llava_*) RUN_NAME="${_cfg_stem/llava_/llava15_}" ;;
  *)       RUN_NAME="${_cfg_stem}" ;;
esac
EXP_DIR="${EXP_DIR:-${REPO_DIR}/results/pope/${RUN_NAME}}"
mkdir -p "${EXP_DIR}"
CFG="${EXP_DIR}/config.json"
if [[ -e "${CFG}" ]] && [[ "$(readlink -f "${CONFIG_FILE}")" == "$(readlink -f "${CFG}")" ]]; then
  echo "[pope.slurm] CONFIG_FILE already at ${CFG}; skipping copy"
else
  cp "${CONFIG_FILE}" "${CFG}"
fi

bash "${REPO_DIR}/scripts/run_pope.sh" "${CFG}" "${EXP_DIR}" "${POPE_QUESTION_DIR}" "${POPE_IMAGE_FOLDER}" "${CACHE_DIR:-}"
