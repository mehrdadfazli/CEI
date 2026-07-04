#!/bin/bash
#SBATCH --job-name=CEI_AMBER
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=0-12:00:00
#SBATCH --output=slurm_output/amber_%j.out
#SBATCH --error=slurm_output/amber_%j.err

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
: "${AMBER_PATH:?Set AMBER_PATH to AMBER dataset root}"

_cfg_stem="$(basename "${CONFIG_FILE}" .json)"
case "${_cfg_stem}" in
  llava_*) RUN_NAME="${_cfg_stem/llava_/llava15_}" ;;
  *)       RUN_NAME="${_cfg_stem}" ;;
esac
EXP_DIR="${EXP_DIR:-${REPO_DIR}/results/amber/${RUN_NAME}}"
mkdir -p "${EXP_DIR}"
CFG="${EXP_DIR}/config.json"
if [[ -e "${CFG}" ]] && [[ "$(readlink -f "${CONFIG_FILE}")" == "$(readlink -f "${CFG}")" ]]; then
  echo "[amber.slurm] CONFIG_FILE already at ${CFG}; skipping copy"
else
  cp "${CONFIG_FILE}" "${CFG}"
fi

bash "${REPO_DIR}/scripts/run_amber.sh" "${CFG}" "${EXP_DIR}" "${AMBER_PATH}" "${CACHE_DIR:-}"
