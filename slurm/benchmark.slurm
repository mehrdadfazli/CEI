#!/bin/bash
# One entrypoint for all benchmarks: set BENCHMARK and the benchmark-specific paths, then sbatch.
# Example (from repo root):
#   export BENCHMARK=chair CONFIG_FILE="$PWD/configs/llava_w_CEI.json" CHAIR_DATA="/path/to/coco/val2014"
#   sbatch slurm/benchmark.slurm
# For clusters that need different GPU types per benchmark, use slurm/chair.slurm etc. instead.
#SBATCH --job-name=CEI_BENCH
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=0-10:00:00
#SBATCH --output=slurm_output/bench_%j.out
#SBATCH --error=slurm_output/bench_%j.err

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/slurm"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common.sh"

: "${BENCHMARK:?Set BENCHMARK to chair, amber, mmhal, pope, or mmstar}"
: "${CONFIG_FILE:?Set CONFIG_FILE to a JSON config under configs/}"

# Derive a human-readable run name from the config filename.
# e.g. configs/llava_w_CEI.json -> llava15_w_CEI  (llava_ is mapped to llava15_)
# e.g. configs/llavanext_base.json -> llavanext_base
_cfg_stem="$(basename "${CONFIG_FILE}" .json)"
case "${_cfg_stem}" in
  llava_*) RUN_NAME="${_cfg_stem/llava_/llava15_}" ;;
  *)       RUN_NAME="${_cfg_stem}" ;;
esac
EXP_DIR="${EXP_DIR:-${REPO_DIR}/results/${BENCHMARK}/${RUN_NAME}}"
mkdir -p "${EXP_DIR}"
CFG="${EXP_DIR}/config.json"
if [[ -e "${CFG}" ]] && [[ "$(readlink -f "${CONFIG_FILE}")" == "$(readlink -f "${CFG}")" ]]; then
  echo "[benchmark.slurm] CONFIG_FILE already at ${CFG}; skipping copy"
else
  cp "${CONFIG_FILE}" "${CFG}"
fi

case "${BENCHMARK}" in
  chair)
    : "${CHAIR_DATA:?Set CHAIR_DATA to COCO val2014 image directory}"
    bash "${REPO_DIR}/scripts/run_chair.sh" "${CFG}" "${EXP_DIR}" "${CHAIR_DATA}" "${CACHE_DIR:-}"
    ;;
  amber)
    : "${AMBER_PATH:?Set AMBER_PATH to AMBER dataset root}"
    bash "${REPO_DIR}/scripts/run_amber.sh" "${CFG}" "${EXP_DIR}" "${AMBER_PATH}" "${CACHE_DIR:-}"
    ;;
  mmhal)
    : "${MMHAL_INPUT:?}" "${MMHAL_IMAGES:?}"
    bash "${REPO_DIR}/scripts/run_mmhal.sh" "${CFG}" "${EXP_DIR}" "${MMHAL_INPUT}" "${MMHAL_IMAGES}" "${CACHE_DIR:-}"
    ;;
  pope)
    : "${POPE_QUESTION_DIR:?}" "${POPE_IMAGE_FOLDER:?}"
    bash "${REPO_DIR}/scripts/run_pope.sh" "${CFG}" "${EXP_DIR}" "${POPE_QUESTION_DIR}" "${POPE_IMAGE_FOLDER}" "${CACHE_DIR:-}"
    ;;
  mmstar)
    : "${MMSTAR_DATA_ROOT:?}"
    bash "${REPO_DIR}/scripts/run_mmstar.sh" "${CFG}" "${EXP_DIR}" "${MMSTAR_DATA_ROOT}" "${MMSTAR_HF_CACHE:-}" "${CACHE_DIR:-}"
    ;;
  *)
    echo "Unknown BENCHMARK=${BENCHMARK} (use chair, amber, mmhal, pope, mmstar)" >&2
    exit 1
    ;;
esac
