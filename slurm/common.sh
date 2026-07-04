#!/usr/bin/env bash
# Shared Slurm/local setup. Edit CONDA_SH and CONDA_ENV for your cluster.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

# Default: adjust to your Miniforge/Anaconda install
CONDA_SH="${CONDA_SH:-/path/to/miniforge3/bin/activate}"
CONDA_ENV="${CONDA_ENV:-cei}"

if [[ -f "${CONDA_SH}" ]]; then
  # Conda/MKL activate.d scripts reference unset vars; nounset breaks them.
  set +u
  # shellcheck source=/dev/null
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
  set -u
else
  echo "[common.sh] CONDA_SH not found (${CONDA_SH}); set CONDA_SH and CONDA_ENV" >&2
fi

export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p results slurm_output
