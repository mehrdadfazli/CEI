#!/usr/bin/env bash
set -euo pipefail
# Usage: ./scripts/run_mmstar.sh <config.json> <log_dir> <mmstar_data_root> [mmstar_hf_cache] [cache_dir]
# mmstar_data_root should contain MMStar/images/*.png

CONFIG_FILE="${1:?config.json}"
LOG_DIR="${2:?log_dir}"
MMSTAR_ROOT="${3:?parent of MMStar/images}"
HF_CACHE="${4:-}"
CACHE_DIR="${5:-}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

read_config() {
  python - "$CONFIG_FILE" "$1" << 'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
v = cfg.get(sys.argv[2], "")
print("" if v is None else v)
PY
}

gpu_id=$(read_config gpu_id)
cache_cfg=$(read_config cache_dir)
cache_dir="${CACHE_DIR:-$cache_cfg}"
hf_cache="${HF_CACHE:-}"
load_in_8bit=$(read_config load_in_8bit)
do_sample=$(read_config do_sample)
max_new_tokens=$(read_config max_new_tokens)
repetition_penalty=$(read_config repetition_penalty)
KV_cache=$(read_config KV_cache)
injection_layer=$(read_config injection_layer)
context_embedding_idx=$(read_config context_embedding_idx)
context_embedding_layer=$(read_config context_embedding_layer)
K_mass=$(read_config K_mass)
start_layer=$(read_config start_layer)
alpha_method=$(read_config alpha_method)
alpha=$(read_config alpha)
tau=$(read_config tau)
Tval=$(read_config T)
beta=$(read_config beta)
topK_mass_start_layer=$(read_config topK_mass_start_layer)
delta=$(read_config delta)
gamma=$(read_config gamma)

cmd="python run_MMStar.py --config \"$CONFIG_FILE\" --mmstar_data_root \"$MMSTAR_ROOT\" --log_dir \"$LOG_DIR\""
cmd="$cmd --gpu_id ${gpu_id:-0} --max_new_tokens 32"
cmd="$cmd --repetition_penalty ${repetition_penalty:-1.1}"
cmd="$cmd --injection_layer ${injection_layer:-10}"
cmd="$cmd --context_embedding_idx ${context_embedding_idx:--1}"
cmd="$cmd --context_embedding_layer ${context_embedding_layer:--1}"
cmd="$cmd --K_mass ${K_mass:-40} --start_layer ${start_layer:-1}"
cmd="$cmd --alpha_method ${alpha_method:-sigmoid} --alpha ${alpha:-0.1}"
cmd="$cmd --tau ${tau:-0.2} --T ${Tval:-0.05} --beta ${beta:-0.3}"
cmd="$cmd --delta ${delta:-0.3} --gamma ${gamma:-0.2}"
[[ -n "$cache_dir" ]] && cmd="$cmd --cache_dir \"$cache_dir\""
[[ -n "$hf_cache" ]] && cmd="$cmd --mmstar_hf_cache \"$hf_cache\""
if [[ -n "$topK_mass_start_layer" && "$topK_mass_start_layer" != "-1" ]]; then
  cmd="$cmd --topK_mass_start_layer $topK_mass_start_layer"
fi
case "$load_in_8bit" in true|True) cmd="$cmd --load_in_8bit" ;; esac
case "$do_sample" in true|True) cmd="$cmd --do_sample" ;; esac
case "$KV_cache" in true|True) cmd="$cmd --KV_cache" ;; esac
echo "$cmd"
eval "$cmd"
