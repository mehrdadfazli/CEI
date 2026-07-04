#!/usr/bin/env bash
set -euo pipefail
# Usage: ./scripts/run_chair.sh <config.json> <log_dir> <chair_data_path> [cache_dir]

CONFIG_FILE="${1:?config.json}"
LOG_DIR="${2:?log_dir}"
CHAIR_DATA="${3:?path to COCO val2014 directory}"
CACHE_DIR="${4:-}"

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

model_type=$(read_config model_type)
cache_cfg=$(read_config cache_dir)
cache_dir="${CACHE_DIR:-$cache_cfg}"
load_in_8bit=$(read_config load_in_8bit)
use_CEI=$(read_config use_CEI)
do_sample=$(read_config do_sample)
num_beams=$(read_config num_beams)
max_new_tokens=$(read_config max_new_tokens)
context_embedding_idx=$(read_config context_embedding_idx)
context_embedding_layer=$(read_config context_embedding_layer)
injection_layer=$(read_config injection_layer)
alpha=$(read_config alpha)
dynamic_mode=$(read_config dynamic_mode)
alpha_method=$(read_config alpha_method)
K_mass=$(read_config K_mass)
start_layer=$(read_config start_layer)
topK_mass_start_layer=$(read_config topK_mass_start_layer)
tau=$(read_config tau)
Tval=$(read_config T)
beta=$(read_config beta)
num_images=$(read_config num_images)
random_seed=$(read_config random_seed)
opera_results=$(read_config opera_results)
delta=$(read_config delta)
gamma=$(read_config gamma)
repetition_penalty=$(read_config repetition_penalty)
KV_cache=$(read_config KV_cache)

cmd="python run_CHAIR.py --config \"$CONFIG_FILE\" --chair_data_path \"$CHAIR_DATA\" --log_dir \"$LOG_DIR\""
[[ -n "$cache_dir" ]] && cmd="$cmd --cache_dir \"$cache_dir\""
case "$load_in_8bit" in true|True) cmd="$cmd --load_in_8bit" ;; esac
case "$use_CEI" in false|False) cmd="$cmd --no_cei" ;; esac
case "$do_sample" in true|True) cmd="$cmd --do_sample" ;; esac
case "$opera_results" in true|True) cmd="$cmd --opera_results" ;; esac
case "$KV_cache" in true|True) cmd="$cmd --KV_cache" ;; esac
cmd="$cmd --num_beams ${num_beams:-1} --max_new_tokens ${max_new_tokens:-512}"
cmd="$cmd --context_embedding_idx ${context_embedding_idx:--1}"
cmd="$cmd --context_embedding_layer ${context_embedding_layer:--1}"
cmd="$cmd --injection_layer ${injection_layer:-10} --alpha ${alpha:-0.1}"
cmd="$cmd --dynamic_mode ${dynamic_mode:-two_pass} --alpha_method ${alpha_method:-sigmoid}"
cmd="$cmd --K_mass ${K_mass:-40} --start_layer ${start_layer:-1} --tau ${tau:-0.2} --T ${Tval:-0.05}"
cmd="$cmd --beta ${beta:-0.3}"
cmd="$cmd --delta ${delta:-0.5} --gamma ${gamma:-0.0} --repetition_penalty ${repetition_penalty:-1.1}"
cmd="$cmd --num_images ${num_images:-500} --random_seed ${random_seed:-42}"
if [[ -n "$topK_mass_start_layer" && "$topK_mass_start_layer" != "-1" ]]; then
  cmd="$cmd --topK_mass_start_layer $topK_mass_start_layer"
fi
echo "$cmd"
eval "$cmd"
