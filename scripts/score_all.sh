#!/usr/bin/env bash
# score_all.sh — Score all ready CHAIR / MMStar / MMHal results.
#
# Required environment variables:
#   COCO_ANNO_PATH   Directory with COCO annotation JSONs needed by eval/chair.py:
#                      captions_train2014.json  captions_val2014.json
#                      instances_train2014.json instances_val2014.json
#
#   OPENAI_API_KEY   Standard OpenAI key — used by MMHal judge and MMStar vlmeval judge.
#
# Optional:
#   JUDGE_MODEL      GPT model for both MMHal and MMStar (default: gpt-4o)
#   CHAIR_CACHE      Path to pre-built CHAIR evaluator pickle (default: results/chair/chair_evaluator.pkl)
#                    If it already exists it is loaded; if not it is built and saved there.
#   FORCE            Set to 1 to re-score even if a summary already exists
#
# Usage (from repo root):
#   export COCO_ANNO_PATH=/path/to/coco/annotations
#   export OPENAI_API_KEY=sk-...
#   bash scripts/score_all.sh

set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

# ── parameter checks ──────────────────────────────────────────────────────────
: "${COCO_ANNO_PATH:?[score_all] Please export COCO_ANNO_PATH to the dir with COCO annotation JSONs}"
: "${OPENAI_API_KEY:?[score_all] Please export OPENAI_API_KEY for MMHal and MMStar judges}"

JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o}"
FORCE="${FORCE:-0}"

# Shared CHAIR evaluator cache (built once, reused for every run).
# Override with: export CHAIR_CACHE=/path/to/chair.pkl
CHAIR_CACHE="${CHAIR_CACHE:-${REPO_DIR}/results/chair/chair_evaluator.pkl}"

# ── helpers ───────────────────────────────────────────────────────────────────
log()  { echo "[score_all] $*"; }
skip() { echo "[score_all] SKIP $*"; }
ok()   { echo "[score_all] DONE $*"; }

# Count unique integer values for a given field across JSONL lines.
jsonl_unique_count() {
    local file="$1" field="$2"
    python3 - "${file}" "${field}" << 'PY'
import json, sys
seen = set()
for line in open(sys.argv[1], encoding='utf-8'):
    line = line.strip()
    if not line: continue
    try:
        v = json.loads(line).get(sys.argv[2])
        if v is not None: seen.add(int(v))
    except: pass
print(len(seen))
PY
}

json_list_count() {
    python3 - "$1" << 'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1], encoding='utf-8'))
    print(len(d) if isinstance(d, list) else 0)
except: print(0)
PY
}

# ── run names to score ────────────────────────────────────────────────────────
RUN_NAMES=(
    instructblip_base
    instructblip_w_CEI
    llava15_base
    llava15_w_CEI
    llavanext_base
    llavanext_w_CEI
)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CHAIR
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ CHAIR ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
CHAIR_EXPECTED=500

for run in "${RUN_NAMES[@]}"; do
    model_tag="${run%%_*}"  # instructblip | llava15 | llavanext
    run_dir="${REPO_DIR}/results/chair/${run}"
    cap_file="${run_dir}/${model_tag}_chair.jsonl"
    summary="${run_dir}/summary_chair.json"

    if [[ ! -f "${cap_file}" ]]; then
        skip "chair/${run}: output file not found"
        continue
    fi

    n="$(jsonl_unique_count "${cap_file}" "image_id")"
    if (( n < CHAIR_EXPECTED )); then
        skip "chair/${run}: only ${n}/${CHAIR_EXPECTED} samples — not complete"
        continue
    fi

    if [[ -f "${summary}" && "${FORCE}" != "1" ]]; then
        skip "chair/${run}: summary already exists (use FORCE=1 to re-score)"
        continue
    fi

    log "Scoring chair/${run} (${n} samples)..."
    python eval/chair.py \
        --cap_file    "${cap_file}" \
        --caption_key caption_512 \
        --coco_path   "${COCO_ANNO_PATH}" \
        --cache       "${CHAIR_CACHE}" \
        --summary_file "${summary}"
    ok "chair/${run} → ${summary}"
done

# ═══════════════════════════════════════════════════════════════════════════════
# 2. MMStar
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ MMSTAR ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
MMSTAR_EXPECTED=1500

for run in "${RUN_NAMES[@]}"; do
    model_tag="${run%%_*}"
    run_dir="${REPO_DIR}/results/mmstar/${run}"
    cap_file="${run_dir}/${model_tag}_mmstar.jsonl"
    summary="${run_dir}/summary_mmstar.json"

    if [[ ! -f "${cap_file}" ]]; then
        skip "mmstar/${run}: output file not found"
        continue
    fi

    n="$(jsonl_unique_count "${cap_file}" "index")"
    if (( n < MMSTAR_EXPECTED )); then
        skip "mmstar/${run}: only ${n}/${MMSTAR_EXPECTED} samples — not complete"
        continue
    fi

    if [[ -f "${summary}" && "${FORCE}" != "1" ]]; then
        skip "mmstar/${run}: summary already exists (use FORCE=1 to re-score)"
        continue
    fi

    log "Scoring mmstar/${run} (${n} samples, vlmeval judge: ${JUDGE_MODEL})..."
    python eval/mmstar_eval.py \
        --cap_file    "${cap_file}" \
        --scoring     vlmeval \
        --judge-model "${JUDGE_MODEL}" \
        --api-provider openai \
        --judge-sleep 0.3 \
        --summary_file "${summary}"
    ok "mmstar/${run} → ${summary}"
done

# ═══════════════════════════════════════════════════════════════════════════════
# 3. MMHal-Bench
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ MMHAL ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
MMHAL_EXPECTED=96

for run in "${RUN_NAMES[@]}"; do
    model_tag="${run%%_*}"
    run_dir="${REPO_DIR}/results/mmhal/${run}"
    resp_file="${run_dir}/${model_tag}_mmhal.json"
    summary="${run_dir}/summary_mmhal.json"

    if [[ ! -f "${resp_file}" ]]; then
        skip "mmhal/${run}: output file not found"
        continue
    fi

    n="$(json_list_count "${resp_file}")"
    if (( n < MMHAL_EXPECTED )); then
        skip "mmhal/${run}: only ${n}/${MMHAL_EXPECTED} samples — not complete"
        continue
    fi

    if [[ -f "${summary}" && "${FORCE}" != "1" ]]; then
        skip "mmhal/${run}: summary already exists (use FORCE=1 to re-score)"
        continue
    fi

    log "Scoring mmhal/${run} (${n} samples, judge: ${JUDGE_MODEL})..."
    python eval/eval_gpt4.py \
        --response    "${resp_file}" \
        --gpt-model   "${JUDGE_MODEL}" \
        --api-provider openai \
        --request-sleep 1.0 \
        --summary_file "${summary}"
    ok "mmhal/${run} → ${summary}"
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[score_all] All done. Summary files are in results/<bench>/<run>/summary_<bench>.json"
