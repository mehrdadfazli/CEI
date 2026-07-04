# CEI runbook: Slurm jobs and scoring

Run everything from the **repository root** (`cd` to `CEI`) so Slurm’s `SLURM_SUBMIT_DIR` and `results/` paths resolve correctly.

- Edit **`slurm/common.sh`** for your cluster (`CONDA_SH`, `CONDA_ENV`).
- Optional for any GPU job: `export CACHE_DIR=/path/to/hf_cache` (model weights cache).

Inference defaults to **`results/<benchmark>/job_<jobid>/`** on Slurm unless you set `EXP_DIR`.

**Resume:** main outputs use fixed names (`llava15_chair.jsonl`, `{tag}_amber.json`, `{tag}_mmhal.json`, `{tag}_mmstar.jsonl`, `{tag}_pope_{strategy}.jsonl`). Re-running with the same `--log_dir` continues from the existing file. Delete those outputs for a clean restart.

---

## Slurm: unified script (`slurm/benchmark.slurm`)

Use **`CONFIG_FILE` as an absolute path** (e.g. `"$PWD/configs/llava_w_CEI.json"`).

Optional: pass everything in one line:

`sbatch --export=ALL,BENCHMARK=chair,CONFIG_FILE=$PWD/configs/llava_w_CEI.json,CHAIR_DATA=/path/to/val2014 slurm/benchmark.slurm`

### CHAIR

```bash
export BENCHMARK=chair
export CONFIG_FILE="$PWD/configs/llava_w_CEI.json"
export CHAIR_DATA="/path/to/coco/val2014"
# optional: export CACHE_DIR="/path/to/hf_cache"
sbatch slurm/benchmark.slurm
```

### AMBER

```bash
export BENCHMARK=amber
export CONFIG_FILE="$PWD/configs/llava_w_CEI.json"
export AMBER_PATH="/path/to/AMBER"
sbatch slurm/benchmark.slurm
```

### MMHal-Bench

```bash
export BENCHMARK=mmhal
export CONFIG_FILE="$PWD/configs/llava_w_CEI.json"
export MMHAL_INPUT="/path/to/MMHal-Bench/response_template.json"
export MMHAL_IMAGES="/path/to/MMHal-Bench/images"
sbatch slurm/benchmark.slurm
```

### POPE

```bash
export BENCHMARK=pope
export CONFIG_FILE="$PWD/configs/llava_w_CEI.json"
export POPE_QUESTION_DIR="/path/to/POPE/questions"
export POPE_IMAGE_FOLDER="/path/to/coco/val2014"
sbatch slurm/benchmark.slurm
```

### MMStar

`MMSTAR_DATA_ROOT` must be the **parent** of `MMStar/images/` (the code joins `MMStar/images` itself).  
Example if data lives under `.../data/MMStar/images/`:

```bash
export MMSTAR_DATA_ROOT="/path/to/data"
```

Submit:

```bash
export BENCHMARK=mmstar
export CONFIG_FILE="$PWD/configs/llavanext_w_CEI.json"
export MMSTAR_DATA_ROOT="/path/to/parent_of_MMStar"
# optional: export MMSTAR_HF_CACHE="/path/to/hf_datasets_cache"
sbatch slurm/benchmark.slurm
```

`benchmark.slurm` uses generic `#SBATCH` lines (`gpu:1`). If your site needs different partitions or GRES (e.g. CHAIR’s GPU type), use the per-benchmark scripts below.

---

## Slurm: per-benchmark scripts (`slurm/*.slurm`)

Use these when you need the GPU/QoS/time limits defined in each file (e.g. **`slurm/chair.slurm`** for `gpu:3g.40gb` and `--qos=gpu`).

### CHAIR

```bash
export CONFIG_FILE="$PWD/configs/llava_w_CEI.json"
export CHAIR_DATA="/path/to/coco/val2014"
sbatch slurm/chair.slurm
```

### AMBER

```bash
export CONFIG_FILE="$PWD/configs/llava_w_CEI.json"
export AMBER_PATH="/path/to/AMBER"
sbatch slurm/amber.slurm
```

### MMHal-Bench

```bash
export CONFIG_FILE="$PWD/configs/llava_w_CEI.json"
export MMHAL_INPUT="/path/to/MMHal-Bench/response_template.json"
export MMHAL_IMAGES="/path/to/MMHal-Bench/images"
sbatch slurm/mmhal.slurm
```

### POPE

```bash
export CONFIG_FILE="$PWD/configs/llava_w_CEI.json"
export POPE_QUESTION_DIR="/path/to/POPE/questions"
export POPE_IMAGE_FOLDER="/path/to/coco/val2014"
sbatch slurm/pope.slurm
```

### MMStar

```bash
export CONFIG_FILE="$PWD/configs/llavanext_w_CEI.json"
export MMSTAR_DATA_ROOT="/path/to/parent_of_MMStar"
sbatch slurm/mmstar.slurm
```

---

## Scoring

All `eval/` scripts below support **`--summary_file path.json`** for a compact JSON summary (in addition to printed metrics). Replace paths with your real **`job_<id>`** folders. Inference caps use tags **`instructblip`**, **`llava15`**, **`llavanext`** (e.g. `llava15_chair.jsonl`).

### CHAIR

Requires COCO annotation JSONs under `--coco_path`:  
`captions_train2014.json`, `captions_val2014.json`, `instances_train2014.json`, `instances_val2014.json`.

```bash
python eval/chair.py \
  --cap_file ./results/chair/job_12345/llava15_chair.jsonl \
  --caption_key caption_512 \
  --coco_path /path/to/coco/annotations \
  --cache ./results/chair/chair_evaluator.pkl \
  --summary_file ./results/chair/job_12345/chair_summary.json
```

Use `caption_64` instead of `caption_512` if you score the short prefix.

### AMBER

Paths depend on where you cloned AMBER eval assets; see `python eval/amber.py --help`.

```bash
python eval/amber.py \
  --inference_data /path/to/model_responses.json \
  --evaluation_type g \
  --summary_file ./results/amber/amber_summary.json
```

### MMHal-Bench (GPT judge)

Expects the **96-record** JSON format used by `eval/eval_gpt4.py`; requires an OpenAI API key.

```bash
python eval/eval_gpt4.py \
  --response ./results/mmhal/job_12345/llava15_mmhal.json \
  --api-key "$OPENAI_API_KEY" \
  --summary_file ./results/mmhal/job_12345/mmhal_summary.json
```

### POPE

There is no POPE scorer in this repo; use your cluster’s or upstream POPE accuracy script on the **`{tag}_pope_*.jsonl`** files under `results/pope/job_<id>/`.

### MMStar

```bash
python eval/mmstar_eval.py \
  --cap_file ./results/mmstar/job_12345/llavanext_mmstar.jsonl \
  --scoring exact \
  --summary_file ./results/mmstar/job_12345/mmstar_summary.json
```

Optional tabular export:

```bash
python eval/mmstar_eval.py \
  --cap_file ./results/mmstar/job_12345/llavanext_mmstar.jsonl \
  --scoring exact \
  --summary_csv ./results/mmstar/job_12345/mmstar_summary.csv
```

---

## Local runs (no Slurm)

See **README.md** for `bash scripts/run_*.sh` examples; use `./results/<benchmark>/<run_name>/` as the log directory argument.
