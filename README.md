# [ACL 2026 Finding] [Inject to Heal: Alleviating hallucination in LVLMs via Context Embedding Injection](https://aclanthology.org/2026.findings-acl.2048/)

## Project Description
This repository contains the official implementation of Context Embedding Injection (CEI), a training-free hallucination mitigation method for large vision-language models (LVLM). It provides a robust framework for researchers and developers to conduct experiments and analyze results in a structured and reproducible manner.

Context Embedding Injection (CEI) framework addresses hallucination challenge by leveraging a key mechanistic discovery: the **commitment-depth gap**, which reveals that truthful tokens accumulate probability mass earlier across decoder layers than hallucinatory ones. CEI employs a two-variant approach utilizing the initial context embedding as a grounding signal: **Static CEI** to inject a fixed visual grounding signal across all decoding steps, and **Dynamic CEI** to adaptively modulate the injection strength per token based on the model's ongoing confidence. This confidence-driven adjustment ensures consistent visual alignment during generation.

## CEI Framework


## Supported models (7B)

- InstructBLIP (Vicuna-7B)
- LLaVA-1.5-7B
- LLaVA-NeXT (Vicuna-7B)

## Benchmarks

| Benchmark | Runner | Evaluation |
|-----------|--------|------------|
| CHAIR | `run_CHAIR.py` | `eval/chair.py` |
| AMBER | `run_AMBER.py` | `eval/amber.py` |
| MMHal-Bench | `run_MMHal.py` | `eval/eval_gpt4.py` (GPT judge) |
| POPE | `run_POPE.py` | your POPE scoring script |
| MMStar | `run_MMStar.py` | `eval/mmstar_eval.py` |

## Installation

```bash
conda create -n cei python=3.10 -y
conda activate cei
pip install -r requirements.txt
```

You need a CUDA-capable GPU, PyTorch compatible with your driver, and Hugging Face access for model weights.

## Repository layout

```
CEI/
├── cei_core.py          # alpha_from_mass, setup_injection_hook, get_context_embedding, generate_two_pass_dynamic
├── model_utils.py       # load_model_and_processor, process_inputs
├── run_CHAIR.py … run_MMStar.py
├── configs/             # JSON hyperparameters per model / CEI setup
├── scripts/             # run_*.sh wrappers (pass dataset paths as arguments)
├── eval/                # benchmark scoring scripts
└── slurm/               # Slurm jobs + common.sh
```

Inference outputs for each benchmark go under **`results/<benchmark>/`**, typically in a per-job subdirectory such as `results/chair/job_<SLURM_JOB_ID>/` when using Slurm. Pass any subdirectory you like as the log directory when running locally.

Dataset paths are **not** stored in the JSON configs; pass them via CLI or the shell wrappers (third argument onward).

## Quick start (from repo root)

Set `PYTHONPATH=.` or run from this directory so `cei_core` and `model_utils` import correctly.

### CHAIR

```bash
bash scripts/run_chair.sh configs/llava_w_CEI.json ./results/chair/my_run /path/to/coco/val2014 /path/to/hf_cache
```

### AMBER

```bash
bash scripts/run_amber.sh configs/llava_w_CEI.json ./results/amber/my_run /path/to/AMBER /path/to/hf_cache
```

### MMHal-Bench

```bash
bash scripts/run_mmhal.sh configs/llava_w_CEI.json ./results/mmhal/my_run \
  /path/to/MMHal-Bench/response_template.json \
  /path/to/MMHal-Bench/images \
  /path/to/hf_cache
```

### POPE

```bash
bash scripts/run_pope.sh configs/llava_w_CEI.json ./results/pope/my_run \
  /path/to/POPE/questions \
  /path/to/coco/val2014 \
  /path/to/hf_cache
```

### MMStar

Requires `datasets` and Hugging Face metadata (`Lin-Chen/MMStar`); images should live under `<mmstar_data_root>/MMStar/images/{index}.png`.

```bash
bash scripts/run_mmstar.sh configs/llavanext_w_CEI.json ./results/mmstar/my_run \
  /path/to/data_root_parent \
  /path/to/hf_datasets_cache \
  /path/to/hf_cache
```

## Slurm (single submission command)

Edit `slurm/common.sh` (`CONDA_SH`, `CONDA_ENV`) for your cluster. From the repo root, set `BENCHMARK` to `chair`, `amber`, `mmhal`, `pope`, or `mmstar`, set `CONFIG_FILE`, then set the dataset variables required for that benchmark (same names as in the per-benchmark `slurm/*.slurm` files). Example for CHAIR:

```bash
export BENCHMARK=chair
export CONFIG_FILE="$PWD/configs/llava_w_CEI.json"
export CHAIR_DATA="/path/to/coco/val2014"
sbatch slurm/benchmark.slurm
```

Outputs are written under `results/${BENCHMARK}/job_<jobid>/`. If your site needs different GPU partitions or GRES lines (for example the CHAIR script’s GPU type), use the matching file under `slurm/` (`chair.slurm`, `amber.slurm`, …) instead; the layout under `results/<benchmark>/` is the same.

## Scoring (single command pattern)

All scoring scripts under `eval/` support **`--summary_file <path.json>`** to write a compact JSON summary in addition to printing metrics. Use your own inference output paths; placeholders below illustrate the pattern.

Inference outputs use **fixed names** (no random suffix): e.g. `llava15_chair.jsonl`, `llavanext_mmstar.jsonl`, `{tag}_pope_{strategy}.jsonl`.  
`tag` is `instructblip`, `llava15`, or `llavanext`. **Re-running the same `--log_dir` resumes** from the existing output (skip finished items). Delete the output file(s) to start from scratch.

**CHAIR** (requires COCO annotation JSONs under `--coco_path`:  
`captions_train2014.json`, `captions_val2014.json`, `instances_train2014.json`, `instances_val2014.json`):

```bash
python eval/chair.py \
  --cap_file ./results/chair/job_12345/llava15_chair.jsonl \
  --caption_key caption_512 \
  --coco_path /path/to/coco/annotations \
  --cache ./results/chair/chair_evaluator.pkl \
  --summary_file ./results/chair/job_12345/chair_summary.json
```

**AMBER** (paths are relative to the AMBER eval data layout; see `eval/amber.py --help`):

```bash
python eval/amber.py \
  --inference_data /path/to/model_responses.json \
  --evaluation_type g \
  --summary_file ./results/amber/amber_summary.json
```

**MMHal-Bench** (GPT judge; requires API key and 96-record response JSON as in the original eval script):

```bash
python eval/eval_gpt4.py \
  --response ./results/mmhal/job_12345/llava15_mmhal.json \
  --api-key "$OPENAI_API_KEY" \
  --summary_file ./results/mmhal/job_12345/mmhal_summary.json
```

**MMStar**:

```bash
python eval/mmstar_eval.py \
  --cap_file ./results/mmstar/job_12345/llavanext_mmstar.jsonl \
  --scoring exact \
  --summary_file ./results/mmstar/job_12345/mmstar_summary.json
```

`eval/mmstar_eval.py` also accepts **`--summary_csv`** for a tabular accuracy export (unchanged).

## Config keys (JSON)

| Key | Meaning |
|-----|---------|
| `model_type` | `instructblip`, `llava`, or `llava-next` |
| `use_CEI` | `true` / `false` (shell maps false to `--no_cei` where applicable) |
| `dynamic_mode` | `two_pass` or `none` (static CEI with fixed `alpha`) |
| `alpha` | Max injection strength (`alpha_max` in two-pass mode) |
| `alpha_method` | `sigmoid` or `cosine` (maps mean Top-K mass to `alpha`) |
| `injection_layer` | Decoder layer index for the hook |
| `context_embedding_layer`, `context_embedding_idx` | Layer and token index for context vector (negative = from end) |
| `K_mass`, `start_layer`, `topK_mass_start_layer` | Top-K mass probe hyperparameters |
| `tau`, `T` | Sigmoid mapping parameters (when `alpha_method` is `sigmoid`) |
| `beta` | Cosine mapping parameter (when `alpha_method` is `cosine`) |
| `delta`, `gamma` | Word-boundary gate and plausibility fallback |
| `KV_cache` | Use KV cache inside two-pass for LLaVA / LLaVA-NeXT |
| `max_new_tokens` | Generation budget (use smaller values for MMStar / POPE) |

## Citation

```
@inproceedings{fazli2026inject,
  title={Inject to Heal: Alleviating hallucination in LVLMs via Context Embedding Injection},
  author={Fazli, Mehrdad and Wei, Bowen and Zhu, Ziwei},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2026},
  pages={41177--41193},
  year={2026}
}
```

If you use this code, please cite the ACL Findings paper (bibtex to be added when proceedings are available).

## License

Use and modification permitted for research; include citation when redistributing derived work.
