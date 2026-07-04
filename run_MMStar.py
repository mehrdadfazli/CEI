#!/usr/bin/env python3
"""MMStar benchmark: HF metadata + local PNG cache; CEI inference; JSONL for eval/mmstar_eval.py."""
import argparse
import gc
import json
import logging
import os
from datetime import datetime
from io import BytesIO

import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from bench_io import benchmark_file_tag, load_jsonl_int_field
from cei_core import generate_two_pass_dynamic, get_context_embedding, load_image
from model_utils import load_model_and_processor, model_names


def merge_json_config(namespace, config_path):
    if not config_path:
        return namespace
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    for k, v in cfg.items():
        setattr(namespace, k, v)
    return namespace


def process_data_mmstar(
    data_root,
    hf_dataset_id="Lin-Chen/MMStar",
    hf_config="val",
    split="val",
    cache_dir=None,
    max_samples=None,
):
    kwargs = {"trust_remote_code": True}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    ds = load_dataset(hf_dataset_id, hf_config, split=split, **kwargs)
    img_root = os.path.join(data_root, "MMStar", "images")
    os.makedirs(img_root, exist_ok=True)

    data_list = []
    for i, row in enumerate(ds):
        if max_samples is not None and i >= max_samples:
            break
        idx = int(row["index"])
        img_path = os.path.join(img_root, f"{idx}.png")
        if not os.path.isfile(img_path):
            im = row["image"]
            if isinstance(im, Image.Image):
                im = im.convert("RGB")
            elif isinstance(im, dict):
                if im.get("bytes"):
                    im = Image.open(BytesIO(im["bytes"])).convert("RGB")
                elif im.get("path"):
                    im = Image.open(im["path"]).convert("RGB")
                else:
                    raise ValueError("MMStar image dict missing bytes/path")
            else:
                im = Image.open(im).convert("RGB")
            im.save(img_path)

        question = row["question"]
        prompt = question.rstrip()
        if "Please select the correct answer" not in prompt:
            prompt = prompt + "\nPlease select the correct answer from the options above.\n"

        new_result = {
            "index": idx,
            "img_url": img_path,
            "prompt": prompt,
            "lan": "mmstar",
            "type": "choice",
            "answer": str(row["answer"]).strip(),
            "question": question,
        }
        if row.get("category") is not None:
            new_result["category"] = row["category"]
        if row.get("l2_category") is not None:
            new_result["l2_category"] = row["l2_category"]
        data_list.append(new_result)
    return data_list


def build_parser():
    p = argparse.ArgumentParser(description="Run CEI on MMStar")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--mmstar_data_root", type=str, default=None, help="Parent of MMStar/images/")
    p.add_argument("--mmstar_hf_cache", type=str, default=None, help="HF datasets cache for metadata")
    p.add_argument("--mmstar_hf_id", type=str, default="Lin-Chen/MMStar")
    p.add_argument("--mmstar_hf_config", type=str, default="val")
    p.add_argument("--mmstar_split", type=str, default="val")
    p.add_argument("--mmstar_max_samples", type=int, default=None)
    p.add_argument("--log_dir", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--cache_dir", type=str, default=None)

    p.add_argument("--model_type", default="instructblip", choices=["instructblip", "llava", "llava-next"])
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--load_in_8bit", action="store_true", default=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=None)

    p.add_argument("--do_sample", action="store_true", default=False)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--repetition_penalty", type=float, default=1.10)
    p.add_argument("--KV_cache", action="store_true", default=False)

    p.add_argument("--injection_layer", type=int, default=10)
    p.add_argument("--context_embedding_idx", type=int, default=-1)
    p.add_argument("--context_embedding_layer", type=int, default=-1)

    p.add_argument("--K_mass", type=int, default=40)
    p.add_argument("--start_layer", type=int, default=1)
    p.add_argument("--alpha_method", default="sigmoid", choices=["sigmoid", "cosine"])
    p.add_argument("--alpha", type=float, default=0.10, help="alpha_max")
    p.add_argument("--tau", type=float, default=0.20)
    p.add_argument("--T", type=float, default=0.05)
    p.add_argument("--beta", type=float, default=0.30)
    p.add_argument("--topK_mass_start_layer", type=int, default=-1)
    p.add_argument("--delta", type=float, default=0.30)
    p.add_argument("--gamma", type=float, default=0.20)
    p.add_argument("--trace_dir", type=str, default=None)
    return p


def main():
    args = build_parser().parse_args()
    args = merge_json_config(args, args.config)

    if not args.mmstar_data_root:
        raise SystemExit("--mmstar_data_root is required")
    if not args.log_dir:
        raise SystemExit("--log_dir is required")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, f"log_mmstar_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    model, processor = load_model_and_processor(
        args.model_type, model_names, args.cache_dir, device, args.load_in_8bit
    )
    model.eval()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    tag = benchmark_file_tag(args.model_type)
    with open(os.path.join(args.log_dir, "config_mmstar.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    out_path = args.output or os.path.join(args.log_dir, f"{tag}_mmstar.jsonl")
    hf_cache = args.mmstar_hf_cache or args.cache_dir

    data = process_data_mmstar(
        args.mmstar_data_root,
        hf_dataset_id=args.mmstar_hf_id,
        hf_config=args.mmstar_hf_config,
        split=args.mmstar_split,
        cache_dir=hf_cache,
        max_samples=args.mmstar_max_samples,
    )
    if args.limit is not None:
        data = data[: args.limit]

    done_idx = load_jsonl_int_field(out_path, "index")
    data = [row for row in data if int(row["index"]) not in done_idx]
    if done_idx:
        logger.info(
            "Resume MMStar: skipping %d rows already in %s; %d remaining",
            len(done_idx),
            out_path,
            len(data),
        )

    if not data:
        logger.info("MMStar: nothing to run (all indices already in %s).", out_path)
        return

    with open(out_path, "a", encoding="utf-8", buffering=1) as fout:
        for idx, item in enumerate(tqdm(data, desc="MMStar")):
            img_path = item["img_url"]
            prompt = item["prompt"]
            try:
                raw_image = load_image(img_path)
            except OSError as e:
                logger.error("[%s] %s", idx, e)
                fout.write(json.dumps({**item, "response": "Error"}, default=str) + "\n")
                fout.flush()
                os.fsync(fout.fileno())
                continue

            try:
                c = get_context_embedding(
                    raw_image,
                    prompt,
                    model=model,
                    processor=processor,
                    model_type=args.model_type,
                    ctx_layer=args.context_embedding_layer,
                    ctx_idx=args.context_embedding_idx,
                )
            except Exception as e:
                logger.error("[%s] ctx: %s", idx, e)
                fout.write(json.dumps({**item, "response": "Error"}, default=str) + "\n")
                fout.flush()
                os.fsync(fout.fileno())
                continue

            trace_path = None
            if args.trace_dir:
                os.makedirs(args.trace_dir, exist_ok=True)
                trace_path = os.path.join(args.trace_dir, f"{idx}_{item['index']}.jsonl")

            try:
                cap64, cap512 = generate_two_pass_dynamic(
                    raw_image,
                    prompt,
                    model=model,
                    processor=processor,
                    model_type=args.model_type,
                    context_embedding=c,
                    injection_layer=args.injection_layer,
                    K_mass=args.K_mass,
                    start_layer=args.start_layer,
                    alpha_method=args.alpha_method,
                    alpha_max=args.alpha,
                    tau=args.tau,
                    T=args.T,
                    beta=args.beta,
                    max_new_tokens=args.max_new_tokens,
                    topK_mass_start_layer=args.topK_mass_start_layer,
                    do_sample=args.do_sample,
                    logger=logger,
                    trace_path=trace_path,
                    question_id=idx,
                    delta=args.delta,
                    gamma=args.gamma,
                    repetition_penalty=args.repetition_penalty,
                    KV_cache=args.KV_cache,
                )
                answer = cap512.strip() if cap512.strip() else cap64.strip()
                record = {**item, "response": answer if answer else " "}
            except Exception as e:
                logger.error("[%s] gen: %s", idx, e)
                record = {**item, "response": "Error"}

            fout.write(json.dumps(record, default=str) + "\n")
            fout.flush()
            os.fsync(fout.fileno())
            torch.cuda.empty_cache()
            gc.collect()

    logger.info("Done: %s", out_path)


if __name__ == "__main__":
    main()
