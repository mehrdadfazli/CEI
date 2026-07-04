#!/usr/bin/env python3
"""AMBER benchmark inference with CEI."""
import argparse
import gc
import json
import logging
import os
from datetime import datetime

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from bench_io import benchmark_file_tag
from cei_core import generate_two_pass_dynamic, get_context_embedding, setup_injection_hook
from model_utils import load_model_and_processor, process_inputs, model_names


def merge_json_config(namespace, config_path):
    if not config_path:
        return namespace
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    for k, v in cfg.items():
        setattr(namespace, k, v)
    if not hasattr(namespace, "use_CEI"):
        namespace.use_CEI = True
    return namespace


def recorder(out):
    neg_words = ["No", "not", "no", "NO"]
    out = out.replace(".", "").replace(",", "")
    words = out.split()
    if any(w in neg_words for w in words) or any(w.endswith("n't") for w in words):
        return "No"
    return "Yes"


def is_generative_item(item):
    t = str(item.get("type", "")).lower()
    if any(k in t for k in ["caption", "describe", "generation", "generative"]):
        return True
    return int(item.get("id", 10**9)) <= 1004


def build_parser():
    p = argparse.ArgumentParser(description="Run CEI on AMBER benchmark")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--amber_path", type=str, default=None, help="AMBER dataset root")
    p.add_argument("--log_dir", type=str, default=None)
    p.add_argument("--cache_dir", type=str, default=None)

    p.add_argument("--queries_json", default="data/query/query_all.json")
    p.add_argument("--annotations_json", default="data/annotations.json")
    p.add_argument("--image_dirname", default="image")

    p.add_argument("--model_type", default="llava", choices=["instructblip", "llava", "llava-next"])
    p.add_argument("--load_in_8bit", action="store_true", default=True)
    p.add_argument("--no_cei", action="store_true", help="Disable CEI")
    p.add_argument("--do_sample", action="store_true", default=False)
    p.add_argument("--num_beams", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=512)

    p.add_argument("--context_embedding_idx", type=int, default=-1)
    p.add_argument("--context_embedding_layer", type=int, default=-1)
    p.add_argument("--injection_layer", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.1)

    p.add_argument("--dynamic_mode", default="two_pass", choices=["none", "two_pass"])
    p.add_argument("--alpha_method", default="sigmoid", choices=["sigmoid", "cosine"])
    p.add_argument("--K_mass", type=int, default=40)
    p.add_argument("--start_layer", type=int, default=1)
    p.add_argument("--topK_mass_start_layer", type=int, default=-1)
    p.add_argument("--tau", type=float, default=0.2)
    p.add_argument("--T", type=float, default=0.05)
    p.add_argument("--beta", type=float, default=0.30)

    p.add_argument("--only_generative", action="store_true", default=False)
    p.add_argument("--num_items", type=int, default=None)
    p.add_argument("--random_seed", type=int, default=42)

    p.add_argument("--repetition_penalty", type=float, default=1.0)
    p.add_argument("--delta", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=0.0)
    p.add_argument("--KV_cache", action="store_true", default=False)
    return p


def main():
    args = build_parser().parse_args()
    args = merge_json_config(args, args.config)
    if not args.amber_path:
        raise SystemExit("--amber_path is required")
    if not args.log_dir:
        raise SystemExit("--log_dir is required")

    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    use_cei = bool(getattr(args, "use_CEI", True)) and not args.no_cei

    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_config = {
        "context_embedding_idx": args.context_embedding_idx,
        "context_embedding_layer": args.context_embedding_layer,
        "injection_layer": args.injection_layer,
        "alpha": args.alpha,
        "max_new_tokens": args.max_new_tokens,
        "dynamic_mode": args.dynamic_mode,
        "alpha_method": args.alpha_method,
        "K_mass": args.K_mass,
        "start_layer": args.start_layer,
        "topK_mass_start_layer": args.topK_mass_start_layer,
        "tau": args.tau,
        "T": args.T,
        "beta": args.beta,
        "repetition_penalty": args.repetition_penalty,
    }
    tag = benchmark_file_tag(args.model_type)
    with open(os.path.join(args.log_dir, "config_amber.json"), "w", encoding="utf-8") as f:
        json.dump(exp_config, f, indent=4)

    output_file = os.path.join(args.log_dir, f"{tag}_amber.json")

    model, processor = load_model_and_processor(
        args.model_type, model_names, args.cache_dir, device, args.load_in_8bit
    )

    json_query_path = os.path.join(args.amber_path, args.queries_json)
    image_dir = os.path.join(args.amber_path, args.image_dirname)
    with open(json_query_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = [it for it in data if (is_generative_item(it) if args.only_generative else True)]
    if args.num_items is not None:
        rng = np.random.default_rng(args.random_seed)
        idxs = rng.permutation(len(items))[: args.num_items]
        items = [items[i] for i in idxs]

    responses = []
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                responses = json.load(f)
        except json.JSONDecodeError:
            responses = []
    processed_ids = {r.get("id") for r in responses}

    for item in tqdm(items, desc="AMBER"):
        question_id = int(item["id"])
        if question_id in processed_ids:
            continue
        img_path = os.path.join(image_dir, item["image"])
        try:
            raw_image = Image.open(img_path).convert("RGB")
        except OSError as e:
            logger.error("Skip %s: %s", img_path, e)
            continue

        query = item.get("query", "Describe this image.")

        context_embedding = None
        if use_cei:
            context_embedding = get_context_embedding(
                raw_image,
                query,
                model=model,
                processor=processor,
                model_type=args.model_type,
                ctx_layer=args.context_embedding_layer,
                ctx_idx=args.context_embedding_idx,
            )

        if use_cei and args.dynamic_mode == "two_pass":
            caption_64, caption_512 = generate_two_pass_dynamic(
                raw_image,
                query,
                model=model,
                processor=processor,
                model_type=args.model_type,
                context_embedding=context_embedding,
                injection_layer=exp_config["injection_layer"],
                K_mass=args.K_mass,
                start_layer=args.start_layer,
                topK_mass_start_layer=args.topK_mass_start_layer,
                alpha_method=args.alpha_method,
                alpha_max=exp_config["alpha"],
                tau=args.tau,
                T=args.T,
                max_new_tokens=exp_config["max_new_tokens"],
                do_sample=args.do_sample,
                logger=logger,
                beta=args.beta,
                repetition_penalty=args.repetition_penalty,
                delta=args.delta,
                gamma=args.gamma,
                KV_cache=args.KV_cache,
                question_id=question_id,
            )
        else:
            hook_handle = None
            if use_cei:
                hook_handle = setup_injection_hook(
                    model, exp_config["injection_layer"], context_embedding, exp_config["alpha"]
                )
            inputs = process_inputs(raw_image, query, processor, args.model_type)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    do_sample=args.do_sample,
                    max_new_tokens=exp_config["max_new_tokens"],
                    num_beams=args.num_beams,
                    repetition_penalty=args.repetition_penalty,
                )
            gen_ids = outputs[:, inputs["input_ids"].shape[-1] :]
            caption_64 = processor.batch_decode(gen_ids[:, :64], skip_special_tokens=True)[0].strip()
            caption_512 = processor.batch_decode(gen_ids[:, :512], skip_special_tokens=True)[0].strip()
            if hook_handle is not None:
                hook_handle.remove()

        if question_id > 1004:
            result = {"id": question_id, "response": recorder(caption_64)}
        else:
            result = {"id": question_id, "response_64": caption_64, "response_512": caption_512}

        responses.append(result)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(responses, f, indent=4)
            f.flush()
            os.fsync(f.fileno())
        processed_ids.add(question_id)
        torch.cuda.empty_cache()
        gc.collect()

    logger.info("Done. Results: %s", output_file)


if __name__ == "__main__":
    main()
