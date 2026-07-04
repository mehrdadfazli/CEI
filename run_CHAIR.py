#!/usr/bin/env python3
"""CHAIR benchmark inference with CEI."""
import argparse
import gc
import json
import logging
import os
import random
from datetime import datetime

import torch
from PIL import Image
from tqdm import tqdm

from bench_io import benchmark_file_tag, load_jsonl_int_field
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


def load_opera_image_ids(opera_results_path):
    image_ids = []
    with open(opera_results_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            image_ids.append(data["image_id"])
    return image_ids


def build_parser():
    p = argparse.ArgumentParser(description="Run CEI on CHAIR benchmark")
    p.add_argument("--config", type=str, default=None, help="JSON file with CEI hyperparameters")
    p.add_argument("--chair_data_path", type=str, default=None, help="Directory with COCO val2014 JPGs")
    p.add_argument("--log_dir", type=str, default=None, help="Output directory")
    p.add_argument("--cache_dir", type=str, default=None, help="Hugging Face cache")

    p.add_argument("--model_type", default="instructblip", choices=["instructblip", "llava", "llava-next"])
    p.add_argument("--load_in_8bit", action="store_true", default=True)
    p.add_argument("--no_cei", action="store_true", help="Disable CEI (vanilla generation)")
    p.add_argument("--do_sample", action="store_true", default=False)
    p.add_argument("--num_beams", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=512)

    p.add_argument("--context_embedding_idx", type=int, default=-1)
    p.add_argument("--context_embedding_layer", type=int, default=-1)
    p.add_argument("--injection_layer", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.1)

    p.add_argument("--opera_results", action="store_true", default=False)
    p.add_argument("--opera_results_path", type=str, default=None, help="JSONL with image_id per line")
    p.add_argument("--num_images", type=int, default=500)
    p.add_argument("--random_seed", type=int, default=42)

    p.add_argument("--dynamic_mode", default="two_pass", choices=["none", "two_pass"])
    p.add_argument("--alpha_method", default="sigmoid", choices=["sigmoid", "cosine"])
    p.add_argument("--K_mass", type=int, default=40)
    p.add_argument("--start_layer", type=int, default=1)
    p.add_argument("--tau", type=float, default=0.2)
    p.add_argument("--T", type=float, default=0.05)
    p.add_argument("--topK_mass_start_layer", type=int, default=-1)
    p.add_argument("--beta", type=float, default=0.30)

    p.add_argument("--trace_alpha", action="store_true", default=False)
    p.add_argument("--trace_dir", type=str, default=None)
    p.add_argument("--trace_samples", type=int, default=20)

    p.add_argument("--repetition_penalty", type=float, default=1.1)
    p.add_argument("--delta", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=0.0)
    p.add_argument("--KV_cache", action="store_true", default=False)
    return p


def main():
    args = build_parser().parse_args()
    args = merge_json_config(args, args.config)

    if not args.chair_data_path:
        raise SystemExit("--chair_data_path is required (or set in --config)")
    if not args.log_dir:
        raise SystemExit("--log_dir is required (or set in --config)")

    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    use_cei = bool(getattr(args, "use_CEI", True)) and not args.no_cei
    logger.info("use_CEI=%s", use_cei)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

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
    config_path = os.path.join(args.log_dir, "config_chair.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({**exp_config, "chair_data_path": args.chair_data_path}, f, indent=4)

    output_file = os.path.join(args.log_dir, f"{tag}_chair.jsonl")
    done_ids = load_jsonl_int_field(output_file, "image_id")

    trace_root = args.trace_dir or os.path.join(args.log_dir, "traces")
    if args.trace_alpha:
        os.makedirs(trace_root, exist_ok=True)
    n_traced = 0

    model, processor = load_model_and_processor(
        args.model_type, model_names, args.cache_dir, device, args.load_in_8bit
    )

    if args.opera_results:
        if not args.opera_results_path:
            raise SystemExit("--opera_results_path required when --opera_results")
        image_ids = load_opera_image_ids(args.opera_results_path)
        image_list = [(f"COCO_val2014_{i:012d}.jpg", i) for i in image_ids]
    else:
        img_files = [f for f in os.listdir(args.chair_data_path) if f.endswith(".jpg")]
        image_list = [(f, int(f.split(".jpg")[0][-6:])) for f in img_files]
        random.seed(args.random_seed)
        random.shuffle(image_list)
        if args.num_images is not None:
            image_list = image_list[: args.num_images]

    image_list = [(f, i) for f, i in image_list if i not in done_ids]
    if done_ids:
        logger.info("Resume CHAIR: skipping %d images already in %s; %d remaining", len(done_ids), output_file, len(image_list))

    query = "Describe this image."

    for img_file, img_id in tqdm(image_list, desc="CHAIR"):
        img_path = os.path.join(args.chair_data_path, img_file)
        raw_image = Image.open(img_path).convert("RGB")

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

        do_trace = args.trace_alpha and (n_traced < args.trace_samples)
        trace_path = None
        if do_trace:
            trace_path = os.path.join(trace_root, f"{tag}_chair_{img_id}.trace.jsonl")

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
                trace_path=trace_path,
                image_id=img_id,
            )
            if do_trace:
                n_traced += 1
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

        result = {"image_id": img_id, "caption_64": caption_64, "caption_512": caption_512}
        with open(output_file, "a", encoding="utf-8") as f:
            json.dump(result, f)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

        torch.cuda.empty_cache()
        gc.collect()

    logger.info("Done. Results: %s", output_file)


if __name__ == "__main__":
    main()
