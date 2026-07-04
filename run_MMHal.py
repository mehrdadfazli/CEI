#!/usr/bin/env python3
"""MMHal-Bench inference with CEI."""
import argparse
import gc
import json
import logging
import os
from datetime import datetime

import torch
from tqdm import tqdm

from bench_io import benchmark_file_tag
from cei_core import generate_two_pass_dynamic, get_context_embedding, load_image, setup_injection_hook
from model_utils import load_model_and_processor, process_inputs, model_names


def merge_json_config(namespace, config_path):
    if not config_path:
        return namespace
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    for k, v in cfg.items():
        setattr(namespace, k, v)
    return namespace


def build_parser():
    p = argparse.ArgumentParser(description="Run CEI on MMHal-Bench")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--input", type=str, default=None, help="response_template.json")
    p.add_argument("--images_root", type=str, default=None, help="Directory for MMHal images")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--log_dir", type=str, default=None)
    p.add_argument("--cache_dir", type=str, default=None)

    p.add_argument("--model_type", default="instructblip", choices=["instructblip", "llava", "llava-next"])
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--load_in_8bit", action="store_true", default=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=None)

    p.add_argument("--no_cei", action="store_true", help="Disable CEI (vanilla generation)")
    p.add_argument("--do_sample", action="store_true", default=False)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--repetition_penalty", type=float, default=1.10)
    p.add_argument("--KV_cache", action="store_true", default=False)

    p.add_argument("--injection_layer", type=int, default=10)
    p.add_argument("--context_embedding_idx", type=int, default=-1)
    p.add_argument("--context_embedding_layer", type=int, default=-1)

    p.add_argument("--K_mass", type=int, default=40)
    p.add_argument("--start_layer", type=int, default=1)
    p.add_argument("--alpha_method", default="sigmoid", choices=["sigmoid", "cosine"])
    p.add_argument("--alpha", type=float, default=0.10, help="alpha_max for dynamic CEI")
    p.add_argument("--tau", type=float, default=0.20)
    p.add_argument("--T", type=float, default=0.05)
    p.add_argument("--beta", type=float, default=0.30)
    p.add_argument("--topK_mass_start_layer", type=int, default=-1)

    p.add_argument("--dynamic_mode", default="two_pass", choices=["none", "two_pass"])
    p.add_argument("--delta", type=float, default=0.30)
    p.add_argument("--gamma", type=float, default=0.20)
    p.add_argument("--trace_dir", type=str, default=None)
    return p


def merge_mmhal_checkpoint(data, out_path):
    """Reuse non-empty, non-Error model_answer from a previous partial run."""
    if not os.path.isfile(out_path):
        return data
    try:
        with open(out_path, "r", encoding="utf-8") as f:
            prev = json.load(f)
    except (json.JSONDecodeError, OSError):
        return data
    if not isinstance(prev, list) or len(prev) != len(data):
        return data
    for i, item in enumerate(data):
        pa = prev[i].get("model_answer")
        if isinstance(pa, str) and pa.strip() and pa.strip() != "Error":
            item["model_answer"] = pa
    return data


def _save_mmhal_out(out_path, data):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


def main():
    args = build_parser().parse_args()
    args = merge_json_config(args, args.config)

    if not args.input:
        raise SystemExit("--input (MMHal JSON) is required")
    if not args.images_root:
        raise SystemExit("--images_root is required")
    log_dir = args.log_dir or "./results/mmhal"
    os.makedirs(log_dir, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    log_file = os.path.join(log_dir, f"log_mmhal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    model, processor = load_model_and_processor(
        args.model_type, model_names, args.cache_dir, device, args.load_in_8bit
    )
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    tag = benchmark_file_tag(args.model_type)
    out_path = args.output or os.path.join(log_dir, f"{tag}_mmhal.json")
    with open(os.path.join(log_dir, "config_mmhal.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    if args.limit is not None:
        data = data[: args.limit]
    data = merge_mmhal_checkpoint(data, out_path)

    for idx, item in enumerate(tqdm(data, desc="MMHal")):
        ma = item.get("model_answer")
        if isinstance(ma, str) and ma.strip() and ma.strip() != "Error":
            continue
        image_src = item.get("image_src", "")
        question = item.get("question", "")
        if image_src.startswith("http://") or image_src.startswith("https://"):
            image_path = image_src
        else:
            image_path = (
                image_src if os.path.isabs(image_src) else os.path.join(args.images_root, image_src)
            )

        try:
            raw_image = load_image(image_path)
        except OSError as e:
            logger.error("[%s] image error %s: %s", idx, image_path, e)
            item["model_answer"] = "Error"
            _save_mmhal_out(out_path, data)
            continue

        prompt = question
        use_cei = not args.no_cei
        c = None
        if use_cei:
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
                logger.error("[%s] context emb: %s", idx, e)
                item["model_answer"] = "Error"
                _save_mmhal_out(out_path, data)
                continue

        trace_path = None
        if args.trace_dir:
            os.makedirs(args.trace_dir, exist_ok=True)
            basename = os.path.basename(image_src).replace("/", "_")
            trace_path = os.path.join(args.trace_dir, f"{idx}_{basename}.jsonl")

        try:
            if use_cei and args.dynamic_mode == "two_pass":
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
            elif use_cei:
                hook_handle = setup_injection_hook(model, args.injection_layer, c, args.alpha)
                inputs = process_inputs(raw_image, prompt, processor, args.model_type)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        do_sample=args.do_sample,
                        max_new_tokens=args.max_new_tokens,
                        repetition_penalty=args.repetition_penalty,
                    )
                gen_ids = outputs[:, inputs["input_ids"].shape[-1] :]
                cap64 = processor.batch_decode(gen_ids[:, :64], skip_special_tokens=True)[0].strip()
                cap512 = processor.batch_decode(gen_ids[:, :512], skip_special_tokens=True)[0].strip()
                hook_handle.remove()
            else:
                inputs = process_inputs(raw_image, prompt, processor, args.model_type)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        do_sample=args.do_sample,
                        max_new_tokens=args.max_new_tokens,
                        repetition_penalty=args.repetition_penalty,
                    )
                gen_ids = outputs[:, inputs["input_ids"].shape[-1] :]
                cap64 = processor.batch_decode(gen_ids[:, :64], skip_special_tokens=True)[0].strip()
                cap512 = processor.batch_decode(gen_ids[:, :512], skip_special_tokens=True)[0].strip()

            answer = cap512.strip() if cap512.strip() else cap64.strip()
            item["model_answer"] = answer if answer else " "
        except Exception as e:
            logger.error("[%s] gen: %s", idx, e)
            item["model_answer"] = "Error"

        _save_mmhal_out(out_path, data)

        torch.cuda.empty_cache()
        gc.collect()

    logger.info("Done: %s", out_path)


if __name__ == "__main__":
    main()
