#!/usr/bin/env python3
"""POPE benchmark inference with CEI (three strategies: adversarial, popular, random)."""
import argparse
import gc
import json
import logging
import os
from datetime import datetime

from bench_io import benchmark_file_tag, load_jsonl_int_field
from cei_core import generate_two_pass_dynamic, get_context_embedding, setup_injection_hook
from model_utils import load_model_and_processor, process_inputs, model_names

STRATEGIES = ["adversarial", "popular", "random"]


def merge_json_config(namespace, config_path):
    if not config_path:
        return namespace
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    for k, v in cfg.items():
        setattr(namespace, k, v)
    return namespace


def build_parser():
    p = argparse.ArgumentParser(description="Run CEI on POPE")
    p.add_argument("--config", type=str, default=None, help="JSON hyperparameters")
    p.add_argument("--question_dir", type=str, default=None, help="Dir with coco_pope_{strategy}.jsonl")
    p.add_argument("--image_folder", type=str, default=None, help="COCO val2014 images")
    p.add_argument("--log_dir", type=str, default=None)
    p.add_argument("--cache_dir", type=str, default=None)

    p.add_argument("--model_type", default="instructblip", choices=["instructblip", "llava", "llava-next"])
    p.add_argument("--load_in_8bit", action="store_true", default=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--do_sample", action="store_true", default=False)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--repetition_penalty", type=float, default=1.0)

    p.add_argument("--injection_layer", type=int, default=10)
    p.add_argument("--context_embedding_idx", type=int, default=-1)
    p.add_argument("--context_embedding_layer", type=int, default=-1)

    p.add_argument("--K_mass", type=int, default=40)
    p.add_argument("--start_layer", type=int, default=1)
    p.add_argument("--alpha_method", default="cosine", choices=["sigmoid", "cosine"])
    p.add_argument("--alpha", type=float, default=0.10, help="alpha_max")
    p.add_argument("--tau", type=float, default=0.20)
    p.add_argument("--T", type=float, default=0.05)
    p.add_argument("--beta", type=float, default=0.30)
    p.add_argument("--topK_mass_start_layer", type=int, default=-1)

    p.add_argument("--dynamic_mode", default="two_pass", choices=["none", "two_pass"])
    p.add_argument("--delta", type=float, default=0.30)
    p.add_argument("--gamma", type=float, default=0.20)
    p.add_argument("--KV_cache", action="store_true", default=False)
    p.add_argument("--trace_dir", type=str, default=None)
    return p


def run_strategy(strategy, model, processor, args, logger, tag):
    q_path = os.path.join(args.question_dir, f"coco_pope_{strategy}.json")
    out_path = os.path.join(args.log_dir, f"{tag}_pope_{strategy}.jsonl")

    with open(q_path, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]
    done = load_jsonl_int_field(out_path, "question_id")
    questions = [q for q in questions if q["question_id"] not in done]
    if done:
        logger.info(
            "[%s] resume: %d already in %s; %d remaining",
            strategy,
            len(done),
            out_path,
            len(questions),
        )
    else:
        logger.info("[%s] %d questions from %s", strategy, len(questions), q_path)

    os.makedirs(args.log_dir, exist_ok=True)
    mode = "a" if done else "w"
    with open(out_path, mode, encoding="utf-8", buffering=1) as fout:
        for item in tqdm(questions, desc=f"POPE-{strategy}"):
            qid = item["question_id"]
            iq = item["text"]
            img_file = item["image"]
            img_path = os.path.join(args.image_folder, img_file)
            prompt = iq.strip() + " Please answer this question with one word."

            try:
                raw_image = Image.open(img_path).convert("RGB")
            except OSError as e:
                logger.error("[%s] qid=%s image: %s", strategy, qid, e)
                continue

            try:
                context_embedding = get_context_embedding(
                    raw_image,
                    prompt,
                    model=model,
                    processor=processor,
                    model_type=args.model_type,
                    ctx_layer=args.context_embedding_layer,
                    ctx_idx=args.context_embedding_idx,
                )
            except Exception as e:
                logger.error("[%s] qid=%s ctx: %s", strategy, qid, e)
                continue

            trace_path = None
            if args.trace_dir:
                os.makedirs(args.trace_dir, exist_ok=True)
                trace_path = os.path.join(args.trace_dir, f"{strategy}_{qid}.jsonl")

            try:
                if args.dynamic_mode == "two_pass":
                    cap64, _ = generate_two_pass_dynamic(
                        raw_image,
                        prompt,
                        model=model,
                        processor=processor,
                        model_type=args.model_type,
                        context_embedding=context_embedding,
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
                        question_id=qid,
                        delta=args.delta,
                        gamma=args.gamma,
                        repetition_penalty=args.repetition_penalty,
                        KV_cache=args.KV_cache,
                    )
                    text = cap64.strip()
                else:
                    hook_handle = setup_injection_hook(
                        model, args.injection_layer, context_embedding, args.alpha
                    )
                    inputs = process_inputs(raw_image, prompt, processor, args.model_type)
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            do_sample=args.do_sample,
                            max_new_tokens=args.max_new_tokens,
                            repetition_penalty=args.repetition_penalty,
                        )
                    gen_ids = outputs[:, inputs["input_ids"].shape[-1] :]
                    text = processor.batch_decode(gen_ids[:, :64], skip_special_tokens=True)[0].strip()
                    hook_handle.remove()
            except Exception as e:
                logger.error("[%s] qid=%s gen: %s", strategy, qid, e)
                continue

            record = {
                "question_id": qid,
                "prompt": prompt,
                "text": text,
                "model_id": args.model_type,
                "image": img_file,
                "metadata": {},
            }
            fout.write(json.dumps(record) + "\n")
            fout.flush()
            os.fsync(fout.fileno())
            torch.cuda.empty_cache()
            gc.collect()

    logger.info("[%s] wrote %s", strategy, out_path)


def main():
    args = build_parser().parse_args()
    args = merge_json_config(args, args.config)
    if not args.question_dir or not args.image_folder:
        raise SystemExit("--question_dir and --image_folder are required")
    if not args.log_dir:
        raise SystemExit("--log_dir is required")

    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, f"log_pope_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())

    model, processor = load_model_and_processor(
        args.model_type, model_names, args.cache_dir, args.device, args.load_in_8bit
    )
    model.eval()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    with open(os.path.join(args.log_dir, "pope_run_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    for strat in STRATEGIES:
        run_strategy(strat, model, processor, args, logger, benchmark_file_tag(args.model_type))


if __name__ == "__main__":
    main()
