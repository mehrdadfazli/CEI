import argparse
import os
import json
import random
import torch
from PIL import Image
from tqdm import tqdm
import logging
from datetime import datetime
import numpy as np
import gc

from model_utils import load_model_and_processor, process_inputs
from CEI_utils import setup_injection_hook

# Set up argument parser
parser = argparse.ArgumentParser(description="Run CEI on CHAIR benchmark")
parser.add_argument("--model_type", default="instructblip", choices=["instructblip", "llava"], help="Model type")
parser.add_argument("--load_in_8bit", action="store_true", default=True, help="Load model in 8-bit precision")
parser.add_argument("--cache_dir", default="/scratch/mfazli/huggingface_cache", help="Cache directory for model")
parser.add_argument("--log_dir", default="./results/CHAIR", help="Directory for logs and results")
parser.add_argument("--use_CEI", action="store_true", default=False, help="Use CEI for hallucination mitigation")
parser.add_argument("--do_sample", action="store_true", default=False, help="Use sampling for generation")
parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for generation")
parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum new tokens to generate")

parser.add_argument("--context_embedding_idx", type=int, default=-1, help="Reference input token index for context embedding")
parser.add_argument("--context_embedding_layer", type=int, default=31, help="Reference layer for context embedding")
parser.add_argument("--injection_layer", type=int, default=10, help="Layer to inject context embedding")
parser.add_argument("--alpha", type=float, default=0.1, help="Weighting factor for context embedding")

parser.add_argument("--opera_results", action="store_true", default=False, help="Whether to use OPERA results")
parser.add_argument("--data_path", type=str, default="../CAG/datasets/coco2014/val2014/", help="Path to image dataset")
parser.add_argument("--num_images", type=int, default=500, help="Number of images to process")
parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
args = parser.parse_args()


def load_opera_image_ids(opera_results_path):
    image_ids = []
    try:
        with open(opera_results_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                image_ids.append(data['image_id'])
        return image_ids
    except Exception as e:
        logging.error(f"Error reading OPERA results file {opera_results_path}: {e}")
        raise


def main():
    # Set up logging
    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info(f"use_CEI is set to: {args.use_CEI}")

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Experiment configuration
    exp_config = {
        "context_embedding_idx": args.context_embedding_idx,
        "context_embedding_layer": args.context_embedding_layer,
        "injection_layer": args.injection_layer,
        "alpha": args.alpha,
        "max_new_tokens": args.max_new_tokens
    }

    # Generate random exp_id
    exp_id = np.random.randint(1000, 9999)

    # Save experiment configuration
    config_path = os.path.join(args.log_dir, f"{exp_id}_config.json")
    with open(config_path, 'w') as f:
        json.dump(exp_config, f, indent=4)
    logger.info(f"Saved experiment configuration to {config_path}")

    # Set up output file
    output_file = os.path.join(args.log_dir, f"{exp_id}_{args.model_type}.jsonl")
    logger.info(f"Output will be saved to {output_file}")

    # Load model and processor
    model_names = {
        "instructblip": "Salesforce/instructblip-vicuna-7b",
        "llava": "llava-hf/llava-1.5-7b-hf"
    }
    model, processor = load_model_and_processor(args.model_type, model_names, args.cache_dir, device, args.load_in_8bit)

    
    if args.opera_results:
        if args.model_type == "llava":
            opera_results_path = "/projects/zzhu20/Mehrdad/CAG/results/CHAIR/OPERA/llava/ours.jsonl"
        elif args.model_type == "instructblip":
            opera_results_path = "/projects/zzhu20/Mehrdad/CAG/results/CHAIR/OPERA/instructblip/ours.jsonl"

        image_ids = load_opera_image_ids(opera_results_path)
        image_list = [(f"COCO_val2014_{id:012d}.jpg", id) for id in image_ids]
        logger.info(f"Loaded {len(image_list)} image IDs from OPERA results")
    else:
        raise NotImplementedError("Failed to load images from OPERA results")
        img_files = [f for f in os.listdir(args.data_path) if f.endswith('.jpg')]
        image_list = [(f, int(f.split(".jpg")[0][-6:])) for f in img_files]
        random.seed(args.random_seed)
        random.shuffle(image_list)
        if args.num_images is not None:
            image_list = image_list[:args.num_images]
        logger.info(f"Processing {len(image_list)} images from {args.data_path}")    
    
    # Load image list
    # img_files = [f for f in os.listdir(args.data_path) if f.endswith('.jpg')]
    # random.seed(args.random_seed)
    # random.shuffle(img_files)
    # img_files = img_files[:args.num_images]
    # image_list = [(f, int(f.split('_')[-1].split('.')[0])) for f in img_files]
    # logger.info(f"Processing {len(image_list)} images")

    # Process each image
    for img_file, img_id in tqdm(image_list, desc="Processing images"):
        img_path = os.path.join(args.data_path, img_file)
        try:
            raw_image = Image.open(img_path).convert('RGB')
            query = "Describe this image."

            if args.use_CEI:
                # Get context embedding
                inputs = process_inputs(raw_image, query, processor, args.model_type)
                batch_size, seq_len = inputs["input_ids"].shape
                inputs["attention_mask"] = torch.ones((batch_size, seq_len), device=inputs["input_ids"].device, dtype=torch.long)
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True, return_dict=True)
                if args.model_type == "llava":
                    hidden_states = outputs['hidden_states']
                else:
                    hidden_states = outputs['language_model_outputs']['hidden_states']
                context_embedding = hidden_states[exp_config["context_embedding_layer"]][0, exp_config["context_embedding_idx"], :]
                # Set up injection hook
                hook_handle = setup_injection_hook(model, exp_config["injection_layer"], context_embedding, exp_config["alpha"])

            # Process inputs for generation
            inputs = process_inputs(raw_image, query, processor, args.model_type)
            # Generate caption
            with torch.no_grad():
                outputs = model.generate(**inputs, do_sample=args.do_sample, max_new_tokens=exp_config["max_new_tokens"], num_beams=args.num_beams)
            generated_ids = outputs[:, inputs['input_ids'].shape[-1]:]
            caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

            # Remove hook if CEI is used
            if args.use_CEI:
                hook_handle.remove()

            # Save result
            result = {
                "image_id": img_id,
                "caption": caption
            }
            with open(output_file, "a") as f:
                json.dump(result, f)
                f.write('\n')

            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            logger.error(f"Error processing image {img_file}: {e}")
            continue

    logger.info(f"Evaluation complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()