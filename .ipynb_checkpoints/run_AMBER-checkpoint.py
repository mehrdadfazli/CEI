import os
import gc
import re
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from io import BytesIO
import requests
from collections import defaultdict
# import seaborn as sns
# import matplotlib.pyplot as plt
import spacy
from IPython.display import display, HTML
from nltk.corpus import wordnet as wn
import inflect
from tqdm import tqdm
import argparse
import logging

from model_utils import load_model_and_processor, process_inputs
from CEI_utils import setup_injection_hook

def recorder(out):
    NEG_WORDS = ["No", "not", "no", "NO"]

    out = out.replace('.', '')
    out = out.replace(',', '')
    words = out.split(' ')
    if any(word in NEG_WORDS for word in words) or any(word.endswith("n't") for word in words):
        return "No"
    else:
        return "Yes"

parser = argparse.ArgumentParser(description="Run AMBER benchmark with hallucination mitigation")
parser.add_argument("--model_type", default="instructblip", choices=["instructblip", "llava"], help="Model type")
parser.add_argument("--load_in_8bit", action="store_true", default=True, help="Load model in 8-bit precision")
parser.add_argument("--cache_dir", default="/scratch/mfazli/huggingface_cache", help="Cache directory for model")
parser.add_argument("--amber_path", default="/projects/zzhu20/Mehrdad/AMBER", help="Path to AMBER dataset")
parser.add_argument("--log_dir", default="/projects/zzhu20/Mehrdad/CEI/results/AMBER/", help="Directory for logs and results")
parser.add_argument("--use_CEI", action="store_true", default=False, help="Use CEI for hallucination mitigation")
parser.add_argument("--do_sample", action="store_true", default=False, help="Use sampling for generation")
parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for generation")
parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum new tokens to generate")

parser.add_argument("--context_embedding_idx", type=int, default=-1, help="reference input token index for context embedding")
parser.add_argument("--context_embedding_layer", type=int, default=31, help="reference layer for context embedding")
parser.add_argument("--injection_layer", type=int, default=15, help="layer to inject context embedding")
parser.add_argument("--alpha", type=float, default=0.1, help="weighting factor for context embedding")

args = parser.parse_args()

# Derived paths
os.makedirs(args.log_dir, exist_ok=True)
EXP_ID = np.random.randint(1000,9999)
JSON_QUERY_PATH = os.path.join(args.amber_path, "data/query/query_all.json")
JSON_ANNOTATION_PATH = os.path.join(args.amber_path, "data/annotations.json")
IMAGE_DIR = os.path.join(args.amber_path, "image")
EXP_CONFIG_PATH = os.path.join(args.log_dir, f"{args.model_type}_{EXP_ID}_config.json")
RESPONSES_PATH = os.path.join(args.log_dir, f"{args.model_type}_{EXP_ID}_responses.json")


LOG_FILE = os.path.join(args.log_dir, f"log_{args.model_type}_{EXP_ID}.log")
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info(f"use_CEI is set to: {args.use_CEI}")


# Model names
model_names = {
    "instructblip": "Salesforce/instructblip-vicuna-7b",
    "llava": "llava-hf/llava-1.5-7b-hf"
}

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Experiment configuration
EXP_CONFIG = {
    "context_embedding_idx": args.context_embedding_idx,
    "context_embedding_layer": args.context_embedding_layer,
    "injection_layer": args.injection_layer,
    "alpha": args.alpha,
    "max_new_tokens": args.max_new_tokens
}

# Save EXP_CONFIG to JSON
with open(EXP_CONFIG_PATH, 'w') as file:
    json.dump(EXP_CONFIG, file, indent=4)
logger.info(f"Saved experiment configuration to {EXP_CONFIG_PATH}")

def main():
    """
    Main function to run the AMBER benchmark.
    """
    # Load model and processor
    model, processor = load_model_and_processor(args.model_type, model_names, args.cache_dir, device, args.load_in_8bit)
    tokenizer = processor.tokenizer
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Load dataset and annotations
    with open(JSON_QUERY_PATH, 'r') as file:
        data = json.load(file)
    with open(JSON_ANNOTATION_PATH, 'r') as file:
        annotations = json.load(file)
    logger.info("Loaded dataset and annotations")
    
    # Initialize responses
    responses = []

    
    # Process dataset
    for item in tqdm(data, desc="Processing Dataset"):
        if item['id'] > 1004:
            EXP_CONFIG["max_new_tokens"] = 10

        # if item['id'] in processed_ids:
        #     continue
        image_id = item['id']
        image_file = item['image']
        img_path = os.path.join(IMAGE_DIR, image_file)
        try:
            raw_image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            continue
        query = item['query']

        if args.use_CEI:

            inputs = process_inputs(raw_image, query, processor, args.model_type)
            batch_size, seq_len = inputs["input_ids"].shape
            inputs["attention_mask"] = torch.ones((batch_size, seq_len), device=inputs["input_ids"].device, dtype=torch.long)
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)

            if args.model_type == "llava":
                hidden_states = outputs['hidden_states']
            else:
                hidden_states = outputs['language_model_outputs']['hidden_states']

            context_embedding = hidden_states[EXP_CONFIG["context_embedding_layer"]][0, EXP_CONFIG["context_embedding_idx"],:]

            hook_handle = setup_injection_hook(model, EXP_CONFIG["injection_layer"], context_embedding, EXP_CONFIG["alpha"])
        
        inputs = process_inputs(raw_image, query, processor, args.model_type)

        # Run inference
        with torch.no_grad():
            outputs = model.generate(**inputs, do_sample=args.do_sample, max_new_tokens=args.max_new_tokens, num_beams=args.num_beams)
        
        generated_ids = outputs[:, inputs['input_ids'].shape[-1]:] #- remove the inputs from output sequence

        # Remove the hook after generation if CEI is enabled
        if args.use_CEI:
            hook_handle.remove()

        if item['id'] <= 1004:
            response_text_64 = processor.batch_decode(generated_ids[..., :64], skip_special_tokens=True)[0].strip()
            response_text_128 = processor.batch_decode(generated_ids[..., :128], skip_special_tokens=True)[0].strip()
            response_text_256 = processor.batch_decode(generated_ids[..., :256], skip_special_tokens=True)[0].strip()
            response_text_512 = processor.batch_decode(generated_ids[..., :512], skip_special_tokens=True)[0].strip()
            
            responses.append({
                'id': image_id,
                'response_64': response_text_64,
                'response_128': response_text_128,
                'response_256': response_text_256,
                'response_512': response_text_512,
                'response_length': generated_ids.shape[-1]
            })
        else:
            response_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            response_text = recorder(response_text)

            responses.append({
                'id': image_id,
                'response': response_text,
                'response_length': generated_ids.shape[-1]
            })
        
        with open(RESPONSES_PATH, 'w') as file:
            json.dump(responses, file, indent=4)
        logger.info(f"Saved responses to {RESPONSES_PATH}")
        
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()
