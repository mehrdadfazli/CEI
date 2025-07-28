import os
import gc
import json
import torch
from PIL import Image
from tqdm import tqdm
import argparse
import logging
from datetime import datetime
from model_utils import load_model_and_processor, process_inputs
from CEIdyn_utils import setup_cei_hooks

def recorder(out):
    """
    Convert model output to 'Yes' or 'No' based on presence of negative words.
    
    Args:
        out (str): Generated text.
    
    Returns:
        str: 'Yes' or 'No'.
    """
    NEG_WORDS = ["No", "not", "no", "NO"]
    out = out.replace('.', '').replace(',', '')
    words = out.split()
    if any(word in NEG_WORDS for word in words) or any(word.endswith("n't") for word in words):
        return "No"
    return "Yes"

def main():
    """Run dynamic CEI on the AMBER benchmark to generate image captions."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run AMBER benchmark with dynamic CEI")
    parser.add_argument("--model_type", default="instructblip", choices=["instructblip", "llava"], help="Model type")
    parser.add_argument("--load_in_8bit", action="store_true", default=True, help="Load model in 8-bit precision")
    parser.add_argument("--cache_dir", default="/scratch/mfazli/huggingface_cache", help="Cache directory for model")
    parser.add_argument("--amber_path", default="/projects/zzhu20/Mehrdad/AMBER", help="Path to AMBER dataset")
    parser.add_argument("--log_dir", default="./results/AMBER/prelim_dynamic_ctx_embed/", help="Directory for logs and results")
    parser.add_argument("--use_CEI", action="store_true", default=True, help="Enable dynamic CEI")
    parser.add_argument("--do_sample", action="store_true", default=False, help="Use sampling for generation")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for generation")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum new tokens to generate")
    parser.add_argument("--injection_layer", type=int, default=10, help="Layer for context injection")
    parser.add_argument("--alpha", type=float, default=0.1, help="Injection weight (0-1)")
    parser.add_argument("--context_strategy", default="top1", choices=["top1", "weighted_avg", "topk_avg"], help="Context embedding selection strategy")
    parser.add_argument("--topk", type=int, default=3, help="Number of top tokens for topk_avg strategy")
    args = parser.parse_args()

    # Set up logging
    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Derived paths
    exp_id = torch.randint(1000, 10000, (1,)).item()
    json_query_path = os.path.join(args.amber_path, "data/query/query_all.json")
    json_annotation_path = os.path.join(args.amber_path, "data/annotations.json")
    image_dir = os.path.join(args.amber_path, "image")
    responses_path = os.path.join(args.log_dir, f"{args.model_type}_{exp_id}_alpha{args.alpha}_layer_{args.injection_layer}_strategy_{args.context_strategy}.jsonl")

    # Model names
    model_names = {
        "instructblip": "Salesforce/instructblip-vicuna-7b",
        "llava": "llava-hf/llava-1.5-7b-hf"
    }

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    try:
        # Load model and processor
        model, processor = load_model_and_processor(args.model_type, model_names, args.cache_dir, device, args.load_in_8bit)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Load dataset and annotations
        with open(json_query_path, 'r') as file:
            data = json.load(file)
        with open(json_annotation_path, 'r') as file:
            annotations = json.load(file)
        logger.info("Loaded dataset and annotations")
        
        # Load or initialize responses
        responses = []
        # if os.path.exists(responses_path):
        #     try:
        #         with open(responses_path, 'r') as file:
        #             responses = json.load(file)
        #         logger.info(f"Loaded existing responses from {responses_path}")
        #     except Exception as e:
        #         logger.error(f"Error reading responses file: {e}")
        
        # processed_ids = set(item['id'] for item in responses)
        
        # Process dataset
        for item in tqdm(data[:1004], desc="Processing Dataset"):
            max_new_tokens = 10 if item['id'] > 1004 else args.max_new_tokens
            # if item['id'] in processed_ids:
            #     continue
            image_id = item['id']
            image_file = item['image']
            img_path = os.path.join(image_dir, image_file)
            try:
                raw_image = Image.open(img_path).convert('RGB')
            except Exception as e:
                logger.error(f"Error loading image {img_path}: {e}")
                continue
            query = item['query']

            # Preprocess inputs
            inputs = process_inputs(raw_image, query, processor, args.model_type)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Set up dynamic CEI if enabled
            hook_last, hook_injection = None, None
            if args.use_CEI and args.model_type == "instructblip":
                with torch.no_grad():
                    # Encode image
                    vision_outputs = model.vision_model(pixel_values=inputs['pixel_values'], return_dict=True)
                    image_embeds = vision_outputs.last_hidden_state

                    # Prepare QFormer inputs
                    query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
                    query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=device)
                    qformer_input_ids = inputs.get('qformer_input_ids')
                    qformer_attention_mask = inputs.get('qformer_attention_mask')
                    if qformer_input_ids is not None:
                        qformer_attention_mask = torch.cat([query_attention_mask, qformer_attention_mask], dim=1)
                    else:
                        qformer_attention_mask = query_attention_mask

                    # Pass through QFormer
                    query_outputs = model.qformer(
                        input_ids=qformer_input_ids,
                        attention_mask=qformer_attention_mask,
                        query_embeds=query_tokens,
                        encoder_hidden_states=image_embeds,
                        encoder_attention_mask=torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device),
                        return_dict=True,
                    )
                    query_output = query_outputs.last_hidden_state[:, :query_tokens.size(1), :]

                    # Project to language model space
                    visual_token_embeds = model.language_projection(query_output)

                    # Configure CEI hooks
                    batch_size = 1
                    hidden_dim = model.language_model.config.hidden_size
                    hook_last, hook_injection = setup_cei_hooks(
                        model, batch_size, hidden_dim, visual_token_embeds, args.injection_layer, 
                        device, args.alpha, context_strategy=args.context_strategy, topk=args.topk
                    )

            # Generate caption
            try:
                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs, do_sample=args.do_sample, max_new_tokens=max_new_tokens, num_beams=args.num_beams, use_cache=True
                    )
                generated_ids = generated_ids[:, inputs['input_ids'].shape[-1]:]
            except Exception as e:
                logger.error(f"Error generating for image {img_path}: {e}")
                continue

            # Clean up hooks
            if hook_last and hook_injection:
                hook_last.remove()
                hook_injection.remove()

            # Process response
            if image_id <= 1004:
                response_text_64 = processor.batch_decode(generated_ids[:, :64], skip_special_tokens=True)[0].strip()
                response_text_128 = processor.batch_decode(generated_ids[:, :128], skip_special_tokens=True)[0].strip()
                response_text_256 = processor.batch_decode(generated_ids[:, :256], skip_special_tokens=True)[0].strip()
                response_text_512 = processor.batch_decode(generated_ids[:, :512], skip_special_tokens=True)[0].strip()
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

            # Save responses
            # processed_ids.add(image_id)
            try:
                with open(responses_path, 'w') as file:
                    json.dump(responses, file, indent=4)
                logger.info(f"Saved responses to {responses_path}")
            except Exception as e:
                logger.error(f"Error saving responses: {e}")
            torch.cuda.empty_cache()
            gc.collect()
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()