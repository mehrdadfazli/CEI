import argparse
import os
import json
import random
import torch
from model_utils import load_model_and_processor, process_inputs
from CEIdyn_utils import setup_cei_hooks
from PIL import Image
from tqdm import tqdm

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
    """Run dynamic CEI on the CHAIR benchmark to generate image captions."""
    parser = argparse.ArgumentParser(description="Run dynamic CEI on CHAIR benchmark")
    parser.add_argument("--model_type", default="instructblip", choices=["instructblip", "llava"], help="Model type")
    parser.add_argument("--load_in_8bit", action="store_true", default=True, help="Load model in 8-bit precision")
    parser.add_argument("--cache_dir", default="/scratch/mfazli/huggingface_cache", help="Cache directory for model")
    parser.add_argument("--log_dir", default="./results/CHAIR/prelim_dynamic_ctx_embed", help="Directory for logs and results")
    parser.add_argument("--use_CEI", action="store_true", default=True, help="Enable dynamic CEI")
    parser.add_argument("--do_sample", action="store_true", default=False, help="Use sampling for generation")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for generation")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum new tokens to generate")
    parser.add_argument("--injection_layer", type=int, default=10, help="Layer for context injection")
    parser.add_argument("--alpha", type=float, default=0.1, help="Injection weight (0-1)")
    parser.add_argument("--context_strategy", default="top1", choices=["top1", "weighted_avg", "topk_avg"], help="Context embedding selection strategy")
    parser.add_argument("--topk", type=int, default=3, help="Number of top tokens for topk_avg strategy")
    parser.add_argument("--opera_results", type=str, default=None, help="Path to OPERA results JSONL file to read image IDs from")
    parser.add_argument("--data_path", type=str, default="../CAG/datasets/coco2014/val2014/", help="Path to images")
    parser.add_argument("--num_images", type=int, default=500, help="Number of images to process")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Set up logging and device
    os.makedirs(args.log_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and processor
    MODEL_MAP = {
        "instructblip": "Salesforce/instructblip-vicuna-7b",
        "llava": "llava-hf/llava-1.5-7b-hf"
    }
    model, processor = load_model_and_processor(
        args.model_type, MODEL_MAP, args.cache_dir, device, args.load_in_8bit
    )

    # Prepare image list
    if args.opera_results:
            image_ids = load_opera_image_ids(args.opera_results)
            image_list = [(f"COCO_val2014_{id:012d}.jpg", id) for id in image_ids]        
    else:
        img_files = [f for f in os.listdir(args.data_path) if f.endswith('.jpg')]
        random.seed(args.random_seed)
        random.shuffle(img_files)
        img_files = img_files[:args.num_images]
        image_list = [(f, int(f.split('_')[-1].split('.')[0])) for f in img_files]

    # Process each image
    for img_file, img_id in tqdm(image_list, desc="Processing images"):
        img_path = os.path.join(args.data_path, img_file)
        raw_image = Image.open(img_path).convert('RGB')
        query = "Describe this image."

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

                # Handle QFormer text inputs
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
                batch_size = 1  # Single-image processing
                hidden_dim = model.language_model.config.hidden_size  # Correct hidden dimension
                hook_last, hook_injection = setup_cei_hooks(
                    model, batch_size, hidden_dim, visual_token_embeds, args.injection_layer, 
                    device, args.alpha, context_strategy=args.context_strategy, topk=args.topk
                )

        # Generate caption
        with torch.no_grad():
            gen_outputs = model.generate(
                **inputs, do_sample=args.do_sample, max_new_tokens=args.max_new_tokens, num_beams=args.num_beams, use_cache=True
            )
        generated_ids = gen_outputs[:, inputs['input_ids'].shape[-1]:]
        caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        # Clean up hooks
        if hook_last and hook_injection:
            hook_last.remove()
            hook_injection.remove()

        # Save result
        result = {"image_id": img_id, "caption": caption}
        output_path = os.path.join(args.log_dir, f"dynamic_CEI_results_alpha_{args.alpha}_layer_{args.injection_layer}_{args.context_strategy}.jsonl")
        with open(output_path, 'a') as f:
            json.dump(result, f)
            f.write("\n")

        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
