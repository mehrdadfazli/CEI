"""Load LVLMs and build model inputs (InstructBLIP, LLaVA-1.5, LLaVA-NeXT)."""
import logging

import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    InstructBlipForConditionalGeneration,
    InstructBlipProcessor,
    LlavaForConditionalGeneration,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
)

logger = logging.getLogger(__name__)

MODEL_IDS = {
    "instructblip": "Salesforce/instructblip-vicuna-7b",
    "llava": "llava-hf/llava-1.5-7b-hf",
    "llava-next": "llava-hf/llava-v1.6-vicuna-7b-hf",
}

# Backward-compatible alias for runners
model_names = MODEL_IDS


def load_model_and_processor(model_type, model_ids, cache_dir, device, load_in_8bit=True):
    """Load HF model + processor for supported LVLMs."""
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=load_in_8bit,
        llm_int8_threshold=200.0,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    if model_type == "instructblip":
        model = InstructBlipForConditionalGeneration.from_pretrained(
            model_ids[model_type],
            torch_dtype=torch.float16,
            attn_implementation="eager",
            cache_dir=cache_dir,
            quantization_config=quantization_config,
            device_map="auto",
        )
        model.tie_weights()
        processor = InstructBlipProcessor.from_pretrained(model_ids[model_type], cache_dir=cache_dir)
    elif model_type == "llava":
        model = LlavaForConditionalGeneration.from_pretrained(
            model_ids[model_type],
            torch_dtype=torch.float16,
            attn_implementation="eager",
            cache_dir=cache_dir,
            quantization_config=quantization_config,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(model_ids[model_type], cache_dir=cache_dir)
        processor.patch_size = model.config.vision_config.patch_size
        processor.vision_feature_select_strategy = model.config.vision_feature_select_strategy
    elif model_type == "llava-next":
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_ids[model_type],
            torch_dtype=torch.float16,
            attn_implementation="eager",
            cache_dir=cache_dir,
            quantization_config=quantization_config,
            device_map="auto",
        )
        processor = LlavaNextProcessor.from_pretrained(model_ids[model_type], cache_dir=cache_dir)
        processor.patch_size = model.config.vision_config.patch_size
        processor.vision_feature_select_strategy = model.config.vision_feature_select_strategy
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    logger.info("Loaded %s", model_type)
    return model, processor


def process_inputs(raw_image, query, processor, model_type, device="cuda"):
    """Tokenize image + text for the given LVLM."""
    if model_type in ("llava", "llava-next"):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": query},
                ],
            }
        ]
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        return processor(images=raw_image, text=text_prompt, padding=True, return_tensors="pt").to(
            device, torch.float16
        )
    if model_type == "instructblip":
        return processor(images=raw_image, text=query, return_tensors="pt").to(device)
    raise ValueError(f"Unsupported model type: {model_type}")
