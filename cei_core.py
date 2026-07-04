"""
Shared CEI (Context Embedding Injection) implementation:
alpha mapping, injection hook, context embedding, two-pass dynamic decoding.
"""
import json
import math
from io import BytesIO

import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image

from model_utils import process_inputs


def alpha_from_mass(
    m,
    method="sigmoid",
    *,
    alpha_max=0.1,
    tau=0.2,
    T=0.05,
    beta=0.30,
):
    """Map mean Top-K mass m in [0,1] to injection strength alpha in [0, alpha_max]."""
    m = float(m)
    if method == "sigmoid":
        p = 1.0 / (1.0 + np.exp(-(tau - m) / max(1e-12, T)))
        return alpha_max * p
    if method == "cosine":
        if beta <= 0:
            raise ValueError("beta must be > 0 for cosine method.")
        z = max(0.0, min(1.0, m / beta))
        w = math.cos(0.5 * math.pi * z)
        return alpha_max * w
    raise ValueError("method must be 'sigmoid' or 'cosine'")


def setup_injection_hook(model, injection_layer, context_embedding, alpha, normalize_context=False, eps=1e-6):
    """Forward hook: mix last-position hidden state with context_embedding using alpha."""

    def injection_hook(module, input, output):
        hidden_states = output[0].clone()
        hs_last = hidden_states[:, -1, :]
        ctx = context_embedding.to(hs_last.device).type(hs_last.dtype)
        ctx = ctx.unsqueeze(0).expand_as(hs_last)
        if normalize_context:
            hs_norm = hs_last.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
            ctx_norm = ctx.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
            ctx = ctx * (hs_norm / ctx_norm)
        hidden_states[:, -1, :] = (1.0 - alpha) * hs_last + alpha * ctx
        return (hidden_states, output[1])

    return model.language_model.model.layers[injection_layer].register_forward_hook(injection_hook)


def load_image(image_file: str) -> Image.Image:
    if image_file.startswith("http://") or image_file.startswith("https://"):
        resp = requests.get(image_file, timeout=30)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    return Image.open(image_file).convert("RGB")


def get_context_embedding(raw_image, query, *, model, processor, model_type, ctx_layer: int, ctx_idx: int):
    """Single forward with hidden_states; return h[ctx_layer][0, ctx_idx, :]."""
    with torch.no_grad():
        inputs = process_inputs(raw_image, query, processor, model_type)
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        if model_type in ("llava", "llava-next"):
            hiddens = outputs["hidden_states"]
        else:
            hiddens = outputs["language_model_outputs"]["hidden_states"]

    layers = list(hiddens)
    L = len(layers)
    layer_idx = ctx_layer if ctx_layer >= 0 else (L + ctx_layer)
    if not (0 <= layer_idx < L):
        raise ValueError(f"context_layer={ctx_layer} out of range [0, {L - 1}]")

    hs = layers[layer_idx]
    T_tok = hs.shape[1]
    tok_idx = ctx_idx if ctx_idx >= 0 else (T_tok + ctx_idx)
    if not (0 <= tok_idx < T_tok):
        raise ValueError(f"context_idx={ctx_idx} out of range [0, {T_tok - 1}]")

    return hs[0, tok_idx, :].detach()


def generate_two_pass_dynamic(
    raw_image,
    query,
    *,
    model,
    processor,
    model_type,
    context_embedding,
    injection_layer,
    K_mass=40,
    start_layer=1,
    alpha_method="sigmoid",
    alpha_max=0.1,
    tau=0.2,
    T=0.05,
    beta=0.30,
    max_new_tokens=512,
    topK_mass_start_layer=-1,
    do_sample=False,
    logger=None,
    trace_path=None,
    question_id=None,
    image_id=None,
    delta=0.3,
    gamma=0.2,
    repetition_penalty=1.1,
    KV_cache=False,
):
    """
    Two-pass per token: probe (no injection) -> dynamic alpha -> inject -> next token.
    Optional JSONL trace at trace_path (question_id / image_id when provided).
    """
    if logger is not None:
        pass  # reserved for future logging hooks
    if do_sample:
        raise ValueError("do_sample=True is not supported in two_pass CEI (greedy only).")

    model.eval()
    lm = getattr(model, "language_model", None)
    if lm is None or not hasattr(lm, "lm_head"):
        raise RuntimeError("Expected `model.language_model.lm_head` for logit-lens.")
    lm_head = lm.lm_head

    tf = None
    trace_meta = None
    if trace_path is not None:
        tf = open(trace_path, "a", encoding="utf-8")
        trace_meta = {
            "model_type": model_type,
            "injection_layer": injection_layer,
            "K_mass": K_mass,
            "start_layer": start_layer,
        }
        if question_id is not None:
            trace_meta["question_id"] = question_id
        if image_id is not None:
            trace_meta["image_id"] = image_id

    supports_kv_cache = (model_type in ("llava", "llava-next")) and KV_cache

    inputs_base = process_inputs(raw_image, query, processor, model_type)
    generated = inputs_base["input_ids"]
    input_length = generated.shape[-1]

    attention_mask = inputs_base.get("attention_mask", None)
    if attention_mask is None:
        bsz, seq_len = generated.shape
        attention_mask = torch.ones((bsz, seq_len), device=generated.device, dtype=torch.long)

    current_input_ids = generated
    past_key_values = None
    eos_id = processor.tokenizer.eos_token_id

    def build_step_inputs(current_ids, attn_mask):
        step_inputs = {k: v for k, v in inputs_base.items()}
        step_inputs["input_ids"] = current_ids
        step_inputs["attention_mask"] = attn_mask
        return step_inputs

    def _convert_id_to_token_str(tid: int) -> str:
        if hasattr(processor.tokenizer, "convert_ids_to_tokens"):
            tok = processor.tokenizer.convert_ids_to_tokens([tid])[0]
        else:
            tok = processor.tokenizer.decode([tid], skip_special_tokens=False)
        return tok or ""

    def _is_word_start(tok_str: str) -> bool:
        return tok_str.startswith("▁") or tok_str.startswith("Ġ") or tok_str.startswith(" ")

    try:
        with torch.no_grad():
            for step in range(max_new_tokens):
                if supports_kv_cache and past_key_values is not None:
                    probe_input_ids = current_input_ids
                    inputs_p1 = build_step_inputs(probe_input_ids, attention_mask)
                    out1 = model(
                        **inputs_p1,
                        output_hidden_states=True,
                        return_dict=True,
                        use_cache=True,
                        past_key_values=past_key_values,
                    )
                else:
                    probe_input_ids = generated
                    inputs_p1 = build_step_inputs(probe_input_ids, attention_mask)
                    if supports_kv_cache:
                        out1 = model(
                            **inputs_p1, output_hidden_states=True, return_dict=True, use_cache=False
                        )
                    else:
                        out1 = model(**inputs_p1, output_hidden_states=True, return_dict=True)

                if model_type in ("llava", "llava-next"):
                    hidden_states = out1["hidden_states"]
                    final_logits = out1.logits[:, -1, :]
                else:
                    lm_out = out1["language_model_outputs"]
                    hidden_states = lm_out["hidden_states"]
                    final_logits = lm_out.logits[:, -1, :]

                final_probs = F.softmax(final_logits, dim=-1)
                top1_id_p1 = torch.argmax(final_probs, dim=-1)
                topk_idx = torch.topk(final_probs, k=K_mass, dim=-1).indices

                masses = []
                L = len(hidden_states)
                layer_start = (
                    topK_mass_start_layer
                    if (topK_mass_start_layer is not None and topK_mass_start_layer >= 0)
                    else start_layer
                )
                for layer_idx in range(layer_start, L):
                    hs = hidden_states[layer_idx][:, -1, :]
                    logits_l = lm_head(hs).float()
                    probs_l = F.softmax(logits_l, dim=-1)
                    mass_l = probs_l.gather(-1, topk_idx).sum(dim=-1)
                    masses.append(mass_l)

                masses = (
                    torch.stack(masses, dim=0)
                    if len(masses) > 0
                    else torch.zeros((1, final_probs.shape[0]), device=final_probs.device)
                )
                mean_mass = masses.mean(dim=0)
                sum_mass = masses.sum(dim=0)

                m = float(mean_mass[0].item())
                a = alpha_from_mass(
                    m,
                    method=alpha_method,
                    alpha_max=alpha_max,
                    tau=tau,
                    T=T,
                    beta=beta,
                )

                tok1_text = _convert_id_to_token_str(int(top1_id_p1[0].item()))
                if not _is_word_start(tok1_text):
                    a = a * float(delta)

                max_p1 = final_probs.max(dim=-1, keepdim=True).values
                thresh = float(gamma) * max_p1
                plausible_mask = final_probs >= thresh
                plausible_ids = torch.nonzero(plausible_mask[0], as_tuple=False)[:, 0]
                plausible_set = set(plausible_ids.tolist())

                hook_handle = setup_injection_hook(model, injection_layer, context_embedding, a)
                try:
                    inputs_p2 = build_step_inputs(current_input_ids, attention_mask)
                    if supports_kv_cache:
                        out2 = model(
                            **inputs_p2, return_dict=True, use_cache=True, past_key_values=past_key_values
                        )
                    else:
                        out2 = model(**inputs_p2, return_dict=True)
                finally:
                    hook_handle.remove()

                last_logits = out2.logits[:, -1, :]
                if repetition_penalty is not None and repetition_penalty > 1.0:
                    hist = generated[:, input_length:]
                    if hist.numel() > 0:
                        uniq = torch.unique(hist[0])
                        lv = last_logits[0]
                        for tid in uniq.tolist():
                            val = lv[tid]
                            lv[tid] = val / repetition_penalty if val > 0 else val * repetition_penalty

                next_id_p2 = int(torch.argmax(last_logits, dim=-1)[0].item())

                if next_id_p2 in plausible_set:
                    chosen_id = next_id_p2
                    chosen_source = "pass2"
                else:
                    chosen_id = int(top1_id_p1[0].item())
                    chosen_source = "pass1_fallback"

                prev_id = int(generated[0, -1].item())
                if chosen_id == prev_id:
                    _, idxs = torch.topk(last_logits[0], k=5)
                    picked = None
                    for cand in idxs.tolist():
                        if cand != prev_id and cand in plausible_set:
                            picked = cand
                            break
                    if picked is None:
                        picked = int(top1_id_p1[0].item())
                    chosen_id = picked

                next_token = torch.tensor([[chosen_id]], device=generated.device, dtype=generated.dtype)

                if tf is not None and trace_meta is not None:
                    tok_text = processor.tokenizer.decode([chosen_id], skip_special_tokens=False)
                    record = {
                        **trace_meta,
                        "step_idx": step,
                        "is_eos": bool(chosen_id == eos_id),
                        "token_id": chosen_id,
                        "token_text": tok_text,
                        "alpha": float(a),
                        "mean_topK_mass": float(m),
                        "sum_topK_mass": float(sum_mass[0].item()),
                        "within_word_gate_applied": bool(not _is_word_start(tok1_text)),
                        "gamma": float(gamma),
                        "choice": chosen_source,
                        "repetition_penalty": float(repetition_penalty),
                    }
                    json.dump(record, tf)
                    tf.write("\n")
                    tf.flush()

                if supports_kv_cache:
                    past_key_values = out2.past_key_values

                generated = torch.cat([generated, next_token], dim=-1)
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones(
                            (attention_mask.shape[0], 1),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                    ],
                    dim=-1,
                )
                current_input_ids = next_token if supports_kv_cache else generated

                if chosen_id == eos_id:
                    break

                torch.cuda.empty_cache()

        gen_only = generated[:, input_length:]
        gen_64 = gen_only[:, :64]
        gen_512 = gen_only[:, :512]
        caption_64 = processor.batch_decode(gen_64, skip_special_tokens=True)[0].strip()
        caption_512 = processor.batch_decode(gen_512, skip_special_tokens=True)[0].strip()
        return caption_64, caption_512
    finally:
        if tf is not None:
            tf.close()
