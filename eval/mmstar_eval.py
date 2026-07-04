"""
Score MMStar JSONL outputs from inference_editing.py (multiple-choice accuracy).

Supports two scoring modes:

- **exact**: VLMEvalKit-style `can_infer` heuristics only (no API calls).
- **vlmeval**: Same as VLMEvalKit `mcq_vanilla_eval` with `model != None`: run `can_infer`
  on the model output first; if that fails, call an OpenAI chat model with the official
  judge prompt from VLMEvalKit (`multiple_choice.build_prompt`), then parse with
  `can_infer` again (see OpenCompass VLMEvalKit).

Requires `OPENAI_API_KEY` in the environment when using `--scoring vlmeval`.
"""
import argparse
import copy as cp
import json
import os
import re
import string
import time
from collections import defaultdict

import jsonlines
import pandas as pd


def parse_mcq_choices_from_question(question):
    """Parse 'Options: A: ..., B: ...' into a dict of letter -> option text."""
    choices = {}
    if "Options:" not in question:
        return choices
    rest = question.split("Options:", 1)[1].strip()
    segments = re.split(r",\s*(?=[A-D]:)", rest)
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        m = re.match(r"^([A-D]):\s*(.+)$", seg, re.DOTALL)
        if m:
            choices[m.group(1)] = m.group(2).strip().rstrip(".")
    return choices


def question_stem_only(full_question):
    """Stem without the Options block (for VLMEvalKit judge prompt)."""
    if "Options:" in full_question:
        return full_question.split("Options:", 1)[0].strip()
    return full_question.strip()


def build_option_str(choices):
    """VLMEvalKit-style compact option line (see `build_option_str` in VLMEvalKit)."""
    if not choices:
        return ""
    return " ".join(f"{k}. {choices[k]}" for k in sorted(choices.keys()))


def vlmeval_build_prompt(question_stem, option_str, prediction):
    """Copy of VLMEvalKit `build_prompt` for English MCQ judge (multiple_choice.py)."""
    tmpl = (
        "You are an AI assistant who will help me to match "
        "an answer with several options of a single-choice question. "
        "You are provided with a question, several options, and an answer, "
        "and you need to find which option is most similar to the answer. "
        "If the meaning of all options are significantly different from the answer, output Z. "
        "Your should output a single uppercase character in A, B, C, D (if they are valid options), and Z. \n"
        "Example 1: \n"
        "Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n"
        "Answer: a cute teddy bear\nYour output: A\n"
        "Example 2: \n"
        "Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n"
        "Answer: Spider\nYour output: Z\n"
        "Example 3: \n"
        "Question: {}?\nOptions: {}\nAnswer: {}\nYour output: "
    )
    stem = question_stem.strip().rstrip("?").strip()
    return tmpl.format(stem, option_str, prediction)


def can_infer_option(answer, choices):
    """Match a single option letter in the model output (VLMEvalKit-style)."""
    if "Failed to obtain answer via API" in answer:
        return False

    reject_to_answer = [
        "Sorry, I can't help with images of people yet.",
        "I can't process this file.",
        "I'm sorry, but without the image provided",
        "Cannot determine the answer",
    ]
    for err in reject_to_answer:
        if err in answer:
            return "Z"

    def count_choice(splits, chs, prefix="", suffix=""):
        cnt = 0
        for c in chs:
            if prefix + c + suffix in splits:
                cnt += 1
        return cnt

    answer_mod = cp.copy(answer)
    for c in ".()[],:;!*#{}":
        answer_mod = answer_mod.replace(c, " ")
    splits = [x.strip() for x in answer_mod.split()]
    count = count_choice(splits, choices)

    if count == 1:
        for ch in choices:
            if ch in splits and splits.index(ch) > (len(splits) - 5):
                return ch
    elif count == 0 and count_choice(splits, {"Z", ""}) == 1:
        return "Z"
    return False


def can_infer_text(answer, choices):
    answer_l = answer.lower()
    if len(answer_l) > 2 * sum(len(str(v)) for v in choices.values()):
        return False
    choices_l = {k: str(v).lower() for k, v in choices.items()}
    cands = []
    for k, v in choices_l.items():
        if v in answer_l:
            cands.append(k)
    if len(cands) == 1:
        return cands[0]
    return False


def can_infer(answer, choices):
    answer = str(answer)
    copt = can_infer_option(answer, choices)
    return copt if copt else can_infer_text(answer, choices)


def infer_prediction_exact_only(response, choices):
    """Heuristic path used when `--scoring exact` (and as prefetch before GPT)."""
    if not choices:
        letters = list(string.ascii_uppercase[:4])
        copt = can_infer_option(response, letters)
        if copt and copt in string.ascii_uppercase:
            return str(copt)
        m = re.findall(r"\b([A-D])\b", response.upper())
        if len(m) == 1:
            return m[0]
        if m:
            return m[-1]
        return None

    pred = can_infer(response, choices)
    if pred in (False, None):
        pred = None
    elif pred == "Z":
        pred = None
    else:
        pred = str(pred).strip().upper()[:1]
    if pred is None:
        m = re.findall(r"\b([A-D])\b", response.upper())
        if len(m) == 1:
            return m[0]
        if m:
            return m[-1]
    return pred


def extract_answer_vlmeval_item(
    prediction,
    choices,
    question_full,
    *,
    judge_generate,
    retries=3,
    judge_sleep=0.0,
):
    """
    Mirrors VLMEvalKit `extract_answer_from_item` for vanilla MCQ:
    prefetch with `can_infer(prediction)`; if missing, query GPT judge; parse judge output.
    """
    option_str = build_option_str(choices)
    stem = question_stem_only(question_full)

    prefetch = can_infer(prediction, choices) if choices else False
    if prefetch:
        return str(prefetch).strip().upper()[:1], "prefetch"

    if judge_generate is None:
        return None, "exact_fail"

    prompt = vlmeval_build_prompt(stem, option_str, prediction)
    attempt = retries
    last_ans = None
    while attempt > 0:
        ans = judge_generate(prompt)
        if judge_sleep > 0:
            time.sleep(judge_sleep)
        if ans and "Failed to obtain answer via API" in ans:
            attempt -= 1
            continue
        last_ans = ans
        ret = can_infer(ans, choices) if choices else None
        if not choices:
            m = re.findall(r"\b([A-D])\b", (ans or "").upper())
            ret = m[0] if len(m) == 1 else (m[-1] if m else None)
        if ret:
            return str(ret).strip().upper()[:1], "gpt"
        attempt -= 1

    # Lenient fallback: GPT responded but strict heuristic couldn't parse it.
    # Take any A-D letter that appears in the response (first occurrence).
    if last_ans:
        m = re.findall(r"\b([A-D])\b", last_ans.upper())
        if m:
            return m[0], "gpt_lenient"

    return "Z", "gpt_fail"


def make_openai_judge(model_name, api_key, base_url=None, default_headers=None):
    """Returns a function prompt -> assistant text. Supports openai v0.x and v1.x."""
    try:
        from openai import OpenAI  # v1.x
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        if default_headers:
            kwargs["default_headers"] = default_headers
        client = OpenAI(**kwargs)

        def judge_generate(prompt: str) -> str:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=64,
            )
            return (resp.choices[0].message.content or "").strip()

    except ImportError:
        import openai as _oa  # v0.x
        _oa.api_key = api_key
        if base_url:
            _oa.api_base = base_url

        def judge_generate(prompt: str) -> str:
            resp = _oa.ChatCompletion.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=64,
            )
            return (resp["choices"][0]["message"]["content"] or "").strip()

    return judge_generate


def report_acc_mmstar(df):
    """Overall + per-category accuracy (VLMEvalKit report_acc pattern)."""
    res = defaultdict(list)
    if "split" in df.columns:
        splits = sorted(set(df["split"]))
        res["split"] = splits
    else:
        df = df.copy()
        df["split"] = "none"
        splits = ["none"]
        res["split"] = splits

    for group in [None, "l2_category", "category"]:
        if group is None:
            for sp in splits:
                sub = df[df["split"] == sp]
                res["Overall"].append(float(sub["hit"].mean()) if len(sub) else 0.0)
        elif group not in df.columns:
            continue
        else:
            abilities = sorted(set(df[group].dropna()))
            for ab in abilities:
                sub_g = df[df[group] == ab]
                col_name = ab if len(ab) <= 32 else ab[:29] + "..."
                for sp in splits:
                    sub = sub_g[sub_g["split"] == sp]
                    res[col_name].append(float(sub["hit"].mean()) if len(sub) else float("nan"))
    return pd.DataFrame(res)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cap_file",
        type=str,
        required=True,
        help="JSONL from inference_editing (e.g. results/MMStar/...jsonl)",
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default=None,
        help="Optional path to write accuracy summary CSV",
    )
    parser.add_argument(
        "--summary_file",
        type=str,
        default=None,
        help="Optional JSON path: overall accuracy, scoring mode, and breakdown table (records).",
    )
    parser.add_argument(
        "--scoring",
        type=str,
        choices=["exact", "vlmeval"],
        default="exact",
        help=(
            "exact: heuristic matching only. "
            "vlmeval: prefetch then OpenAI judge on failures (VLMEvalKit mcq_vanilla_eval pattern)."
        ),
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=os.environ.get("MMSTAR_JUDGE_MODEL", "gpt-4o-mini"),
        help="Chat model for --scoring vlmeval (VLMEvalKit often uses gpt-4-0125 / chatgpt-0125).",
    )
    parser.add_argument(
        "--api-provider",
        type=str,
        choices=["openai", "openrouter", "custom"],
        default="openai",
        help="LLM API provider for judge calls. Use openrouter for OpenRouter-hosted GPT-4o.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key override. Falls back to OPENAI_API_KEY, then OPENROUTER_API_KEY.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL override (e.g. https://openrouter.ai/api/v1).",
    )
    parser.add_argument(
        "--openrouter-http-referer",
        type=str,
        default=None,
        help="Optional OpenRouter HTTP-Referer header (or env OPENROUTER_HTTP_REFERER).",
    )
    parser.add_argument(
        "--openrouter-x-title",
        type=str,
        default=None,
        help="Optional OpenRouter X-Title header (or env OPENROUTER_X_TITLE).",
    )
    parser.add_argument(
        "--judge-retries",
        type=int,
        default=3,
        help="Retries per sample when calling the judge (VLMEvalKit default is 3).",
    )
    parser.add_argument(
        "--judge-sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between judge API calls (rate limits).",
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    base_url = args.base_url
    default_headers = None
    if args.api_provider == "openrouter":
        base_url = base_url or os.environ.get("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
        referer = args.openrouter_http_referer or os.environ.get("OPENROUTER_HTTP_REFERER")
        title = args.openrouter_x_title or os.environ.get("OPENROUTER_X_TITLE")
        headers = {}
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-Title"] = title
        default_headers = headers or None
    elif args.api_provider == "openai":
        base_url = base_url or os.environ.get("OPENAI_BASE_URL")
    else:
        base_url = base_url or os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENROUTER_BASE_URL")
        headers = {}
        if args.openrouter_http_referer:
            headers["HTTP-Referer"] = args.openrouter_http_referer
        if args.openrouter_x_title:
            headers["X-Title"] = args.openrouter_x_title
        default_headers = headers or None

    judge_generate = None
    if args.scoring == "vlmeval":
        if not api_key:
            raise SystemExit(
                "API key required for --scoring vlmeval. Set --api-key or OPENAI_API_KEY/OPENROUTER_API_KEY."
            )
        judge_generate = make_openai_judge(
            args.judge_model,
            api_key,
            base_url=base_url,
            default_headers=default_headers,
        )

    rows = []
    with jsonlines.open(args.cap_file, "r") as reader:
        for line in reader:
            rows.append(line)

    if not rows:
        print("No rows in", args.cap_file)
        return

    preds = []
    sources = []
    for i, row in enumerate(rows):
        q = row.get("question", "")
        choices = parse_mcq_choices_from_question(q)
        gt = str(row.get("answer", "")).strip().upper()[:1]
        response = row.get("response", "")

        if args.scoring == "exact":
            pred = infer_prediction_exact_only(response, choices)
            src = "exact"
        else:
            pred, src = extract_answer_vlmeval_item(
                response,
                choices,
                q,
                judge_generate=judge_generate,
                retries=args.judge_retries,
                judge_sleep=args.judge_sleep,
            )

        preds.append(pred)
        sources.append(src)

        if (i + 1) % 200 == 0:
            print(f"  scored {i + 1}/{len(rows)}", flush=True)

    df = pd.DataFrame(rows)
    df["prediction_parsed"] = preds
    df["parse_source"] = sources
    hit_list = []
    for row, p in zip(rows, preds):
        gt = str(row.get("answer", "")).strip().upper()[:1]
        if p is None or p == "Z":
            hit_list.append(0)
        else:
            hit_list.append(int(p == gt))
    df["hit"] = hit_list

    acc_df = report_acc_mmstar(df)
    print("MMStar accuracy:")
    print(acc_df.to_string())
    print(
        f"\nOverall (mean hit): {df['hit'].mean():.4f}  (n={len(df)})  scoring={args.scoring}",
    )
    if args.scoring == "vlmeval":
        src_counts = df["parse_source"].value_counts().to_dict()
        print("Parse sources:", src_counts)

    if args.summary_csv:
        parent = os.path.dirname(args.summary_csv)
        if parent:
            os.makedirs(parent, exist_ok=True)
        acc_df.to_csv(args.summary_csv, index=False)
        per_sample = args.summary_csv.replace(".csv", "_per_sample.csv")
        df.to_csv(per_sample, index=False)

    if args.summary_file:
        parent = os.path.dirname(args.summary_file)
        if parent:
            os.makedirs(parent, exist_ok=True)
        summary = {
            "cap_file": args.cap_file,
            "scoring": args.scoring,
            "api_provider": args.api_provider,
            "judge_model": args.judge_model,
            "overall_mean_hit": float(df["hit"].mean()),
            "n": int(len(df)),
            "accuracy_table": acc_df.to_dict(orient="records"),
        }
        if args.scoring == "vlmeval":
            summary["parse_source_counts"] = {
                str(k): int(v) for k, v in df["parse_source"].value_counts().items()
            }
        with open(args.summary_file, "w", encoding="utf-8") as sf:
            json.dump(summary, sf, indent=2)


if __name__ == "__main__":
    main()
