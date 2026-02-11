"""
Evaluate Qwen2.5-VL-3B on the Zebra-CoT Geometry dataset.

- Robust answer extraction from model output (handles missing tags)
- LaTeX normalization (sin, cos, pi, etc.)
- Uses math_verify for LaTeX answer comparison
- Real-time JSONL logging of wrong predictions
- Logs to wandb
"""

import argparse
import json
import os
import re
import time
import traceback

import torch
import wandb
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig, StringExtractionConfig


# ─── Config ───────────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
RESULTS_DIR = "/workspace/vlm-grpo-qwen25vl3b/outputs/zebra_cot_baseline_results"

SYSTEM_PROMPT = (
    "A user asks you a question, and you should try to solve it. "
    "You should first think about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
    "i.e., <think> reasoning process here </think> <answer> answer here </answer>."
    "In <answer>, output only a single LaTeX expression wrapped in $...$."
    "Return ONLY in this exact format:<think>...</think><answer>...</answer>."
    "Do not output anything outside the tags."
)


# ─── LaTeX normalization ──────────────────────────────────────────────────────

# Math functions that need a backslash prefix in LaTeX
_MATH_FUNCTIONS = [
    "arcsin", "arccos", "arctan",  # longer names first to avoid partial match
    "sinh", "cosh", "tanh",
    "sin", "cos", "tan", "cot", "sec", "csc",
    "log", "ln", "exp", "sqrt",
    "lim", "inf", "sup", "min", "max",
    "det", "dim", "gcd", "lcm",
    "mod", "deg",
]

# Greek letters / math symbols that need backslash
_MATH_SYMBOLS = [
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "rho",
    "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega",
    "pi", "Pi", "infty", "cdot", "times", "circ", "degree",
]


def normalize_latex(expr: str) -> str:
    """
    Normalize a LaTeX expression:
    - Add backslash to common math functions (sin -> \\sin)
    - Add backslash to Greek letters / symbols (pi -> \\pi)
    - Normalize multiple choice answers: (A), A) , A. , etc. -> A
    - Clean up whitespace
    """
    if not expr:
        return expr

    # Unwrap $...$ for processing
    inner = expr
    was_wrapped = False
    if inner.startswith('$') and inner.endswith('$'):
        inner = inner[1:-1].strip()
        was_wrapped = True

    # ── Multiple choice normalization ──
    # Match patterns like: A, (A), A), A., a, (a), etc.
    mc_match = re.fullmatch(r'\(?([A-Da-d])\)?\.?', inner.strip())
    if mc_match:
        letter = mc_match.group(1).upper()
        return f'${letter}$' if was_wrapped else letter

    # ── Add backslash to math functions ──
    # Only add if not already preceded by backslash
    for func in _MATH_FUNCTIONS:
        # Negative lookbehind for \, word boundary aware
        # e.g. "sin" but not "\sin", not "asin" (handled by ordering)
        inner = re.sub(
            r'(?<!\\)(?<![a-zA-Z])' + re.escape(func) + r'(?![a-zA-Z])',
            '\\\\' + func,
            inner
        )

    # ── Add backslash to Greek letters / symbols ──
    for sym in _MATH_SYMBOLS:
        inner = re.sub(
            r'(?<!\\)(?<![a-zA-Z])' + re.escape(sym) + r'(?![a-zA-Z])',
            '\\\\' + sym,
            inner
        )

    # ── Clean up double backslashes from over-escaping ──
    inner = re.sub(r'\\\\\\\\', r'\\\\', inner)  # \\\\func -> \\func

    inner = inner.strip()
    if was_wrapped:
        return f'${inner}$'
    return inner


# ─── Answer extraction helpers ────────────────────────────────────────────────

def _find_latex(text: str) -> str | None:
    """Find last $...$ LaTeX expression in text. Returns with $ delimiters."""
    matches = re.findall(r'\$[^$]+\$', text)
    if matches:
        return matches[-1].strip()
    return None


def _find_latex_after_is(text: str) -> str | None:
    """Find LaTeX expression after 'is:' or 'is ' in text."""
    is_match = re.search(r'\bis[\s:]+', text)
    if is_match:
        after = text[is_match.end():]
        latex = _find_latex(after)
        if latex:
            return latex
    return None


def _find_plain_after_is(text: str) -> str | None:
    """
    Find a non-LaTeX expression after 'is:' or 'is ' in text.
    Grabs the remaining trimmed content after the last 'is'/'is:' match.
    Returns the raw string (without $ wrapping) or None.
    """
    # Find the last "is:" or "is " occurrence
    matches = list(re.finditer(r'\bis[\s:]+', text))
    if not matches:
        return None
    last = matches[-1]
    after = text[last.end():].strip()
    if not after:
        return None
    # Strip trailing punctuation like periods
    after = after.rstrip('.').strip()
    if after:
        return after
    return None


def extract_model_answer(response: str) -> tuple[str | None, str]:
    """
    Extract the answer from model response with robust fallback logic.

    Returns (extracted_answer_or_None, extraction_method).

    Logic:
      1. <answer>...</answer> present → prefer $...$ inside, else full string
      2. </answer> present but no <answer>:
         a. LaTeX after "is:"/"is "
         b. Non-LaTeX expression after "is:"/"is " (before </answer>)
         c. LaTeX between </think> and </answer>
         d. Full string between </think> and </answer>
      3. No </answer> → return (None, "no_answer_tag")
    """
    has_answer_open = '<answer>' in response
    has_answer_close = '</answer>' in response

    # ── Case 1: Full <answer>...</answer> tags ──
    if has_answer_open and has_answer_close:
        match = re.search(r'<answer>\s*(.*?)\s*</answer>', response, re.DOTALL)
        if match:
            content = match.group(1).strip()
            if not content:
                return None, "answer_tag_empty"
            # Prefer LaTeX inside
            latex = _find_latex(content)
            if latex:
                return latex, "answer_tag_latex"
            # Otherwise wrap the full content
            if not (content.startswith('$') and content.endswith('$')):
                content = f'${content}$'
            return content, "answer_tag_full"

    # ── Case 2: Only </answer> (no <answer>) ──
    if has_answer_close and not has_answer_open:
        before_close = response[:response.rfind('</answer>')]

        # 2a. LaTeX after "is:"/"is "
        latex_after_is = _find_latex_after_is(before_close)
        if latex_after_is:
            return latex_after_is, "no_open_tag_is_latex"

        # 2b. Non-LaTeX expression after "is:"/"is " and before </answer>
        plain_after_is = _find_plain_after_is(before_close)
        if plain_after_is:
            return f'${plain_after_is}$', "no_open_tag_is_plain"

        # 2c. LaTeX between </think> and </answer>
        if '</think>' in before_close:
            between = before_close[before_close.rfind('</think>') + len('</think>'):]
            between = between.strip()
            if between:
                latex = _find_latex(between)
                if latex:
                    return latex, "no_open_tag_think_latex"
                # 2d. Full string between </think> and </answer>
                if not (between.startswith('$') and between.endswith('$')):
                    between = f'${between}$'
                return between, "no_open_tag_think_full"

        return None, "no_open_tag_unparseable"

    # ── Case 3: No </answer> tag at all ──
    return None, "no_answer_tag"


def classify_format(response: str) -> dict:
    """Classify the output format of the model response. Returns a dict with details."""
    has_think_open = '<think>' in response
    has_think_close = '</think>' in response
    has_answer_open = '<answer>' in response
    has_answer_close = '</answer>' in response

    if has_think_open and has_think_close and has_answer_open and has_answer_close:
        fmt = "think+answer"
    elif has_answer_open and has_answer_close:
        fmt = "answer_only"
    elif has_answer_close and not has_answer_open:
        fmt = "missing_answer_open"
    elif has_think_open or has_think_close:
        fmt = "think_only"
    else:
        fmt = "no_tags"

    return {
        "format": fmt,
        "has_think_open": has_think_open,
        "has_think_close": has_think_close,
        "has_answer_open": has_answer_open,
        "has_answer_close": has_answer_close,
    }


# ─── Ground truth extraction (fallback if no pre-extracted column) ────────────

def clean_latex_equals(answer: str) -> str:
    match = re.match(r'^\$(.*)\$$', answer)
    if not match:
        return answer
    inner = match.group(1).strip()
    if '=' in inner:
        rhs = inner.rsplit('=', 1)[-1].strip()
        return f'${rhs}$'
    return answer


def extract_ground_truth(final_answer: str) -> str:
    """Fallback ground truth extraction (used only if extracted_answer column is missing)."""
    matches = re.findall(r'\$[^$]+\$', final_answer)
    if matches:
        answer = matches[-1].strip()
        return clean_latex_equals(answer)
    matches = re.findall(r'`([^`]+)`', final_answer)
    if matches:
        return f'${matches[-1].strip()}$'
    match = re.search(r'(?:is|=)\s*([+-]?\d+(?:[./]\d+)?(?:\s*[*/+-]\s*\w+)*)', final_answer)
    if match:
        return f'${match.group(1).strip()}$'
    matches = re.findall(r'[+-]?\d+(?:\.\d+)?(?:/\d+)?', final_answer)
    if matches:
        return f'${matches[-1].strip()}$'
    answer = final_answer.strip()
    if not (answer.startswith('$') and answer.endswith('$')):
        answer = f'${answer}$'
    return answer


# ─── Comparison ───────────────────────────────────────────────────────────────

def safe_parse(expr: str, configs: list) -> object | None:
    """Parse an expression, returning None on failure instead of raising."""
    try:
        return parse(expr, extraction_config=configs)
    except Exception:
        return None


def compare_answers(pred_str: str, gt_str: str) -> tuple[bool, str, str]:
    """
    Compare predicted and ground truth answers.
    Returns (is_correct, parsed_gt_repr, parsed_pred_repr).
    """
    gt_parsed = safe_parse(gt_str, [LatexExtractionConfig()])
    pred_parsed = safe_parse(pred_str, [
        LatexExtractionConfig(),
        ExprExtractionConfig(),
        StringExtractionConfig(),
    ])

    gt_repr = repr(gt_parsed) if gt_parsed is not None else "PARSE_FAIL"
    pred_repr = repr(pred_parsed) if pred_parsed is not None else "PARSE_FAIL"

    if gt_parsed is not None and pred_parsed is not None:
        try:
            result = verify(gt_parsed, pred_parsed)
            return result, gt_repr, pred_repr
        except Exception:
            pass

    # Fallback: stripped lowercase comparison
    is_eq = pred_str.strip().lower() == gt_str.strip().lower()
    method = "string_fallback"
    return is_eq, gt_repr + f" ({method})", pred_repr + f" ({method})"


# ─── Main evaluation ─────────────────────────────────────────────────────────

def build_message(question: str, image) -> list:
    content = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": question})
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def _append_jsonl(path: str, obj: dict):
    """Append a single JSON object as a line to a JSONL file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",
                        default="/workspace/vlm-grpo-qwen25vl3b/data/zebra_cot/full_data_for_eval.jsonl",
                        help="Path to input JSONL file.")
    parser.add_argument("--image_dir",
                        default="/workspace/vlm-grpo-qwen25vl3b/data/zebra_cot/extracted/images",
                        help="Directory containing images.")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--output_dir", default=RESULTS_DIR,
                        help="Directory for output files")
    parser.add_argument("--wandb_project", default="vlm-grpo-qwen25vl3b")
    parser.add_argument("--wandb_run_name", default="eval-zebra-cot-baseline")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Output file paths ─────────────────────────────────────────────────
    correct_file = os.path.join(args.output_dir, "correct.jsonl")
    wrong_file = os.path.join(args.output_dir, "wrong.jsonl")
    summary_file = os.path.join(args.output_dir, "summary.json")

    for f in [correct_file, wrong_file]:
        open(f, "w").close()

    print(f"Correct predictions → {correct_file}")
    print(f"Wrong predictions   → {wrong_file}")

    # ── Load dataset ──────────────────────────────────────────────────────
    print(f"\nLoading data from: {args.data}")
    data_path = args.data
    if data_path.endswith(".jsonl"):
        with open(data_path, "r", encoding="utf-8") as f:
            dataset = [json.loads(line) for line in f if line.strip()]
    else:
        with open(data_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
            if isinstance(dataset, dict):
                for v in dataset.values():
                    if isinstance(v, list):
                        dataset = v
                        break
    if args.max_samples is not None:
        dataset = dataset[:args.max_samples]
    print(f"Loaded {len(dataset)} examples")

    # ── Init wandb ────────────────────────────────────────────────────────
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or "eval-zebra-cot",
        config={
            "model": MODEL_ID,
            "data": args.data,
            "num_examples": len(dataset),
            "max_new_tokens": args.max_new_tokens,
            "system_prompt": SYSTEM_PROMPT,
        },
    )

    # ── Load model (device_map="auto" spreads across all available GPUs) ──
    print(f"\nLoading model: {MODEL_ID}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print(f"Model loaded. device_map=auto (uses all visible GPUs)\n")

    # ── Run inference ─────────────────────────────────────────────────────
    correct = 0
    total = 0
    format_counts = {}
    extraction_method_counts = {}
    start_time = time.time()

    wrong_table_columns = [
        "index", "question", "final_answer_raw", "gt_answer",
        "pred_raw", "pred_normalized", "extraction_method", "format",
        "gt_parsed", "pred_parsed", "response_length", "response",
    ]
    wrong_table = wandb.Table(columns=wrong_table_columns)

    for i, row in enumerate(dataset):
        question = row["Question"]
        final_answer_raw = row["Final Answer"]
        gt_answer = row.get("extracted_answer") or extract_ground_truth(final_answer_raw)

        # Get the problem image
        image_name = row.get("image", None)
        image = os.path.join(args.image_dir, image_name) if image_name else None

        # Build messages & run inference
        messages = build_message(question, image)
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )

        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        response = processor.decode(generated_ids, skip_special_tokens=True)
        response_length = len(generated_ids)

        # Classify format
        fmt_info = classify_format(response)
        fmt = fmt_info["format"]
        format_counts[fmt] = format_counts.get(fmt, 0) + 1

        # Extract predicted answer
        pred_answer_raw, extraction_method = extract_model_answer(response)
        extraction_method_counts[extraction_method] = extraction_method_counts.get(extraction_method, 0) + 1

        # Normalize
        if pred_answer_raw:
            pred_answer_normalized = clean_latex_equals(pred_answer_raw)
            pred_answer_normalized = normalize_latex(pred_answer_normalized)
        else:
            pred_answer_normalized = None

        # Compare
        is_correct = False
        gt_parsed_repr = ""
        pred_parsed_repr = ""
        if pred_answer_normalized:
            is_correct, gt_parsed_repr, pred_parsed_repr = compare_answers(
                pred_answer_normalized, gt_answer
            )

        if is_correct:
            correct += 1
        total += 1

        result_entry = {
            "index": row.get("index", i),
            "correct": is_correct,
            "question": question,
            "final_answer_raw": final_answer_raw,
            "extracted_answer_gt": gt_answer,
            "model_response": response,
            "response_length_tokens": response_length,
            "pred_answer_raw": pred_answer_raw,
            "pred_answer_normalized": pred_answer_normalized,
            "extraction_method": extraction_method,
            "format": fmt,
            "format_details": fmt_info,
            "math_verify_gt_parsed": gt_parsed_repr,
            "math_verify_pred_parsed": pred_parsed_repr,
        }

        # Write to JSONL
        if is_correct:
            _append_jsonl(correct_file, result_entry)
        else:
            _append_jsonl(wrong_file, result_entry)
            wrong_table.add_data(
                result_entry["index"], question, final_answer_raw, gt_answer,
                pred_answer_raw or "", pred_answer_normalized or "",
                extraction_method, fmt, gt_parsed_repr, pred_parsed_repr,
                response_length, response,
            )

        # Log scalars
        wandb.log({
            "running_accuracy": correct / total,
            "correct_count": correct,
            "wrong_count": total - correct,
            "total": total,
        })

        # Progress
        elapsed = time.time() - start_time
        avg_time = elapsed / total
        eta = avg_time * (len(dataset) - total)
        acc_str = f"Acc: {correct}/{total} ({100*correct/total:.1f}%)  ETA: {eta:.0f}s"

        if not is_correct:
            print(f"\n{'─'*70}")
            print(f"✗ [{i+1}/{len(dataset)}]  {acc_str}")
            print(f"  Question:       {question}")
            print(f"  Final Answer:   {final_answer_raw}")
            print(f"  GT (extracted): {gt_answer}")
            print(f"  Pred (raw):     {pred_answer_raw}")
            print(f"  Pred (normed):  {pred_answer_normalized}")
            print(f"  Extraction:     {extraction_method}")
            print(f"  Format:         {fmt}")
            print(f"  GT parsed:      {gt_parsed_repr}")
            print(f"  Pred parsed:    {pred_parsed_repr}")
            print(f"  Response len:   {response_length} tokens")
            print(f"  Response:       {response}")
            print(f"{'─'*70}")
        else:
            print(f"  ✓ [{i+1}/{len(dataset)}]  {acc_str}")

    # ── Summary ───────────────────────────────────────────────────────────
    accuracy = correct / total if total > 0 else 0
    elapsed = time.time() - start_time

    summary = {
        "model": MODEL_ID,
        "data": args.data,
        "total": total,
        "correct": correct,
        "wrong": total - correct,
        "accuracy": accuracy,
        "elapsed_seconds": round(elapsed, 1),
        "avg_seconds_per_example": round(elapsed / total, 2) if total > 0 else 0,
        "format_counts": format_counts,
        "extraction_method_counts": extraction_method_counts,
    }

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  Model:    {MODEL_ID}")
    print(f"  Data:     {args.data}")
    print(f"  Total:    {total}")
    print(f"  Correct:  {correct}")
    print(f"  Wrong:    {total - correct}")
    print(f"  Accuracy: {100 * accuracy:.2f}%")
    print(f"  Time:     {elapsed:.1f}s ({elapsed/total:.2f}s/example)")
    print(f"\n  Output Format Distribution:")
    for fmt_name, count in sorted(format_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total if total > 0 else 0
        print(f"    {fmt_name:<25}: {count:>4} ({pct:.1f}%)")
    print(f"\n  Answer Extraction Method Distribution:")
    for method, count in sorted(extraction_method_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total if total > 0 else 0
        print(f"    {method:<30}: {count:>4} ({pct:.1f}%)")
    print("=" * 70)
    print(f"\n  Correct → {correct_file}")
    print(f"  Wrong   → {wrong_file}")

    wandb.log({
        "final_accuracy": accuracy,
        "total_examples": total,
        "correct_count": correct,
        "wrong_count": total - correct,
        "elapsed_seconds": elapsed,
        **{f"format/{k}": v for k, v in format_counts.items()},
        **{f"extraction/{k}": v for k, v in extraction_method_counts.items()},
    })
    if total - correct > 0:
        wandb.log({"wrong_predictions": wrong_table})

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Summary → {summary_file}")

    wandb.finish()


if __name__ == "__main__":
    main()
