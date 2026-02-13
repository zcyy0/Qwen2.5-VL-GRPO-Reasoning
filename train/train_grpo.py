"""
GRPO Curriculum Training Script
================================
Trains a Qwen2.5-VL model with GRPO on geometry datasets using curriculum
learning: difficulty 1 → 2 → 3. Supports multi-source data (geoqa, synthesis,
zebra_cot), per-difficulty and per-source metric logging, 4-GPU training,
and dev set evaluation between phases.

All answer types are treated as math — extraction and normalisation use
utils.extract_answer (extract_model_answer, normalize_answer_comprehensive).

Launch:
    accelerate launch --num_processes 4 --num_machines 1 --mixed_precision bf16 train/train_grpo_curriculum.py

Quick dry-run:
    python train/train_grpo_curriculum.py --phase1_steps 2 --phase2_steps 2 --phase3_steps 2 --use_vllm 0
"""

import argparse
import inspect
import json
import math
import os
import re
import statistics
import sys
import threading
from pathlib import Path
from collections import Counter, defaultdict

# Ensure project root is importable (for utils.extract_answer)
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from math_verify import parse, verify, LatexExtractionConfig
from math_verify.parser import ExprExtractionConfig
from utils.extract_answer import extract_model_answer, normalize_answer_comprehensive

import torch
torch.set_float32_matmul_precision("high")

from datasets import Dataset
from datasets import Image as HFImage
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, TrainerCallback
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer

import wandb

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are VL-Thinking, a helpful assistant with excellent reasoning ability. "
    "A user asks you a question, and you should try to solve it. "
    "You should first think about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
    "i.e., <think> reasoning process here </think> <answer> answer here </answer>. "
    "In <answer></answer>, output only a single LaTeX expression wrapped in $...$. "
    "Return ONLY in this exact format: <think>...</think> <answer>...</answer>. "
    "Do not output anything outside the tags."
)

TAG_BLOCK_RE = re.compile(
    r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$",
    re.DOTALL | re.IGNORECASE,
)

ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


# ──────────────────────────────────────────────────────────────────────────────
# JSONL reader
# ──────────────────────────────────────────────────────────────────────────────

def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"JSON decode error at line {line_no}: {e}") from e


# ──────────────────────────────────────────────────────────────────────────────
# Prompt construction (all math)
# ──────────────────────────────────────────────────────────────────────────────

def make_prompt(question: str):
    """Return conversation list (for GRPOTrainer to template internally)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question.strip()},
            ],
        },
    ]


def make_prompt_text(processor, question: str):
    """Return templated string (for manual evaluation/inference)."""
    conversation = make_prompt(question)
    return processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)


# ──────────────────────────────────────────────────────────────────────────────
# Tag detection (for format reward)
# ──────────────────────────────────────────────────────────────────────────────

def tag_flags(text: str):
    t = (text or "").lower()
    return {
        "has_think_open": "<think>" in t,
        "has_think_close": "</think>" in t,
        "has_answer_open": "<answer>" in t,
        "has_answer_close": "</answer>" in t,
        "has_strict_block": bool(TAG_BLOCK_RE.search(text or "")),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Scoring (all math — normalize + math_verify)
# ──────────────────────────────────────────────────────────────────────────────

def math_verify_equiv(gt_raw: str, pred_raw: str) -> tuple[bool, str, str]:
    """Return (is_equiv, gt_norm, pred_norm).

    Both sides are normalised via normalize_answer_comprehensive before parsing.
    """
    gt_norm = normalize_answer_comprehensive(gt_raw)
    pred_norm = normalize_answer_comprehensive(pred_raw)

    gold_parsed = parse(gt_norm, extraction_config=[LatexExtractionConfig()])
    pred_parsed = parse(pred_norm, extraction_config=[ExprExtractionConfig(), LatexExtractionConfig()])
    return bool(verify(gold_parsed, pred_parsed)), gt_norm, pred_norm


def score_one(gold: str, pred_answer: str) -> tuple[float, str, str]:
    """Score a single prediction against gold (math equivalence).

    Returns (score, gt_norm, pred_norm) so callers can log mismatches.
    """
    is_equiv, gt_norm, pred_norm = math_verify_equiv(gold, pred_answer)
    return (1.0 if is_equiv else 0.0), gt_norm, pred_norm


def completion_to_text(comp):
    if isinstance(comp, str):
        return comp
    if isinstance(comp, list) and comp and isinstance(comp[0], dict) and "content" in comp[0]:
        return comp[0]["content"]
    if isinstance(comp, dict) and "content" in comp:
        return comp["content"]
    return str(comp)


# ──────────────────────────────────────────────────────────────────────────────
# Reward Tracker — per-difficulty & per-source logging
# ──────────────────────────────────────────────────────────────────────────────

class RewardTracker:
    """Thread-safe accumulator for per-group reward statistics.

    Tracks per-generation: reward, has_strict, has_answer_tags, is_correct.
    Groups generations by prompt_id to compute prompt-level metrics.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._records = []  # (reward, difficulty, source, has_strict, has_answer_tags, is_correct, prompt_id, comp_len, parsed_ok)

    def record(self, reward: float, difficulty: int, source: str,
               has_strict: bool, has_answer_tags: bool, is_correct: bool,
               prompt_id: str, comp_len: int = 0, parsed_ok: bool = True):
        with self._lock:
            self._records.append((reward, difficulty, source, has_strict, has_answer_tags, is_correct, prompt_id, comp_len, parsed_ok))

    def flush(self) -> dict:
        """Return per-group averages and clear the buffer."""
        with self._lock:
            recs = self._records[:]
            self._records.clear()

        if not recs:
            return {}

        n_total = len(recs)
        n_strict_correct = 0
        n_strict_incorrect = 0

        by_diff = defaultdict(list)
        by_src = defaultdict(list)
        prompt_has_strict_correct = set()
        prompt_has_strict_incorrect = set()
        all_prompts = set()

        total_comp_len = 0
        total_parse_ok = 0
        for rew, diff, src, has_strict, has_answer_tags, is_correct, prompt_id, comp_len, parsed_ok in recs:
            by_diff[diff].append(rew)
            by_src[src].append(rew)
            all_prompts.add(prompt_id)
            total_comp_len += comp_len
            if parsed_ok:
                total_parse_ok += 1

            if has_strict and is_correct:
                n_strict_correct += 1
                prompt_has_strict_correct.add(prompt_id)
            elif has_strict and not is_correct:
                n_strict_incorrect += 1
                prompt_has_strict_incorrect.add(prompt_id)

        n_prompts = max(1, len(all_prompts))

        metrics = {
            "train/gen_strict_correct_pct": n_strict_correct / n_total,
            "train/gen_strict_incorrect_pct": n_strict_incorrect / n_total,
            "train/prompt_strict_correct_pct": len(prompt_has_strict_correct) / n_prompts,
            "train/prompt_strict_incorrect_pct": len(prompt_has_strict_incorrect) / n_prompts,
            "train/parse_success_rate": total_parse_ok / n_total,
            "train/avg_completion_tokens": total_comp_len / n_total,
        }

        for src, vals in sorted(by_src.items()):
            metrics[f"train/reward_by_source/{src}/mean"] = sum(vals) / len(vals)
            metrics[f"train/reward_by_source/{src}/std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0

        return metrics


# Module-level tracker instance (accessed by reward functions & callback)
_reward_tracker = RewardTracker()
_tokenizer = None  # set in main() for token-count logging


# ──────────────────────────────────────────────────────────────────────────────
# Reward functions
# ──────────────────────────────────────────────────────────────────────────────

def format_reward(completions, **kwargs):
    """Returns 0.1 if strict format (<think>...</think> <answer>...</answer>), 0.0 otherwise."""
    rewards = []
    for comp in completions:
        text = completion_to_text(comp)
        flags = tag_flags(text)
        has_strict = flags["has_strict_block"]
        rewards.append(0.1 if has_strict else 0.0)
    return rewards


def accuracy_reward(completions, gt=None, source=None, difficulty=None, id=None, **kwargs):
    """Returns 0.9 if strict format AND correct answer, 0.0 otherwise.

    Also records combined reward (format + accuracy) to the reward tracker.
    """
    rewards = []
    n = len(completions)
    if gt is None:
        gt = [None] * n
    if source is None:
        source = [""] * n
    if difficulty is None:
        difficulty = [0] * n
    if id is None:
        id = [str(i) for i in range(n)]

    for comp, gold, src, diff, pid in zip(completions, gt, source, difficulty, id):
        text = completion_to_text(comp)
        if gold is None:
            rewards.append(0.0)
            continue

        flags = tag_flags(text)
        has_strict = flags["has_strict_block"]
        has_answer_tags = flags["has_answer_open"] and flags["has_answer_close"]

        pred, _method = extract_model_answer(text)
        parsed_ok = pred is not None
        if parsed_ok:
            score, _, _ = score_one(str(gold), pred)
            is_correct = score > 0
        else:
            is_correct = False

        acc_rew = 0.9 if (has_strict and is_correct) else 0.0
        rewards.append(acc_rew)

        # Record combined reward (format + accuracy) for logging
        total_rew = (0.1 if has_strict else 0.0) + acc_rew

        try:
            diff_int = int(diff)
        except (TypeError, ValueError):
            diff_int = 0
        _reward_tracker.record(total_rew, diff_int, src,
                               has_strict=has_strict,
                               has_answer_tags=has_answer_tags,
                               is_correct=is_correct,
                               prompt_id=str(pid),
                               comp_len=len(_tokenizer.encode(text, add_special_tokens=False)) if _tokenizer else len(text),
                               parsed_ok=parsed_ok)

    return rewards


# ──────────────────────────────────────────────────────────────────────────────
# Reward tracker callback
# ──────────────────────────────────────────────────────────────────────────────

class RewardTrackerCallback(TrainerCallback):
    """Flushes accumulated per-group reward stats every N steps.

    Prints training metrics every log_every steps:
      gen_strict_correct_pct — % of generations with strict format AND correct (reward=1.0)
      gen_strict_incorrect_pct — % with strict format but incorrect answer (reward=0.1)
      prompt_strict_correct_pct — % of prompts with >=1 strict+correct gen
      prompt_strict_incorrect_pct — % of prompts with >=1 strict+incorrect gen
      reward_by_source — mean reward per source
    """

    def __init__(self, log_every: int = 50):
        self.log_every = log_every

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_every != 0:
            return control
        metrics = _reward_tracker.flush()
        if metrics:
            gsc = metrics.get("train/gen_strict_correct_pct", 0)
            gsi = metrics.get("train/gen_strict_incorrect_pct", 0)
            psc = metrics.get("train/prompt_strict_correct_pct", 0)
            psi = metrics.get("train/prompt_strict_incorrect_pct", 0)
            psr = metrics.get("train/parse_success_rate", 0)
            print(f"\n[Step {state.global_step}] Training metrics:")
            print(f"  gen_strict_correct={gsc:.2%}  gen_strict_incorrect={gsi:.2%}")
            print(f"  prompt_strict_correct={psc:.2%}  prompt_strict_incorrect={psi:.2%}")
            print(f"  parse_success_rate={psr:.2%}")
            # Per-source reward (mean ± std)
            sources = sorted(set(
                k.split("/")[-2] for k in metrics if "reward_by_source/" in k and "/mean" in k
            ))
            for src in sources:
                avg_r = metrics.get(f"train/reward_by_source/{src}/mean", 0)
                std_r = metrics.get(f"train/reward_by_source/{src}/std", 0)
                print(f"  reward_by_source/{src}: {avg_r:.3f} ± {std_r:.3f}")
            avg_clen = metrics.get("train/avg_completion_tokens", 0)
            print(f"  avg_completion_tokens={avg_clen:.0f}")
            if wandb.run is not None:
                wandb.log(metrics, step=state.global_step)
        return control


# ──────────────────────────────────────────────────────────────────────────────
# Data loading — multi-source with field unification
# ──────────────────────────────────────────────────────────────────────────────

def _unify_example(ex: dict) -> dict | None:
    """Normalise field names across geoqa, synthesis, zebra_cot."""
    src = ex.get("_source", "")

    if src == "zebra_cot":
        question = ex.get("Question", "")
        image_rel = ex.get("image", "")
        eid = str(ex.get("index", ""))
    else:
        question = ex.get("question", "")
        image_rel = ex.get("image", "")
        eid = str(ex.get("id", ""))

    # Always use extracted_answer: for geoqa, "gt" is an MCQ letter (A/B/C/D),
    # not the math answer; for synthesis, "gt" has \boxed{} wrapping that
    # leaves stray braces after normalisation.
    gt = ex.get("extracted_answer", None)

    difficulty = ex.get("difficulty", 0)

    if not isinstance(question, str) or not question.strip():
        return None
    if gt is None:
        return None

    return {
        "id": eid,
        "source": src,
        "question": question,
        "gt": str(gt),
        "image_rel": image_rel,
        "difficulty": int(difficulty),
    }


def _resolve_image(image_rel: str, images_root: Path, images_root_alt: Path) -> str | None:
    """Try primary root first, then alt root."""
    p1 = (images_root / image_rel).resolve()
    if p1.exists():
        return str(p1)
    p2 = (images_root_alt / image_rel).resolve()
    if p2.exists():
        return str(p2)
    return None


def load_dataset_from_jsonl(
    jsonl_path: Path,
    images_root: Path,
    images_root_alt: Path,
    processor,
    difficulty_filter: int | None = None,
    max_prompt_length: int | None = None,
):
    """Load examples from a JSONL file, unify fields, resolve images.

    If difficulty_filter is set, only keep rows matching that difficulty.
    If max_prompt_length is set, skip examples whose tokenised prompt exceeds it.
    """
    rows = []
    skipped_no_image = 0
    skipped_too_long = 0
    for ex in iter_jsonl(jsonl_path):
        unified = _unify_example(ex)
        if unified is None:
            continue
        if difficulty_filter is not None and unified["difficulty"] != difficulty_filter:
            continue

        img_path = _resolve_image(unified["image_rel"], images_root, images_root_alt)
        if img_path is None:
            skipped_no_image += 1
            continue

        prompt = make_prompt(unified["question"])
        prompt_json = json.dumps(prompt)
        prompt_text = make_prompt_text(processor, unified["question"])

        if max_prompt_length is not None:
            token_len = len(processor.tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
            if token_len > max_prompt_length:
                skipped_too_long += 1
                continue

        rows.append({
            "id": unified["id"],
            "source": unified["source"],
            "prompt": prompt_json,
            "prompt_text": prompt_text,
            "image": img_path,
            "gt": unified["gt"],
            "difficulty": unified["difficulty"],
        })

    if skipped_no_image > 0:
        print(f"  [load] Skipped {skipped_no_image} examples (image not found)")
    if skipped_too_long > 0:
        print(f"  [load] Skipped {skipped_too_long} examples (prompt > {max_prompt_length} tokens)")

    ds = Dataset.from_list(rows)
    ds = ds.cast_column("image", HFImage())
    return ds


# ──────────────────────────────────────────────────────────────────────────────
# Prompt JSON parsing (lazy transform for GRPOTrainer)
# ──────────────────────────────────────────────────────────────────────────────

def parse_prompt_column(ds: Dataset) -> Dataset:
    def _parse(examples):
        prompts = examples["prompt"]
        if isinstance(prompts, list):
            examples["prompt"] = [
                json.loads(p) if isinstance(p, str) else p for p in prompts
            ]
        else:
            examples["prompt"] = json.loads(prompts) if isinstance(prompts, str) else prompts
        return examples

    return ds.with_transform(_parse)


# ──────────────────────────────────────────────────────────────────────────────
# Eval helpers
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_from_prompt(model, processor, prompt_text, image_pil,
                         max_new_tokens: int, num_samples: int = 1,
                         temperature: float = 0.8):
    """Generate text from a prompt.

    num_samples=1: greedy (do_sample=False), returns (text, token_count).
    num_samples>1: temperature sampling, returns list of (text, token_count).
    """
    device = next(model.parameters()).device
    inputs = processor(
        text=[prompt_text],
        images=[image_pil],
        padding=True,
        return_tensors="pt",
    ).to(device)

    if num_samples == 1:
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        input_len = inputs["input_ids"].shape[1]
        gen_tokens = out[0][input_len:]
        decoded = processor.decode(gen_tokens, skip_special_tokens=True)
        return decoded, int(gen_tokens.numel())

    results = []
    for _ in range(num_samples):
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=True, temperature=temperature)
        input_len = inputs["input_ids"].shape[1]
        gen_tokens = out[0][input_len:]
        decoded = processor.decode(gen_tokens, skip_special_tokens=True)
        results.append((decoded, int(gen_tokens.numel())))
    return results


def compute_eval_metrics(model, processor, ds_subset: Dataset, max_new_tokens: int,
                         num_samples: int = 8, temperature: float = 0.8):
    """Compute evaluation metrics, distributing work across GPUs when available.

    Automatically shards the dataset across ranks, evaluates in parallel,
    gathers results, and computes metrics on rank 0.

    Returns metrics dict on rank 0, empty dict on other ranks.
    """
    n_total = len(ds_subset)
    if n_total == 0:
        return {}

    k = num_samples

    # Detect distributed environment
    distributed = torch.distributed.is_initialized()
    if distributed:
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        rank = 0
        world_size = 1

    # Unwrap DDP model for inference
    eval_model = model.module if hasattr(model, "module") else model

    # Shard dataset across ranks
    if distributed:
        indices = list(range(rank, n_total, world_size))
        local_ds = ds_subset.select(indices)
    else:
        local_ds = ds_subset
    n_local = len(local_ds)

    # Pick 5 random local indices for debug printing (rank 0 only)
    import random as _random
    debug_indices = set(_random.sample(range(n_local), min(5, n_local)))

    # Collect per-generation records on each rank
    records = []
    for i in range(n_local):
        if rank == 0 and i % 10 == 0:
            print(f"[Eval] Rank 0: {i}/{n_local} "
                  f"(~{i * world_size}/{n_total} total)...")

        ex = local_ds[i]
        src = ex["source"]
        diff = ex.get("difficulty", 0)
        pid = ex.get("id", str(i))
        gold = ex["gt"]

        samples = generate_from_prompt(
            eval_model, processor, ex["prompt_text"], ex["image"],
            max_new_tokens, num_samples=k, temperature=temperature,
        )
        if k == 1:
            samples = [samples]

        for s_idx, (decoded, clen) in enumerate(samples):
            flags = tag_flags(decoded)
            has_strict = flags["has_strict_block"]
            has_answer_tags = flags["has_answer_open"] and flags["has_answer_close"]

            pred, _method = extract_model_answer(decoded)
            parsed_ok = pred is not None
            if parsed_ok:
                score, gt_norm, pred_norm = score_one(str(gold), pred)
                is_correct = score > 0
            else:
                is_correct = False
                gt_norm = normalize_answer_comprehensive(str(gold))
                pred_norm = None

            if has_strict and is_correct:
                rew = 1.0
            elif has_strict:
                rew = 0.1
            else:
                rew = 0.0

            # Print detailed debug info for 5 random examples (sample 0 only)
            if rank == 0 and i in debug_indices and s_idx == 0:
                print(f"\n[DEBUG] Example {i}: src={src} diff={diff} id={pid}")
                print(f"  Ground truth (raw):  {gold}")
                print(f"  Ground truth (norm): {gt_norm}")
                print(f"  Model output: {decoded}")
                print(f"  Extracted answer: {pred}")
                print(f"  Extracted (norm):  {pred_norm}")
                print(f"  Correct={is_correct}  Strict={has_strict}  Reward={rew}")

            records.append({
                "src": src, "diff": diff, "pid": pid,
                "has_strict": has_strict, "has_answer_tags": has_answer_tags,
                "is_correct": is_correct, "rew": rew, "clen": clen,
                "parsed_ok": parsed_ok,
            })

    # Gather records from all ranks
    if distributed:
        gathered = [None] * world_size
        torch.distributed.all_gather_object(gathered, records)
        all_records = [r for shard in gathered for r in shard]
        # Only rank 0 computes metrics
        if rank != 0:
            return {}
    else:
        all_records = records

    # ── Compute metrics from gathered records (rank 0 only) ──
    total_gens = len(all_records)
    if total_gens == 0:
        return {}

    total_strict_correct = 0
    total_strict_incorrect = 0
    total_parse_ok = 0
    total_comp_len = 0

    all_prompts = set()
    prompt_has_strict_correct = set()
    prompt_has_strict_incorrect = set()

    src_diff_n_gen = Counter()
    src_diff_n_correct = Counter()
    src_diff_prompts = defaultdict(set)
    src_diff_prompt_any_correct = defaultdict(set)

    by_src_rewards = defaultdict(list)
    by_src_diff_rewards = defaultdict(list)

    for rec in all_records:
        src, diff, pid = rec["src"], rec["diff"], rec["pid"]
        sd_key = (src, diff)

        total_comp_len += rec["clen"]
        if rec["parsed_ok"]:
            total_parse_ok += 1

        if rec["has_strict"] and rec["is_correct"]:
            total_strict_correct += 1
            prompt_has_strict_correct.add(pid)
        elif rec["has_strict"] and not rec["is_correct"]:
            total_strict_incorrect += 1
            prompt_has_strict_incorrect.add(pid)

        all_prompts.add(pid)
        src_diff_n_gen[sd_key] += 1
        src_diff_prompts[sd_key].add(pid)
        by_src_rewards[src].append(rec["rew"])
        by_src_diff_rewards[sd_key].append(rec["rew"])

        if rec["is_correct"]:
            src_diff_n_correct[sd_key] += 1
            src_diff_prompt_any_correct[sd_key].add(pid)

    n_prompts = max(1, len(all_prompts))
    total_gens_safe = max(1, total_gens)

    print(f"[Eval] Finished: {len(all_prompts)} prompts, "
          f"{total_gens} generations across {world_size} GPU(s).")

    metrics = {
        "eval/gen_strict_correct_pct": total_strict_correct / total_gens_safe,
        "eval/gen_strict_incorrect_pct": total_strict_incorrect / total_gens_safe,
        "eval/prompt_strict_correct_pct": len(prompt_has_strict_correct) / n_prompts,
        "eval/prompt_strict_incorrect_pct": len(prompt_has_strict_incorrect) / n_prompts,
        "eval/parse_success_rate": total_parse_ok / total_gens_safe,
        "eval/avg_completion_tokens": total_comp_len / total_gens_safe,
    }

    # Per source reward (mean ± std)
    for src, vals in sorted(by_src_rewards.items()):
        metrics[f"eval/reward_by_source/{src}/mean"] = sum(vals) / len(vals)
        metrics[f"eval/reward_by_source/{src}/std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0

    # Per source x difficulty: accuracy, pass@k, reward mean/std
    for sd_key in sorted(src_diff_n_gen.keys()):
        src, diff = sd_key
        n_gen = max(1, src_diff_n_gen[sd_key])
        n_prompt = max(1, len(src_diff_prompts[sd_key]))
        n_correct = src_diff_n_correct[sd_key]
        n_pass = len(src_diff_prompt_any_correct[sd_key])
        metrics[f"eval/accuracy/{src}/d{diff}"] = n_correct / n_gen
        metrics[f"eval/pass@{k}/{src}/d{diff}"] = n_pass / n_prompt
        sd_vals = by_src_diff_rewards[sd_key]
        metrics[f"eval/reward/{src}/d{diff}/mean"] = sum(sd_vals) / len(sd_vals) if sd_vals else 0.0
        metrics[f"eval/reward/{src}/d{diff}/std"] = statistics.pstdev(sd_vals) if len(sd_vals) > 1 else 0.0

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation callback (runs within a phase)
# ──────────────────────────────────────────────────────────────────────────────

class EvalCallback(TrainerCallback):
    """Two-tier evaluation callback with distributed support.

    Quick eval: every ``quick_every`` steps on a fixed subset (greedy, 1 sample).
    Full eval:  once at the midpoint of the phase (end-of-phase is handled
                separately in main()).

    Work is sharded across GPUs automatically by compute_eval_metrics.
    Printing and wandb logging happen on rank 0 only.
    """

    def __init__(self, processor, dev_dataset, max_new_tokens: int,
                 quick_every: int, phase_steps: int,
                 quick_n: int = 100, num_samples: int = 8,
                 temperature: float = 0.8, global_step_offset: int = 0,
                 seed: int = 42):
        self.processor = processor
        self.dev_dataset = dev_dataset          # full dev set
        self.max_new_tokens = max_new_tokens
        self.quick_every = quick_every
        self.mid_step = phase_steps // 2        # full eval at midpoint
        self.num_samples = num_samples
        self.temperature = temperature
        self.global_step_offset = global_step_offset
        self._rank = (torch.distributed.get_rank()
                      if torch.distributed.is_initialized() else 0)

        # Subset for quick eval (same on all ranks)
        n = len(dev_dataset)
        if quick_n >= n:
            self.quick_subset = dev_dataset
        else:
            import random
            rng = random.Random(seed)
            indices = rng.sample(range(n), quick_n)
            self.quick_subset = dev_dataset.select(indices)

    def _print_metrics(self, dev_metrics, prefix="eval"):
        """Print metric summary (rank 0 only — caller must guard)."""
        gsc = dev_metrics.get(f"{prefix}/gen_strict_correct_pct", 0)
        gsi = dev_metrics.get(f"{prefix}/gen_strict_incorrect_pct", 0)
        psc = dev_metrics.get(f"{prefix}/prompt_strict_correct_pct", 0)
        psi = dev_metrics.get(f"{prefix}/prompt_strict_incorrect_pct", 0)
        psr = dev_metrics.get(f"{prefix}/parse_success_rate", 0)
        acl = dev_metrics.get(f"{prefix}/avg_completion_tokens", 0)
        print(f"  gen_strict_correct={gsc:.2%}  gen_strict_incorrect={gsi:.2%}")
        print(f"  prompt_strict_correct={psc:.2%}  prompt_strict_incorrect={psi:.2%}")
        print(f"  parse_success_rate={psr:.2%}  avg_completion_tokens={acl:.0f}")
        # Per-source reward (mean ± std)
        src_keys = sorted(set(
            k.split("/")[-2] for k in dev_metrics if "reward_by_source/" in k and "/mean" in k
        ))
        for src in src_keys:
            m = dev_metrics.get(f"{prefix}/reward_by_source/{src}/mean", 0)
            s = dev_metrics.get(f"{prefix}/reward_by_source/{src}/std", 0)
            print(f"  reward_by_source/{src}: {m:.3f} ± {s:.3f}")
        # Per source x difficulty: accuracy, pass@k, reward mean±std
        for key in sorted(dev_metrics):
            if "accuracy/" in key or "pass@" in key:
                print(f"  {key}: {dev_metrics[key]:.4f}")
            elif key.startswith(f"{prefix}/reward/") and "/mean" in key:
                base = key.replace("/mean", "")
                m = dev_metrics[key]
                s = dev_metrics.get(base + "/std", 0)
                label = base.replace(f"{prefix}/", "")
                print(f"  {label}: {m:.3f} ± {s:.3f}")

    def _run_quick_eval(self, model, state):
        was_training = model.training
        model.eval()
        real_step = state.global_step + self.global_step_offset
        if self._rank == 0:
            print(f"\n[Step {real_step}] Quick eval "
                  f"({len(self.quick_subset)} examples, greedy)...")
        metrics = compute_eval_metrics(
            model, self.processor, self.quick_subset, self.max_new_tokens,
            num_samples=1, temperature=self.temperature,
        )
        if was_training:
            model.train()
        # metrics is non-empty only on rank 0
        if self._rank == 0 and metrics:
            metrics = {k.replace("eval/", "quick_eval/"): v
                       for k, v in metrics.items()}
            print(f"\n[Step {real_step}] Quick Eval Summary:")
            self._print_metrics(metrics, prefix="quick_eval")
            if wandb.run is not None:
                wandb.log(metrics, step=real_step)
        return metrics

    def _run_full_eval(self, model, state):
        was_training = model.training
        model.eval()
        real_step = state.global_step + self.global_step_offset
        if self._rank == 0:
            print(f"\n[Step {real_step}] Full eval "
                  f"({len(self.dev_dataset)} examples, pass@{self.num_samples})...")
        metrics = compute_eval_metrics(
            model, self.processor, self.dev_dataset, self.max_new_tokens,
            num_samples=self.num_samples, temperature=self.temperature,
        )
        if was_training:
            model.train()
        if self._rank == 0 and metrics:
            print(f"\n[Step {real_step}] Full Eval Summary:")
            self._print_metrics(metrics, prefix="eval")
            if wandb.run is not None:
                wandb.log(metrics, step=real_step)
        return metrics

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 0:
            return control
        model = kwargs["model"]
        # Full eval at midpoint (skip quick eval on the same step)
        if self.mid_step > 0 and state.global_step == self.mid_step:
            self._run_full_eval(model, state)
        elif self.quick_every > 0 and state.global_step % self.quick_every == 0:
            self._run_quick_eval(model, state)
        return control


# ──────────────────────────────────────────────────────────────────────────────
# Helper: world-size detection
# ──────────────────────────────────────────────────────────────────────────────

def _get_world_size():
    for k in ("WORLD_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE"):
        v = os.environ.get(k)
        if v is not None:
            try:
                return int(v)
            except ValueError:
                continue
    return 1


# ──────────────────────────────────────────────────────────────────────────────
# Build a GRPOTrainer for one curriculum phase
# ──────────────────────────────────────────────────────────────────────────────

def build_trainer(
    model,
    processor,
    train_ds: Dataset,
    eval_ds: Dataset,
    args,
    phase: int,
    max_steps: int,
    global_step_offset: int,
    dev_subset: Dataset,
    beta: float = 0.0,
):
    """Create a GRPOTrainer for a single curriculum phase."""
    phase_output_dir = Path(args.output_dir) / f"phase{phase}"
    phase_output_dir.mkdir(parents=True, exist_ok=True)

    cfg_kwargs = dict(
        output_dir=str(phase_output_dir),
        learning_rate=args.learning_rate,
        max_steps=max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        report_to=["wandb"],
        bf16=True,
        remove_unused_columns=False,
        scale_rewards="batch",
        beta=beta,
        temperature=args.temperature,
        mask_truncated_completions=True,
        max_grad_norm=1.0,
        loss_type="dapo",
    )

    if int(args.use_vllm) == 1:
        cfg_kwargs.update(dict(
            use_vllm=True,
            vllm_mode="colocate",
            vllm_gpu_memory_utilization=0.3,
            vllm_model_impl="transformers",
        ))
    else:
        cfg_kwargs["use_vllm"] = False

    # Filter for TRL version compatibility
    allowed = set(inspect.signature(GRPOConfig.__init__).parameters.keys())
    allowed.discard("self")
    cfg_kwargs = {k: v for k, v in cfg_kwargs.items() if k in allowed}

    training_args = GRPOConfig(**cfg_kwargs)

    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=[format_reward, accuracy_reward],
        args=training_args,
        train_dataset=parse_prompt_column(train_ds),
        eval_dataset=parse_prompt_column(eval_ds),
    )

    # Add per-group reward tracker callback (same cadence as HF logging)
    trainer.add_callback(RewardTrackerCallback(log_every=args.logging_steps))

    # Add two-tier eval callback (quick every N steps, full at midpoint)
    if dev_subset is not None and args.quick_eval_steps > 0:
        trainer.add_callback(EvalCallback(
            processor=processor,
            dev_dataset=dev_subset,
            max_new_tokens=args.max_completion_length,
            quick_every=args.quick_eval_steps,
            phase_steps=max_steps,
            quick_n=args.quick_eval_n,
            num_samples=args.eval_num_samples,
            temperature=args.temperature,
            global_step_offset=global_step_offset,
        ))

    return trainer


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="GRPO Curriculum Training (difficulty 1→2→3)")

    # Data
    ap.add_argument("--train_file", type=str, default="data/splits/train.jsonl")
    ap.add_argument("--val_file", type=str, default="data/splits/val.jsonl")
    ap.add_argument("--images_root", type=str, default="data/vlaa_thinking_raw/images",
                    help="Primary image root (geoqa, synthesis)")
    ap.add_argument("--images_root_alt", type=str, default="data",
                    help="Fallback image root (zebra_cot)")
    ap.add_argument("--output_dir", type=str, default="outputs/grpo_curriculum")

    # Model
    ap.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")

    # Phase epochs
    ap.add_argument("--phase1_epochs", type=int, default=2)
    ap.add_argument("--phase2_epochs", type=int, default=2)
    ap.add_argument("--phase3_epochs", type=int, default=3)

    # Training
    ap.add_argument("--learning_rate", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.0,
                    help="KL penalty coefficient for GRPO")
    ap.add_argument("--per_device_train_batch_size", type=int, default=2)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--num_generations", type=int, default=8)
    ap.add_argument("--max_prompt_length", type=int, default=1024)
    ap.add_argument("--max_completion_length", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--use_vllm", type=int, default=1)

    # LoRA
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    # Logging & eval
    ap.add_argument("--logging_steps", type=int, default=10,
                    help="Log training metrics (loss, reward stats) every N steps")
    ap.add_argument("--quick_eval_steps", type=int, default=100,
                    help="Quick eval (greedy, subset) every N steps (0 to disable)")
    ap.add_argument("--quick_eval_n", type=int, default=300,
                    help="Number of dev examples for quick eval")
    ap.add_argument("--save_steps", type=int, default=100,
                    help="Save checkpoint every N steps")
    ap.add_argument("--eval_num_samples", type=int, default=8,
                    help="Number of samples per prompt for pass@k (full eval)")

    args = ap.parse_args()

    # Resolve paths relative to working directory
    base = Path.cwd()
    train_file = (base / args.train_file).resolve()
    val_file = (base / args.val_file).resolve()
    images_root = (base / args.images_root).resolve()
    images_root_alt = (base / args.images_root_alt).resolve()
    output_dir = (base / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    is_main = int(os.environ.get("LOCAL_RANK", "0")) == 0

    # Batch-geometry check
    world_size = _get_world_size()
    effective_batch = world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps
    if effective_batch % args.num_generations != 0:
        raise ValueError(
            f"Global batch ({effective_batch}) not divisible by num_generations ({args.num_generations}). "
            f"Adjust GPUs / batch / grad_accum / num_generations."
        )
    if is_main:
        print(f"[Config] world_size={world_size}  effective_batch={effective_batch}  "
              f"num_generations={args.num_generations}")

    # W&B
    if os.environ.get("WANDB_PROJECT") and is_main:
        wandb.init(
            project=os.environ["WANDB_PROJECT"],
            group=os.environ.get("WANDB_RUN_GROUP"),
            name=os.environ.get("WANDB_RUN_NAME"),
        )

    # ── Processor ──
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(args.model_id, use_fast=True, padding_side="left")

    global _tokenizer
    _tokenizer = processor.tokenizer

    # ── Load datasets per difficulty ──
    print("Loading datasets...")
    phase_datasets = {}
    for diff in (1, 2, 3):
        ds = load_dataset_from_jsonl(train_file, images_root, images_root_alt, processor,
                                     difficulty_filter=diff,
                                     max_prompt_length=args.max_prompt_length)
        phase_datasets[diff] = ds
        print(f"  Phase {diff} (difficulty={diff}): {len(ds)} examples")

    # Load val set (no difficulty filter — use everything)
    val_ds = load_dataset_from_jsonl(val_file, images_root, images_root_alt, processor,
                                     max_prompt_length=args.max_prompt_length)
    print(f"  Val set: {len(val_ds)} examples")

    print(f"  Eval: using full val set ({len(val_ds)} examples)")

    # ── Model + LoRA ──
    print("Loading model (bf16)...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16,
    )

    print("Applying LoRA...")
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        init_lora_weights=True,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Curriculum loop ──
    prompts_per_step = world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps // args.num_generations
    phase_epochs = {1: args.phase1_epochs, 2: args.phase2_epochs, 3: args.phase3_epochs}
    phase_betas = {1: 0.0, 2: 0.0, 3: 0.0}
    global_step_offset = 0

    for phase in (1, 2, 3):
        train_ds = phase_datasets[phase]
        beta = phase_betas[phase]
        epochs = phase_epochs[phase]
        steps = math.ceil(len(train_ds) / prompts_per_step) * epochs

        print("\n" + "=" * 70)
        print(f"PHASE {phase}  (difficulty={phase}, {len(train_ds)} examples, "
              f"{epochs} epochs, {steps} steps, {prompts_per_step} prompts/step, beta={beta})")
        print("=" * 70)

        if wandb.run is not None:
            wandb.log({"curriculum/phase": phase}, step=global_step_offset)

        trainer = build_trainer(
            model=model,
            processor=processor,
            train_ds=train_ds,
            eval_ds=val_ds,
            args=args,
            phase=phase,
            max_steps=steps,
            global_step_offset=global_step_offset,
            dev_subset=val_ds,
            beta=beta,
        )

        trainer.train()

        # Run evaluation between phases (all ranks participate, rank 0 reports)
        # Use trainer.model which is the DDP-wrapped version on each rank
        if is_main:
            print(f"\n[Phase {phase}] Post-phase pass@{args.eval_num_samples} evaluation on val set...")
        trainer.model.eval()
        phase_metrics = compute_eval_metrics(
            trainer.model, processor, val_ds, args.max_completion_length,
            num_samples=args.eval_num_samples, temperature=args.temperature,
        )
        trainer.model.train()

        # phase_metrics is non-empty only on rank 0
        if is_main and phase_metrics:
            if wandb.run is not None:
                tagged = {f"phase{phase}/{k}": v for k, v in phase_metrics.items()}
                tagged["curriculum/phase"] = phase
                wandb.log(tagged, step=global_step_offset + steps)

            print(f"\n[Phase {phase}] Results:")
            gsc = phase_metrics.get("eval/gen_strict_correct_pct", 0)
            gsi = phase_metrics.get("eval/gen_strict_incorrect_pct", 0)
            psc = phase_metrics.get("eval/prompt_strict_correct_pct", 0)
            psi = phase_metrics.get("eval/prompt_strict_incorrect_pct", 0)
            psr = phase_metrics.get("eval/parse_success_rate", 0)
            acl = phase_metrics.get("eval/avg_completion_tokens", 0)
            print(f"  gen_strict_correct={gsc:.2%}  gen_strict_incorrect={gsi:.2%}")
            print(f"  prompt_strict_correct={psc:.2%}  prompt_strict_incorrect={psi:.2%}")
            print(f"  parse_success_rate={psr:.2%}  avg_completion_tokens={acl:.0f}")
            for key in sorted(phase_metrics):
                if "reward_by_source/" in key and "/mean" in key:
                    src = key.split("/")[-2]
                    m = phase_metrics[key]
                    s = phase_metrics.get(key.replace("/mean", "/std"), 0)
                    print(f"  reward_by_source/{src}: {m:.3f} ± {s:.3f}")
            for key in sorted(phase_metrics):
                if "accuracy/" in key or "pass@" in key:
                    print(f"  {key}: {phase_metrics[key]:.4f}")
                elif key.startswith("eval/reward/") and "/mean" in key:
                    base = key.replace("/mean", "")
                    m = phase_metrics[key]
                    s = phase_metrics.get(base + "/std", 0)
                    label = base.replace("eval/", "")
                    print(f"  {label}: {m:.3f} ± {s:.3f}")

        global_step_offset += steps

        # Clean up trainer (but keep the model — weights carry over)
        del trainer
        torch.cuda.empty_cache()

    # ── Final save ──
    final_dir = output_dir / "final"
    model.save_pretrained(str(final_dir))
    processor.save_pretrained(str(final_dir))
    print(f"\nTraining complete. Final model saved to {final_dir}")

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
