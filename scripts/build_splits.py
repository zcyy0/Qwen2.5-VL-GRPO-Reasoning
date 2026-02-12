#!/usr/bin/env python3
"""
Rebuild train / dev / test splits from scratch.

Difficulty labelling (applied per-dataset):
    difficulty=3  top 30%  (hardest)
    difficulty=2  30–70%   (medium)
    difficulty=1  bottom 30% (easiest)

Target: --n-train examples (default 5000) with 20% d=3 / 60% d=2 / 20% d=1.

GeoQA selection:
    - geo3k:      ALL examples kept
    - geoqa_plus: random sample (default 3500)

Synthesis selection:
    - gen_func-func_polynomial, gen_solid: ALL kept
    - 5 trig categories: random sample each (default 700)
    - gen_plane: dropped

Zebra-CoT: ALL 262 → train only

Splitting (from geoqa + synthesis pool):
    - 300 dev + 1000 test, stratified by (source, difficulty)
    - Remainder capped to n_train (stratified), + zebra_cot → train

Usage:
    python scripts/rebuild_splits.py
    python scripts/rebuild_splits.py --seed 42 --dry-run
"""

import argparse
import ast
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

BASE = Path("/workspace/vlm-grpo-qwen25vl3b")
GEOQA_NORM = BASE / "data/vlaa_thinking_raw/geoqa/geoqa_open_normalized.jsonl"
SYNTH_NORM = BASE / "data/vlaa_thinking_raw/synthesis/synthesis_normalized.jsonl"
ZEBRA_NORM = BASE / "data/zebra_cot/normalized_zebra_cot_geometry.jsonl"
IMAGES_ROOT = BASE / "data/vlaa_thinking_raw/images"
IMAGES_ROOT_ALT = BASE / "data"
SPLITS_DIR = BASE / "data/splits"

SYNTH_KEEP_ALL = {"gen_func-func_polynomial", "gen_solid"}
SYNTH_DOWNSAMPLE = {
    "gen_func-func_absolute",
    "gen_func-func_cosine",
    "gen_func-func_logarithmic",
    "gen_func-func_sine",
    "gen_func-func_tangent",
}
SYNTH_DROP = {"gen_plane"}


# ──────────────────────────────────────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Difficulty scoring
# ──────────────────────────────────────────────────────────────────────────────

def _parse_meta(meta) -> dict:
    if isinstance(meta, dict):
        return meta
    if isinstance(meta, str):
        try:
            return ast.literal_eval(meta)
        except Exception:
            return {}
    return {}


def _think_length(ds_answer: str) -> int:
    if "</think>" in ds_answer:
        return len(ds_answer.split("</think>")[0])
    return 0


def _geoqa_gt_complexity(gt: str, extracted: str) -> int:
    score = 0
    combined = gt + " " + extracted
    for pat in ("\\frac", "\\pi", "\\sqrt", "\\circ"):
        if pat in combined:
            score += 1
    if re.search(r"[+-]", extracted.strip("$")):
        score += 1
    if len(extracted.strip("$")) > 10:
        score += 1
    return score


def geoqa_raw_scores(row: dict):
    """Return (think_len, caption_len, gt_complexity)."""
    meta = _parse_meta(row.get("meta", {}))
    ds_answer = meta.get("ds_answer", "")
    caption = row.get("caption", "")
    gt = row.get("gt", "")
    extracted = row.get("extracted_answer", "")
    return (_think_length(ds_answer), len(caption), _geoqa_gt_complexity(gt, extracted))


def _synth_gt_complexity(gt: str, extracted: str) -> int:
    score = 0
    if "\\frac" in gt:
        score += 1
    if "\\pi" in gt:
        score += 1
    if "\\left" in gt or "\\right" in gt:
        score += 1
    if extracted.startswith("$-") or "= -" in gt:
        score += 1
    if "\\sqrt" in gt:
        score += 1
    inner = extracted.strip("$")
    if re.search(r"(?<=\w)\s*[+-]\s*(?=\w|\\)", inner):
        score += 1
    if re.search(r"\\(?:sin|cos|tan|log)\s*\\?\(.*\\(?:frac|pi|sqrt)", gt):
        score += 1
    return score


def synth_raw_scores(row: dict):
    """Return (think_len, gt_complexity)."""
    meta = _parse_meta(row.get("meta", {}))
    ds_answer = meta.get("ds_answer", "")
    gt = row.get("gt", "")
    extracted = row.get("extracted_answer", "")
    return (_think_length(ds_answer), _synth_gt_complexity(gt, extracted))


def _zebra_answer_complexity(extracted: str) -> int:
    score = 0
    for pat in ("\\frac", "\\sqrt", "\\pi", "\\circ"):
        if pat in extracted:
            score += 1
    inner = extracted.strip("$")
    if re.search(r"(?<=\w)\s*[+-]\s*(?=\w|\\)", inner):
        score += 1
    if len(inner) > 15:
        score += 1
    return score


def _zebra_question_complexity(question: str) -> int:
    score = 0
    for pat in ("\\frac", "\\sqrt"):
        if pat in question:
            score += 1
    if len(question) > 300:
        score += 1
    if len(question) > 500:
        score += 1
    if re.search(r"(?:if|given that|such that|suppose)", question, re.IGNORECASE):
        score += 1
    return score


def zebra_raw_scores(row: dict):
    """Return (trace_len, n_thoughts, q_complexity, a_complexity)."""
    trace = row.get("Text Reasoning Trace", "")
    question = row.get("Question", "")
    extracted = row.get("extracted_answer", "")
    return (
        len(trace),
        trace.count("THOUGHT"),
        _zebra_question_complexity(question),
        _zebra_answer_complexity(extracted),
    )


def assign_difficulty(rows: list[dict], score_fn, weights: list[float]):
    """Score rows, normalize, combine with weights, assign difficulty.

    difficulty=3: top 30%  (hardest)
    difficulty=2: 30-70%   (medium)
    difficulty=1: bottom 30% (easiest)
    """
    if not rows:
        return rows

    # Compute raw scores
    raw = [score_fn(r) for r in rows]
    n_dims = len(raw[0])

    # Normalize each dimension
    normed = []
    for dim in range(n_dims):
        vals = [r[dim] for r in raw]
        mn, mx = min(vals), max(vals)
        rng = mx - mn if mx > mn else 1
        normed.append([(v - mn) / rng for v in vals])

    # Combined score
    for i, r in enumerate(rows):
        r["_score"] = sum(weights[d] * normed[d][i] for d in range(n_dims))

    # Sort hardest first and assign difficulty by percentile
    rows.sort(key=lambda r: r["_score"], reverse=True)
    n = len(rows)
    for i, r in enumerate(rows):
        pct = i / n
        if pct < 0.30:
            r["difficulty"] = 3
        elif pct < 0.70:
            r["difficulty"] = 2
        else:
            r["difficulty"] = 1

    # Clean
    for r in rows:
        r.pop("_score", None)

    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Synthesis category helper
# ──────────────────────────────────────────────────────────────────────────────

def synth_category(image: str) -> str | None:
    for cat in SYNTH_DROP:
        if cat in image:
            return "DROP"
    for cat in SYNTH_KEEP_ALL | SYNTH_DOWNSAMPLE:
        if cat in image:
            return cat
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Stratified split
# ──────────────────────────────────────────────────────────────────────────────

def stratified_split(rows: list[dict], n_dev: int, n_test: int, rng: random.Random):
    """Split rows into (train, dev, test) maintaining (source, difficulty) proportions.

    Returns (train, dev, test).
    """
    # Group by (source, difficulty)
    groups = defaultdict(list)
    for r in rows:
        key = (r["_source"], r["difficulty"])
        groups[key].append(r)

    total = len(rows)
    train_all, dev_all, test_all = [], [], []

    # Track allocation to hit exact targets
    dev_allocated = 0
    test_allocated = 0
    group_keys = sorted(groups.keys())

    # First pass: compute ideal fractional allocations
    dev_fracs = {}
    test_fracs = {}
    for key in group_keys:
        grp = groups[key]
        frac = len(grp) / total
        dev_fracs[key] = frac * n_dev
        test_fracs[key] = frac * n_test

    # Second pass: allocate with rounding, track remainders
    dev_alloc = {}
    test_alloc = {}
    for key in group_keys:
        dev_alloc[key] = int(dev_fracs[key])
        test_alloc[key] = int(test_fracs[key])

    # Distribute rounding remainders (largest remainder method)
    dev_remaining = n_dev - sum(dev_alloc.values())
    test_remaining = n_test - sum(test_alloc.values())

    dev_remainders = [(key, dev_fracs[key] - dev_alloc[key]) for key in group_keys]
    dev_remainders.sort(key=lambda x: x[1], reverse=True)
    for i in range(dev_remaining):
        dev_alloc[dev_remainders[i][0]] += 1

    test_remainders = [(key, test_fracs[key] - test_alloc[key]) for key in group_keys]
    test_remainders.sort(key=lambda x: x[1], reverse=True)
    for i in range(test_remaining):
        test_alloc[test_remainders[i][0]] += 1

    # Split each group
    for key in group_keys:
        grp = groups[key][:]
        rng.shuffle(grp)

        nd = dev_alloc[key]
        nt = test_alloc[key]

        # Ensure we don't exceed group size (leave at least some for train)
        total_take = nd + nt
        if total_take >= len(grp):
            # Need at least 1 for train if possible
            if len(grp) > 2:
                scale = (len(grp) - 1) / total_take
                nd = max(0, int(nd * scale))
                nt = max(0, int(nt * scale))
            else:
                nd, nt = 0, 0

        dev_all.extend(grp[:nd])
        test_all.extend(grp[nd:nd + nt])
        train_all.extend(grp[nd + nt:])

    rng.shuffle(train_all)
    rng.shuffle(dev_all)
    rng.shuffle(test_all)

    return train_all, dev_all, test_all


def targeted_subsample(
    rows: list[dict],
    diff_targets: dict[int, int],
    rng: random.Random,
) -> list[dict]:
    """Subsample rows to hit exact per-difficulty counts, preserving source ratios.

    diff_targets: {difficulty: count}, e.g. {1: 1000, 2: 3000, 3: 1000}.
    Within each difficulty, source proportions are maintained via largest-remainder.
    """
    # Group by (difficulty, source)
    by_diff_src = defaultdict(list)
    for r in rows:
        by_diff_src[(r["difficulty"], r["_source"])].append(r)

    result = []
    for diff, target_n in sorted(diff_targets.items()):
        # All sources at this difficulty
        sources = {}
        total_at_diff = 0
        for (d, src), grp in by_diff_src.items():
            if d == diff:
                sources[src] = grp
                total_at_diff += len(grp)

        if total_at_diff == 0:
            print(f"    WARNING: no examples at difficulty={diff}")
            continue

        if total_at_diff < target_n:
            print(f"    WARNING: only {total_at_diff} at difficulty={diff}, taking all (wanted {target_n})")
            for grp in sources.values():
                result.extend(grp)
            continue

        # Largest-remainder allocation across sources
        src_keys = sorted(sources.keys())
        alloc = {}
        for src in src_keys:
            alloc[src] = int(target_n * len(sources[src]) / total_at_diff)

        remaining = target_n - sum(alloc.values())
        remainders = [(src, (target_n * len(sources[src]) / total_at_diff) - alloc[src])
                      for src in src_keys]
        remainders.sort(key=lambda x: x[1], reverse=True)
        for i in range(remaining):
            alloc[remainders[i][0]] += 1

        for src in src_keys:
            grp = sources[src][:]
            rng.shuffle(grp)
            result.extend(grp[:alloc[src]])

    rng.shuffle(result)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Image existence check
# ──────────────────────────────────────────────────────────────────────────────

def has_image(row: dict) -> bool:
    img = row.get("image", "")
    if not img:
        return False
    if (IMAGES_ROOT / img).exists():
        return True
    if (IMAGES_ROOT_ALT / img).exists():
        return True
    return False


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Rebuild train/dev/test splits")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--geoqa-plus-n", type=int, default=4500)
    ap.add_argument("--synth-per-cat", type=int, default=900)
    ap.add_argument("--n-train", type=int, default=5000,
                    help="Target training set size (stratified subsample after split)")
    ap.add_argument("--n-dev", type=int, default=300)
    ap.add_argument("--n-test", type=int, default=1000)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # ── 1. GeoQA ─────────────────────────────────────────────────────────
    print("=" * 70)
    print("GEOQA")
    print("=" * 70)

    all_geoqa = load_jsonl(GEOQA_NORM)
    # Filter to valid images and content
    all_geoqa = [r for r in all_geoqa
                 if has_image(r) and r.get("question", "").strip() and r.get("gt") is not None]
    print(f"Loaded: {len(all_geoqa)} (with valid images)")

    geo3k = [r for r in all_geoqa if "geo3k" in r.get("id", "")]
    geoqa_plus = [r for r in all_geoqa if "geoqa_plus" in r.get("id", "")]
    print(f"  geo3k: {len(geo3k)}")
    print(f"  geoqa_plus: {len(geoqa_plus)}")

    # Score ALL geoqa together, assign difficulty
    # Weights: think_len 50%, caption_len 20%, gt_complexity 30%
    assign_difficulty(all_geoqa, geoqa_raw_scores, [0.50, 0.20, 0.30])

    # Re-split after difficulty assignment
    geo3k = [r for r in all_geoqa if "geo3k" in r.get("id", "")]
    geoqa_plus = [r for r in all_geoqa if "geoqa_plus" in r.get("id", "")]

    # Show difficulty distribution
    for name, subset in [("geo3k", geo3k), ("geoqa_plus", geoqa_plus)]:
        dc = Counter(r["difficulty"] for r in subset)
        print(f"  {name} difficulty: {dict(sorted(dc.items()))}")

    # Sample geoqa_plus randomly
    n_gp = min(args.geoqa_plus_n, len(geoqa_plus))
    rng.shuffle(geoqa_plus)
    selected_gp = geoqa_plus[:n_gp]
    print(f"\n  Sampled {n_gp} geoqa_plus (from {len(geoqa_plus)})")
    dc_gp = Counter(r["difficulty"] for r in selected_gp)
    print(f"  Selected difficulty: {dict(sorted(dc_gp.items()))}")

    # Tag source
    all_geoqa_selected = geo3k + selected_gp
    for r in all_geoqa_selected:
        r["_source"] = "geoqa"

    dc = Counter(r["difficulty"] for r in all_geoqa_selected)
    print(f"  Total geoqa (geo3k + selected geoqa_plus): {len(all_geoqa_selected)}")
    print(f"  Difficulty: {dict(sorted(dc.items()))}")

    # ── 2. Synthesis ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SYNTHESIS")
    print("=" * 70)

    all_synth = load_jsonl(SYNTH_NORM)
    all_synth = [r for r in all_synth
                 if has_image(r) and r.get("question", "").strip() and r.get("gt") is not None]
    print(f"Loaded: {len(all_synth)} (with valid images)")

    # Drop gen_plane
    all_synth = [r for r in all_synth if synth_category(r.get("image", "")) != "DROP"]
    print(f"After dropping gen_plane: {len(all_synth)}")

    # Score ALL synthesis together, assign difficulty
    # Weights: think_len 50%, gt_complexity 50%
    assign_difficulty(all_synth, synth_raw_scores, [0.50, 0.50])

    # Bucket by category
    keep_all_rows = []
    downsample_buckets = defaultdict(list)
    uncategorized = []

    for r in all_synth:
        cat = synth_category(r.get("image", ""))
        if cat in SYNTH_KEEP_ALL:
            keep_all_rows.append(r)
        elif cat in SYNTH_DOWNSAMPLE:
            downsample_buckets[cat].append(r)
        elif cat is None:
            uncategorized.append(r)

    print(f"\n  Keep-all categories:")
    for cat in sorted(SYNTH_KEEP_ALL):
        cnt = sum(1 for r in keep_all_rows if cat in r.get("image", ""))
        dc = Counter(r["difficulty"] for r in keep_all_rows if cat in r.get("image", ""))
        print(f"    {cat}: {cnt}  {dict(sorted(dc.items()))}")

    # Downsample categories: take N randomly per category
    print(f"\n  Downsample categories ({args.synth_per_cat} each):")
    selected_synth = list(keep_all_rows)
    for cat in sorted(SYNTH_DOWNSAMPLE):
        pool = downsample_buckets[cat]
        n_cat = min(args.synth_per_cat, len(pool))
        rng.shuffle(pool)
        chosen = pool[:n_cat]
        dc_after = Counter(r["difficulty"] for r in chosen)
        print(f"    {cat}: {len(pool)} -> {n_cat}  {dict(sorted(dc_after.items()))}")
        selected_synth.extend(chosen)

    if uncategorized:
        print(f"\n  Uncategorized: {len(uncategorized)} (included)")
        selected_synth.extend(uncategorized)

    for r in selected_synth:
        r["_source"] = "synthesis"

    dc = Counter(r["difficulty"] for r in selected_synth)
    print(f"\n  Total synthesis: {len(selected_synth)}")
    print(f"  Difficulty: {dict(sorted(dc.items()))}")

    # ── 3. Zebra-CoT (train only) ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ZEBRA-COT")
    print("=" * 70)

    zebra = load_jsonl(ZEBRA_NORM)
    # Re-score with new percentile scheme
    assign_difficulty(zebra, zebra_raw_scores, [0.35, 0.25, 0.15, 0.25])
    for r in zebra:
        r["_source"] = "zebra_cot"
    dc = Counter(r["difficulty"] for r in zebra)
    print(f"  Total: {len(zebra)} (all train-only)")
    print(f"  Difficulty: {dict(sorted(dc.items()))}")

    # ── 4. Stratified split ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SPLITTING")
    print("=" * 70)

    pool = all_geoqa_selected + selected_synth
    print(f"\nTotal pool (geoqa + synthesis): {len(pool)}")

    train, dev, test = stratified_split(pool, args.n_dev, args.n_test, rng)

    print(f"\n  Before cap — Train (geoqa+synth): {len(train)}  Dev: {len(dev)}  Test: {len(test)}")
    print(f"  Zebra-CoT (all → train): {len(zebra)}")

    # Subsample geoqa+synthesis train portion, then add ALL zebra
    n_non_zebra = args.n_train - len(zebra)
    diff_targets = {
        3: round(n_non_zebra * 0.20),
        2: round(n_non_zebra * 0.60),
        1: n_non_zebra - round(n_non_zebra * 0.20) - round(n_non_zebra * 0.60),
    }
    print(f"  Target geoqa+synth: {n_non_zebra}  (d=1:{diff_targets[1]}, d=2:{diff_targets[2]}, d=3:{diff_targets[3]})")
    train = targeted_subsample(train, diff_targets, rng)

    # Add ALL zebra_cot
    train.extend(zebra)
    rng.shuffle(train)

    print(f"\n  Train: {len(train)}")
    print(f"  Dev:   {len(dev)}")
    print(f"  Test:  {len(test)}")

    # ── 5. Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for name, split in [("Train", train), ("Dev", dev), ("Test", test)]:
        print(f"\n{name}: {len(split)}")
        dc = Counter(r["difficulty"] for r in split)
        sc = Counter(r["_source"] for r in split)
        print(f"  by difficulty: {dict(sorted(dc.items()))}")
        print(f"  by source:     {dict(sorted(sc.items()))}")

        # Cross-tabulation
        cross = Counter((r["_source"], r["difficulty"]) for r in split)
        sources = sorted(set(r["_source"] for r in split))
        diffs = sorted(set(r["difficulty"] for r in split))
        print(f"  {'source':<15} " + " ".join(f"d={d:>3}" for d in diffs) + "  total")
        for s in sources:
            vals = [cross.get((s, d), 0) for d in diffs]
            print(f"  {s:<15} " + " ".join(f"{v:>5}" for v in vals) + f"  {sum(vals):>5}")

    # Verify no ID overlap between splits
    def get_ids(split):
        return {str(r.get("id", r.get("index", ""))) for r in split}

    train_ids = get_ids(train)
    dev_ids = get_ids(dev)
    test_ids = get_ids(test)
    assert not (train_ids & dev_ids), f"Train/Dev overlap: {train_ids & dev_ids}"
    assert not (train_ids & test_ids), f"Train/Test overlap: {train_ids & test_ids}"
    assert not (dev_ids & test_ids), f"Dev/Test overlap: {dev_ids & test_ids}"
    print("\nNo ID overlaps between splits ✓")

    # ── 6. Write ─────────────────────────────────────────────────────────
    if args.dry_run:
        print("\n[DRY RUN] Would write to:", SPLITS_DIR)
    else:
        write_jsonl(train, SPLITS_DIR / "train.jsonl")
        write_jsonl(dev, SPLITS_DIR / "val.jsonl")
        write_jsonl(test, SPLITS_DIR / "test.jsonl")
        print(f"\nWrote merged splits to {SPLITS_DIR}/")

        # Per-dataset splits
        for src in ("geoqa", "synthesis", "zebra_cot"):
            for sname, split in [("train", train), ("val", dev), ("test", test)]:
                src_rows = [r for r in split if r.get("_source") == src]
                if src_rows:
                    write_jsonl(src_rows, SPLITS_DIR / src / f"{sname}.jsonl")

        print(f"Wrote per-dataset splits to {SPLITS_DIR}/<dataset>/")

    print("\nDone!")


if __name__ == "__main__":
    main()
