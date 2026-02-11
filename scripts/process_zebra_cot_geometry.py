#!/usr/bin/env python3
"""
Extract the mathematical answer from the "Final Answer" field of the
Zebra-CoT metadata and normalise it into $...$-wrapped LaTeX.

The "Final Answer" field can be:
  - A plain number:            "15"
  - A LaTeX expression:        "$\\frac{7}{8}$"
  - A fraction:                "3/4"
  - An expression with units:  "3√7 inches", "1600π square units"
  - Sentence(s) with answer:   "The value of $x$ is 8."
  - \\boxed{} answers:          "The final answer is $\\boxed{\\frac{1}{3}}$."
  - Multi-line derivation:     "... Thus $m+n = 9+2 = 11.$"

This script extracts the core math expression and then normalises it
with normalize_answer_comprehensive().

Usage:
    python scripts/zebra_cot/extract_and_normalize_final_answer.py
    python scripts/zebra_cot/extract_and_normalize_final_answer.py --input /path/to/metadata.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.extract_answer import normalize_answer_comprehensive


# ── Boxed extraction ─────────────────────────────────────────────────────────

def _extract_boxed(text: str) -> str | None:
    """Extract content from \\boxed{...}, handling nested braces."""
    idx = text.find("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth == 0:
        return text[start : i - 1].strip()
    return None


# ── Sentence-level answer detection ──────────────────────────────────────────

# Words that signal a sentence (not a bare expression)
_SENTENCE_WORDS = {"the", "a", "an", "is", "are", "of", "that", "this", "there"}


def _looks_like_sentence(text: str) -> bool:
    """Does *text* look like an English sentence rather than a bare expression?"""
    words = set(re.sub(r"[^a-zA-Z\s]", "", text.lower()).split())
    return len(_SENTENCE_WORDS & words) >= 2


def _last_latex(text: str) -> str | None:
    """Return last $...$ expression in *text*, or None."""
    matches = re.findall(r"\$[^$]+\$", text)
    return matches[-1].strip() if matches else None


_TRAILING_UNITS_RE = re.compile(
    r"\s*(?:square\s+(?:feet|foot|meters?|units?)|sq\s+units?|"
    r"cubic\s+(?:units?|cm|meters?)|"
    r"degrees?|inches?|feet|foot|centimeters?|meters?|cm|mm|km|AU|"
    r"units?)\s*$",
    re.IGNORECASE,
)

# Number / fraction / unicode-math at the START of a string
_NUM_AT_START_RE = re.compile(
    r"^([+-]?\d+(?:\.\d+)?(?:[/]\d+)?"              # 8, 91/100, 4.5
    r"(?:[π√]\{?\d*\}?)?)"                           # 1600π, 3√7
)


def _extract_num_strip_units(text: str) -> str | None:
    """
    Extract a number/fraction/unicode-math expression from the beginning of
    *text*, stripping any trailing unit words.
    """
    text = text.rstrip(". \t")
    # Strip units first so the number is at the end
    text = _TRAILING_UNITS_RE.sub("", text).strip()
    m = _NUM_AT_START_RE.match(text)
    if m:
        return m.group(1).strip()
    return None


def _find_number_anywhere(text: str) -> str | None:
    """Find the last standalone number in *text*."""
    text = text.rstrip(". \t")
    text = _TRAILING_UNITS_RE.sub("", text).strip()
    m = re.search(
        r"([+-]?\d+(?:\.\d+)?(?:[/]\d+)?(?:[π√]\{?\d*\}?)?)\s*$",
        text,
    )
    if m:
        return m.group(1).strip()
    return None


def _last_number_or_frac(text: str) -> str | None:
    """Return the last number, fraction, or unicode-math token at the end of *text*."""
    text = text.rstrip(". \t")
    text = _TRAILING_UNITS_RE.sub("", text).strip()
    m = re.search(
        r"([+-]?\d+(?:\.\d+)?(?:[/]\d+)?(?:[π√]\{?\d*\}?)?(?:\s*[+\-]\s*\d+(?:\.\d+)?(?:[/]\d+)?(?:[π√]\{?\d*\}?)?)*)\s*$",
        text,
    )
    if m:
        return m.group(1).strip()
    return None


# ── Core extraction ──────────────────────────────────────────────────────────

def extract_final_answer(raw: str) -> tuple[str, str]:
    """
    Extract the mathematical answer from a Final Answer string.

    Returns (extracted_expression, extraction_method).
    """
    text = raw.strip()
    if not text:
        return text, "empty"

    # 1. ── \boxed{...} — highest priority ──
    boxed = _extract_boxed(text)
    if boxed is not None:
        return boxed, "boxed"

    # 2. ── Short / bare expression (no sentence structure) ──
    if not _looks_like_sentence(text) and "\n" not in text:
        return text, "direct"

    # 3. ── Multi-line: take the last meaningful line ──
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    last_line = lines[-1]

    # 4. ── Look for "is" / "are" in the last line (use LAST occurrence) ──
    is_matches = list(re.finditer(r"\b(?:is|are)\s+", last_line))
    for is_match in reversed(is_matches):
        after_is = last_line[is_match.end():]

        # 4a. LaTeX after "is" — prefer this
        latex = _last_latex(after_is)
        if latex:
            return latex, "is_latex"

        # 4b. Number/fraction/unicode at START of text after "is" (strip units)
        num = _extract_num_strip_units(after_is)
        if num:
            return num, "is_plain"

        # 4c. Backtick-wrapped expression: `20000 / pi`
        bt = re.search(r"`([^`]+)`", after_is)
        if bt:
            return bt.group(1).strip(), "is_backtick"

        # 4d. Number somewhere within text after "is" (fallback)
        num = _find_number_anywhere(after_is)
        if num:
            return num, "is_plain_inner"

    # 5. ── Last line is a $...$ equation (e.g. "$m+n = 25+7 = 32$.") ──
    latex = _last_latex(last_line)
    if latex:
        return latex, "lastline_latex"

    # 5b. ── \[...\] display math (e.g. matrices) ──
    disp_match = re.search(r"\\\[(.+?)\\]", last_line, re.DOTALL)
    if disp_match:
        return disp_match.group(1).strip(), "lastline_display"

    # 6. ── Last line has a trailing (x,y) tuple ──
    tuple_match = re.search(r"\(([+-]?\d[^)]*)\)\s*\.?\s*$", last_line)
    if tuple_match:
        return f"({tuple_match.group(1)})", "lastline_tuple"

    # 7. ── Last line is a bare expression (e.g. "(C)", "86") ──
    last_clean = last_line.rstrip(". \t")
    if last_clean and not _looks_like_sentence(last_clean):
        return last_clean, "lastline_text"

    # 8. ── Fallback: search the whole text for last $...$ ──
    latex = _last_latex(text)
    if latex:
        return latex, "fallback_latex"

    return text, "fallback_raw"


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Extract & normalise Final Answer from Zebra-CoT metadata"
    )
    ap.add_argument(
        "--input",
        default="/workspace/vlm-grpo-qwen25vl3b/data/zebra_cot/extracted/metadata.jsonl",
    )
    ap.add_argument(
        "--output",
        default="/workspace/vlm-grpo-qwen25vl3b/data/zebra_cot/normalized_zebra_cot_geometry.jsonl",
    )
    args = ap.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")

    # Counters
    total = 0
    method_counts: dict[str, int] = {}
    changed = []

    with (
        input_path.open("r", encoding="utf-8") as fin,
        output_path.open("w", encoding="utf-8") as fout,
    ):
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            total += 1

            final_answer_raw = row.get("Final Answer", "")

            # Step 1: extract math expression from text
            extracted, method = extract_final_answer(final_answer_raw)
            method_counts[method] = method_counts.get(method, 0) + 1

            # Step 2: normalise
            normalized = normalize_answer_comprehensive(extracted)

            # Store in output row
            new_row = dict(row)
            new_row["extracted_answer"] = normalized

            fout.write(json.dumps(new_row, ensure_ascii=False) + "\n")

            # Track changes for reporting
            if normalized != final_answer_raw:
                changed.append({
                    "index": row.get("index", total - 1),
                    "final_answer_raw": final_answer_raw[:120],
                    "extracted": extracted[:80],
                    "normalized": normalized,
                    "method": method,
                })

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\nProcessed {total} examples.")
    print(f"Changed:  {len(changed)} / {total}")
    print(f"\nExtraction method distribution:")
    for m, c in sorted(method_counts.items(), key=lambda x: -x[1]):
        pct = 100 * c / total if total > 0 else 0
        print(f"  {m:<20}: {c:>4} ({pct:.1f}%)")

    # Show samples per method
    by_method: dict[str, list] = {}
    for c in changed:
        by_method.setdefault(c["method"], []).append(c)

    print(f"\n{'─' * 90}")
    print("Sample extractions by method:")
    for method in sorted(by_method):
        samples = by_method[method][:8]
        print(f"\n  [{method}] ({len(by_method[method])} total)")
        for s in samples:
            raw_short = s["final_answer_raw"]
            if len(raw_short) > 70:
                raw_short = raw_short[:67] + "..."
            print(f"    {raw_short!r}")
            print(f"      → extracted: {s['extracted']!r}")
            print(f"      → normalized: {s['normalized']}")

    print(f"\n{'─' * 90}")
    print(f"Output → {output_path}")


if __name__ == "__main__":
    main()
