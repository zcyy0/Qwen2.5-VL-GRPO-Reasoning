#!/usr/bin/env python3
"""
Reusable answer-extraction and LaTeX-normalisation utilities.

Functions:
    normalize_answer_comprehensive – full normalisation (RHS, units, \\circ, \\text, Unicode, …)
    normalize_geoqa_ground_truth   – geoqa-specific ground-truth normalisation
    extract_model_answer           – robust extraction from <answer>…</answer> model output

Usage:
    from utils.extract_answer import normalize_answer_comprehensive, extract_model_answer
"""

import re
import json
from pathlib import Path


def normalize_geoqa_ground_truth(answer: str) -> str:
    """Normalize a raw extracted answer string into $...$-wrapped LaTeX."""

    # 1. Early exit
    if not answer:
        return answer

    s = str(answer).strip()
    if not s:
        return s

    # 2. Unwrap existing $...$ for processing
    if s.startswith("$") and s.endswith("$") and len(s) > 1:
        s = s[1:-1].strip()

    # 3. Strip choice-letter prefix: "C: 30°" → "30°"
    m = re.match(r"^[A-Da-d]\s*[:：]\s*(.+)$", s)
    if m:
        s = m.group(1).strip()

    # 4. Fullwidth characters
    s = s.replace("：", ":").replace("＝", "=")

    # 5. Chinese "or" (或): take the first alternative
    if "或" in s:
        s = s.split("或")[0].strip()

    # 6. Malformed degree: 55^○^ → 55°
    s = re.sub(r"\^○\^", "°", s)
    s = re.sub(r"\^◦\^", "°", s)

    # 7. Strip units (longest match first)
    # First strip parenthesized units: (cm^{2}), (cm²), etc.
    s = re.sub(r"\((?:cm|m|dm|km|mm)\^?\{?2?\}?\^?2?²?\)", "", s)

    # Strip units inside sqrt braces: √{3米} → √{3}
    s = re.sub(r"(√\{[^}]*?)(?:平方米|厘米|海里|小时|米处|米|度|天|cm²|cm2|cm\^\{2\}|m²|m2|m/s|km|mm|dm|cm|m|㎝)(\})", r"\1\2", s)

    # CJK units (longest first)
    _cjk_units = ["平方米", "厘米/分", "厘米", "海里", "小时", "米处", "米²", "米", "度", "天"]
    for u in _cjk_units:
        if s.endswith(u):
            s = s[: -len(u)]
            break

    # Metric compound units (longest first)
    _metric_compound = [
        "cm^{2}", "cm²", "cm2",
        "m^{2}", "m²", "m2",
        "m/s",
        "km", "mm", "dm",
    ]
    for u in _metric_compound:
        if s.endswith(u):
            s = s[: -len(u)]
            break

    # Metric simple: cm, m — but NOT if preceded by d/k/m (part of dm/km/mm)
    if s.endswith("cm"):
        s = s[:-2]
    elif s.endswith("m") and not s.endswith("dm") and not s.endswith("km") and not s.endswith("mm"):
        # Also don't strip 'm' from \frac or variable contexts — only strip after digit/}/)/π
        if s[:-1] and re.search(r"[\d\})\u03c0]$", s[:-1]):
            s = s[:-1]

    # Special: ㎝
    if s.endswith("㎝"):
        s = s[:-1]  # ㎝ is single char

    s = s.strip()

    # 8. Degree symbol ° → ^{\circ}
    s = s.replace("°", "^{\\circ}")

    # 9. Unicode √{...} → \sqrt{...}; bare √N → \sqrt{N}
    # √{...} form
    s = re.sub(r"√\{([^}]+)\}", r"\\sqrt{\1}", s)
    # bare √ followed by digits
    s = re.sub(r"√(\d+)", r"\\sqrt{\1}", s)
    # bare √ followed by a single letter
    s = re.sub(r"√([a-zA-Z])", r"\\sqrt{\1}", s)

    # 10. Unicode π → \pi (with implicit multiplication: 2π → 2\pi)
    s = s.replace("π", "\\pi")

    # 11. Fix \sqrt without braces: \sqrt2 → \sqrt{2}, \sqrt5\pi → \sqrt{5}\pi
    #     Match \sqrt followed by one or more digits NOT already in braces
    s = re.sub(r"\\sqrt(\d+)", r"\\sqrt{\1}", s)

    # 12. Handle 'cπ' artifact: e.g. after stripping "cm" from "2√{3}cπ" we get "2√{3}c\pi"
    #     Remove stray 'c' before \pi that resulted from partial unit stripping
    s = re.sub(r"c(\\pi)", r"\1", s)

    # 13. Add \ before trig functions (when not already escaped)
    _trig_funcs = ["arcsin", "arccos", "arctan", "sin", "cos", "tan", "log", "ln"]
    for f in _trig_funcs:
        s = re.sub(
            rf"(?<!\\)(?<![a-zA-Z]){f}(?![a-zA-Z])",
            "\\\\" + f,
            s,
        )
    # Also handle trig at start of string (the lookbehind won't match at pos 0 for [a-zA-Z])
    for f in _trig_funcs:
        if s.startswith(f) and not s.startswith("\\" + f):
            s = "\\" + f + s[len(f):]

    # 14. Add space after trig function name before argument (e.g. \tan15 → \tan 15)
    s = re.sub(r"(\\(?:arc)?(?:sin|cos|tan|log|ln))(\d)", r"\1 \2", s)
    # Also \cosA → \cos A
    s = re.sub(r"(\\(?:arc)?(?:sin|cos|tan|log|ln))([A-Z])", r"\1 \2", s)

    # 15. Handle × multiplication sign
    # Remove × before \pi or \sqrt (implicit multiplication)
    s = re.sub(r"×\s*\\", r"\\", s)
    # Other × → \times
    s = s.replace("×", "\\times ")

    # 16. Handle ≤ and ≥
    s = s.replace("≤", "\\leq ")
    s = s.replace("≥", "\\geq ")

    # 17. Handle ≅ (congruent)
    s = s.replace("≅", "\\cong ")

    # Clean up multiple spaces
    s = re.sub(r"  +", " ", s).strip()

    # 18. Wrap in $...$
    return f"${s}$"


# ─── Comprehensive answer normalization ──────────────────────────────────────

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


# English trailing units to strip (longest first to avoid partial matches)
_ENGLISH_UNITS_RE = re.compile(
    r'\s*(?:square\s+units|cubic\s+units|inches|inch|feet|foot|centimeters|meters|cm|units?)\s*$',
    re.IGNORECASE,
)


def _add_backslash_to_functions(s: str) -> str:
    """Add backslash to bare math functions and Greek letters/symbols."""
    for func in _MATH_FUNCTIONS:
        s = re.sub(
            r'(?<!\\)(?<![a-zA-Z])' + re.escape(func) + r'(?![a-zA-Z])',
            '\\\\' + func,
            s
        )
    for sym in _MATH_SYMBOLS:
        s = re.sub(
            r'(?<!\\)(?<![a-zA-Z])' + re.escape(sym) + r'(?![a-zA-Z])',
            '\\\\' + sym,
            s
        )
    # Clean up double backslashes from over-escaping
    s = re.sub(r'\\\\\\\\', r'\\\\', s)
    return s


def _normalize_unicode_math(s: str) -> str:
    """Convert Unicode math symbols to LaTeX commands."""
    # Degree: ° → ^{\circ}
    s = s.replace("°", "^{\\circ}")
    # √{...} → \sqrt{...}
    s = re.sub(r"√\{([^}]+)\}", r"\\sqrt{\1}", s)
    # bare √N → \sqrt{N}
    s = re.sub(r"√(\d+)", r"\\sqrt{\1}", s)
    s = re.sub(r"√([a-zA-Z])", r"\\sqrt{\1}", s)
    # π → \pi
    s = s.replace("π", "\\pi")
    # Fix \sqrt without braces: \sqrt2 → \sqrt{2}
    s = re.sub(r"\\sqrt(\d+)", r"\\sqrt{\1}", s)
    return s


def _extract_rhs(inner: str) -> str:
    """Extract RHS after the last '=' (skip \\leq, \\geq, !=, <=, >=)."""
    if '=' not in inner:
        return inner
    parts = re.split(r'(?<!\\)(?<!<)(?<!>)(?<!!)=', inner)
    if len(parts) > 1:
        rhs = parts[-1].strip()
        if rhs and re.match(r'^[-+\d\\(\{√πa-zA-Z]', rhs):
            return rhs
    return inner


def normalize_answer_comprehensive(answer: str) -> str:
    """
    Comprehensive answer normalisation.  Combines logic from the zebra-cot
    ground-truth pipeline and model-answer normalisation into one function.

    Pipeline (in order):
      1.  Early exit for empty / None
      2.  Unwrap $...$
      3.  Extract RHS after last '='
      4.  Multiple-choice normalisation   ((A) → A)
      5.  Remove \\text{...}              (\\text{ feet} → "")
      6.  Normalise degree markers        (°, \\circ, ^\\circ → ^{\\circ})
      7.  Remove trailing period
      8.  Strip English trailing units    (inches, feet, square units, …)
      9.  Convert programmatic sqrt()     (sqrt(N) → \\sqrt{N})
     10.  Normalise '*' multiplication    (* before \\ removed, else → \\cdot)
     11.  Convert Unicode √ π             (√ → \\sqrt, π → \\pi)
     12.  Fix \\sqrt without braces       (\\sqrt2 → \\sqrt{2})
     13.  Backslash-escape math funcs     (sin → \\sin, cos → \\cos, …)
     14.  Backslash-escape Greek/symbols  (pi → \\pi, alpha → \\alpha, …)
     15.  Clean whitespace
     16.  Wrap in $...$

    Returns a $...$-wrapped LaTeX string.
    """
    if not answer:
        return answer

    s = str(answer).strip()
    if not s:
        return s

    # 2. Unwrap $...$
    if s.startswith('$') and s.endswith('$') and len(s) > 1:
        s = s[1:-1].strip()

    # 3. RHS after last '='
    s = _extract_rhs(s)

    # 4. Multiple-choice
    mc_match = re.fullmatch(r'\(?([A-Da-d])\)?\.?', s.strip())
    if mc_match:
        return f'${mc_match.group(1).upper()}$'

    # 5. Remove \text{...} (usually units like \text{ feet})
    s = re.sub(r'\\text\s*\{[^}]*\}', '', s)

    # 6. Normalise degree markers → ^{\circ}
    s = s.replace("°", "^{\\circ}")
    # Normalise bare \circ and ^\circ to ^{\circ}
    s = re.sub(r'\^\\circ(?!\})', r'^{\\circ}', s)
    s = re.sub(r'(?<!\^)(?<!\{)\\circ', r'^{\\circ}', s)

    # 7. Remove trailing period
    s = re.sub(r'\.\s*$', '', s)

    # 8. Strip English trailing units
    s = _ENGLISH_UNITS_RE.sub('', s).strip()

    # 9. Programmatic sqrt(N) → \sqrt{N}
    s = re.sub(r'sqrt\(([^)]+)\)', r'\\sqrt{\1}', s)

    # 10. Normalise '*' multiplication
    #     * before \ → just \ (e.g. "16*\sqrt{3}" → "16\sqrt{3}")
    s = re.sub(r'\*\\', r'\\', s)
    #     remaining * → \cdot
    s = s.replace('*', ' \\cdot ')

    # 11–12. Unicode → LaTeX (√, π) + fix \sqrt braces
    #         ° already removed in step 6, so _normalize_unicode_math's ° handling is a no-op
    s = _normalize_unicode_math(s)

    # 13–14. Backslash-escape functions and symbols
    s = _add_backslash_to_functions(s)

    # 15. Clean whitespace
    s = re.sub(r'\s+', ' ', s).strip()

    # 16. Wrap in $...$
    return f'${s}$'


# ─── Model-answer extraction ─────────────────────────────────────────────────


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
    matches = list(re.finditer(r'\bis[\s:]+', text))
    if not matches:
        return None
    last = matches[-1]
    after = text[last.end():].strip()
    if not after:
        return None
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
