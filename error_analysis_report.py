#!/usr/bin/env python3
"""
error_analysis_report.py — Dev-set error analysis for AmericasNLP Guaraní pipeline.

Stage 1: Rule-based heuristic pre-analysis (always runs)
Stage 2: LLM-assisted labeling via Gemini API (optional, --use-llm)
Stage 3: Generates TSV + JSON + Markdown report

Usage (heuristic only):
    python error_analysis_report.py \
        --input-spanish outputs/guarani/72b_v3_clean_spanish.txt \
        --predictions   outputs/guarani/gemini_flash_r80_d49_preds.txt \
        --references    outputs/guarani/dev_references.txt \
        --per-example-tsv outputs/guarani/gemini_flash_r80_d49_per_example.tsv \
        --output-dir    error_analysis/guarani_gemini_r80 \
        --language guarani

Usage (with LLM labeling):
    python error_analysis_report.py \
        --input-spanish outputs/guarani/72b_v3_clean_spanish.txt \
        --predictions   outputs/guarani/gemini_flash_r80_d49_preds.txt \
        --references    outputs/guarani/dev_references.txt \
        --per-example-tsv outputs/guarani/gemini_flash_r80_d49_per_example.tsv \
        --output-dir    error_analysis/guarani_gemini_r80 \
        --language guarani \
        --use-llm \
        --api-key $GEMINI_API_KEY

Optional baseline comparison:
    --baseline-preds   outputs/guarani/baseline_preds.txt \
    --baseline-tsv     outputs/guarani/baseline_per_example.tsv
"""

from __future__ import annotations
import argparse
import csv
import json
import math
import os
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SCORE_LOW = 20.0
SCORE_MID = 40.0
SCORE_HIGH = 60.0

SCORE_BAND_LABELS = {
    "very_low":  f"< {SCORE_LOW}",
    "low":       f"{SCORE_LOW}–{SCORE_MID}",
    "mid":       f"{SCORE_MID}–{SCORE_HIGH}",
    "high":      f"> {SCORE_HIGH}",
}

# Markers common in Guaraní that don't appear in Spanish
GUARANI_MARKERS = [
    "ha", "pe", "ndive", "gui", "rehe", "kuri", "niko", "avei",
    "peve", "rire", "rupi", "ári", "upe", "kóva", "ko'ã", "chupe",
    "oĩ", "oiko", "opyta", "oguereko", "ombohasa", "oñepyrũ",
    "jajapó", "ñande", "ore", "pende", "umi", "upéicha",
]

# Spanish tokens that should NOT appear verbatim in Guaraní output
SPANISH_LEAKAGE_WORDS = [
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "en", "con", "por", "para", "que", "se", "su", "sus",
    "de", "del", "al", "una", "hay", "está", "están",
    "mujer", "hombre", "niño", "niña", "persona", "personas",
    "mercado", "casa", "árbol", "flor", "comida", "ropa",
    "vestido", "sombrero", "cesta", "madera",
]

# Vague / speculation markers in the Spanish caption
VAGUE_CAPTION_MARKERS = [
    "parece", "probablemente", "podría ser", "tal vez", "quizá",
    "posiblemente", "como si fuera", "evoca", "simboliza",
    "cultura vibrante", "hermosa escena",
]

# Specific named-place hallucination markers
HALLUCINATED_PLACE_MARKERS = [
    "mercado 4", "caacupé", "corrientes", "concepción",
    "plaza 25 de mayo", "plaza 25 de abril", "parque 12 de octubre",
    "ybyty curupay",
]


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    return re.sub(r"\s+", " ", text.replace("\n", " ")).strip()


def load_per_example_tsv(path: str) -> Dict[int, Dict]:
    """Load per-example TSV from score_outputs.py output."""
    result = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                idx = int(row.get("id", row.get("idx", row.get("index", -1))))
            except (ValueError, TypeError):
                continue

            score = row.get("chrfpp_sentence", row.get("chrf_score", ""))
            row["chrf_score"] = score
            result[idx] = row
    return result

# ─────────────────────────────────────────────────────────────────────────────
# Score computation (chrF++ approximation via sacrebleu if available)
# ─────────────────────────────────────────────────────────────────────────────

def compute_chrf_single(hypothesis: str, reference: str) -> float:
    """Compute sentence-level chrF++ score."""
    try:
        from sacrebleu.metrics import CHRF
        chrf = CHRF(word_order=2)
        score = chrf.sentence_score(hypothesis, [reference])
        return round(score.score, 2)
    except ImportError:
        # Fallback: character n-gram F1 (order=1, no word order)
        return _chrf_fallback(hypothesis, reference)


def _chrf_fallback(hyp: str, ref: str, beta: float = 2.0) -> float:
    """Simple char-bigram F-score fallback."""
    def ngrams(s, n):
        return Counter([s[i:i+n] for i in range(len(s) - n + 1)])
    hyp_ng = ngrams(hyp, 2)
    ref_ng = ngrams(ref, 2)
    if not hyp_ng or not ref_ng:
        return 0.0
    matches = sum((hyp_ng & ref_ng).values())
    p = matches / sum(hyp_ng.values()) if hyp_ng else 0.0
    r = matches / sum(ref_ng.values()) if ref_ng else 0.0
    if p + r == 0:
        return 0.0
    f = (1 + beta**2) * p * r / (beta**2 * p + r)
    return round(f * 100, 2)


def score_band(score: float) -> str:
    if score < SCORE_LOW:
        return "very_low"
    elif score < SCORE_MID:
        return "low"
    elif score < SCORE_HIGH:
        return "mid"
    else:
        return "high"


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_heuristic_features(
    es: str, pred: str, ref: str
) -> Dict:
    """Return a dict of heuristic signal flags and counts."""
    es_n = normalize(es).lower()
    pred_n = normalize(pred).lower()
    ref_n = normalize(ref).lower()

    pred_tokens = pred_n.split()
    ref_tokens = ref_n.split()
    es_tokens = es_n.split()

    # ── Caption quality signals ──
    caption_vague = any(m in es_n for m in VAGUE_CAPTION_MARKERS)
    caption_hallucinated_place = any(m in es_n for m in HALLUCINATED_PLACE_MARKERS)
    caption_very_short = len(es_tokens) < 5
    caption_very_long = len(es_tokens) > 45

    # ── Prediction quality signals ──
    # Spanish token leakage: ES words appearing verbatim in prediction
    leaked_tokens = [w for w in SPANISH_LEAKAGE_WORDS if w in pred_tokens]
    spanish_leak_count = len(leaked_tokens)
    spanish_leak_ratio = spanish_leak_count / max(len(pred_tokens), 1)

    # Repetition: any 3-token repeated span
    repetition = _detect_repetition(pred_tokens)

    # Empty / very short prediction
    pred_empty = len(pred_tokens) < 2
    pred_very_short = len(pred_tokens) < 4

    # Length ratio vs reference
    len_ratio = len(pred_tokens) / max(len(ref_tokens), 1)
    pred_much_shorter = len_ratio < 0.5
    pred_much_longer = len_ratio > 2.0

    # Guaraní marker presence (sanity check that output is actually GN)
    guarani_marker_count = sum(1 for m in GUARANI_MARKERS if m in pred_n)
    likely_not_guarani = guarani_marker_count == 0 and len(pred_tokens) > 3

    # Overlap between prediction and reference
    pred_chars = set(pred_n)
    ref_chars = set(ref_n)
    char_overlap = len(pred_chars & ref_chars) / max(len(pred_chars | ref_chars), 1)

    # Token overlap
    pred_tok_set = set(pred_tokens)
    ref_tok_set = set(ref_tokens)
    tok_overlap = len(pred_tok_set & ref_tok_set) / max(len(pred_tok_set | ref_tok_set), 1)

    return {
        # Caption signals
        "caption_vague": caption_vague,
        "caption_hallucinated_place": caption_hallucinated_place,
        "caption_very_short": caption_very_short,
        "caption_very_long": caption_very_long,
        # Translation signals
        "spanish_leak_count": spanish_leak_count,
        "spanish_leak_ratio": round(spanish_leak_ratio, 3),
        "leaked_tokens": leaked_tokens[:5],
        "repetition": repetition,
        "pred_empty": pred_empty,
        "pred_very_short": pred_very_short,
        "pred_much_shorter": pred_much_shorter,
        "pred_much_longer": pred_much_longer,
        "len_ratio": round(len_ratio, 3),
        "likely_not_guarani": likely_not_guarani,
        "guarani_marker_count": guarani_marker_count,
        # Overlap
        "char_overlap": round(char_overlap, 3),
        "tok_overlap": round(tok_overlap, 3),
    }


def _detect_repetition(tokens: List[str], span: int = 3) -> bool:
    if len(tokens) < span * 2:
        return False
    for i in range(len(tokens) - span * 2 + 1):
        if tokens[i:i+span] == tokens[i+span:i+span*2]:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic root cause labeling
# ─────────────────────────────────────────────────────────────────────────────

def assign_heuristic_label(
    score: float,
    features: Dict,
) -> Tuple[str, str, str]:
    """
    Returns (primary_category, secondary_category, subtype).

    Categories:
      - vision_based     : error likely originates from the captioner
      - translation_based: caption looks OK, but GN translation is wrong
      - metric_mismatch  : translation is plausible, score is low due to wording gap
      - mixed_other      : multiple stages failed or uncertain
    """

    f = features

    # ── Vision-based ──
    if f["caption_hallucinated_place"]:
        return ("vision_based", "hallucinated_location", "specific_place_name_injected")
    if f["caption_vague"]:
        return ("vision_based", "uncertain_description", "speculation_marker_in_caption")
    if f["caption_very_short"] and score < SCORE_MID:
        return ("vision_based", "under_description", "caption_too_short")

    # ── Translation-based ──
    if f["pred_empty"] or f["pred_very_short"]:
        return ("translation_based", "empty_or_degenerate", "prediction_too_short")
    if f["repetition"]:
        return ("translation_based", "repetition_degeneration", "repeated_token_span")
    if f["likely_not_guarani"]:
        return ("translation_based", "wrong_language", "no_guarani_markers")
    if f["spanish_leak_ratio"] > 0.25:
        return ("translation_based", "spanish_leakage", "high_spanish_token_ratio")
    if f["pred_much_shorter"] and score < SCORE_MID:
        return ("translation_based", "omission", "prediction_much_shorter_than_reference")
    if f["pred_much_longer"] and score < SCORE_MID:
        return ("translation_based", "over_generation", "prediction_much_longer_than_reference")

    # ── Metric mismatch (score is low but translation could be valid) ──
    if score >= SCORE_MID:
        # High-scoring: this is not an error case
        return ("no_error", "high_score", "acceptable_translation")
    if 0.3 <= f["tok_overlap"] <= 0.7 and score < SCORE_MID:
        return ("metric_mismatch", "paraphrase_gap", "different_valid_wording")
    if f["len_ratio"] > 0.8 and score < SCORE_LOW:
        return ("metric_mismatch", "style_divergence", "similar_length_low_overlap")

    # ── Mixed / other ──
    if f["caption_very_long"] and score < SCORE_MID:
        return ("mixed_other", "complex_input", "long_caption_low_score")
    if score < SCORE_LOW:
        return ("mixed_other", "uncertain", "low_score_no_dominant_signal")

    return ("mixed_other", "uncertain", "mid_score_ambiguous")


# ─────────────────────────────────────────────────────────────────────────────
# LLM-assisted labeling (optional, via Gemini)
# ─────────────────────────────────────────────────────────────────────────────

LLM_LABEL_SYSTEM = """You are an expert annotator for a Spanish→Guaraní image captioning translation system.
You will be given a source Spanish caption, a model prediction in Guaraní, and the gold reference.
Your job is to identify the root cause of any translation error.

Respond ONLY with a valid JSON object. No prose, no markdown fences.
Schema:
{
  "primary_category": one of ["vision_based", "translation_based", "metric_mismatch", "no_error", "mixed_other"],
  "secondary_category": short string (e.g. "hallucinated_location", "omission", "paraphrase_gap"),
  "subtype": short descriptive string,
  "confidence": one of ["high", "medium", "low"],
  "evidence": short string noting specific tokens or phrases that led to your judgment,
  "comment": one sentence in English summarizing the issue
}

Category definitions:
- vision_based: the Spanish caption itself is wrong, vague, or hallucinated — the error enters the pipeline at the vision/captioning stage
- translation_based: the Spanish caption is reasonable, but the Guaraní translation is incorrect (omission, leakage, degeneration, wrong lexical choice, wrong structure)
- metric_mismatch: the translation looks plausible but the automatic chrF++ score penalizes it due to wording/style divergence from the reference
- no_error: the translation is acceptable
- mixed_other: multiple error sources or impossible to determine

Do not explain your reasoning. Output only the JSON.
"""

LLM_LABEL_USER_TEMPLATE = """
chrF++ score: {score}

Spanish caption (from vision model):
{spanish}

Model prediction (Guaraní):
{prediction}

Gold reference (Guaraní):
{reference}

Heuristic signals (from automatic analysis):
{heuristic_signals}
"""


def call_gemini_label(
    client,
    model: str,
    es: str,
    pred: str,
    ref: str,
    score: float,
    features: Dict,
    max_retries: int = 3,
) -> Dict:
    """Call Gemini for LLM-assisted error label. Returns label dict."""
    heuristic_signals = "; ".join([
        f"{k}={v}" for k, v in features.items()
        if v and v not in (0, 0.0, [], False)
    ])

    user_prompt = LLM_LABEL_USER_TEMPLATE.format(
        score=score,
        spanish=normalize(es),
        prediction=normalize(pred),
        reference=normalize(ref),
        heuristic_signals=heuristic_signals or "none",
    )

    for attempt in range(max_retries):
        try:
            from google import genai as gai
            response = client.models.generate_content(
                model=model,
                contents=user_prompt,
                config=gai.types.GenerateContentConfig(
                    system_instruction=LLM_LABEL_SYSTEM,
                    max_output_tokens=256,
                    temperature=0.0,
                ),
            )
            text = response.text.strip()
            # Strip markdown fences if present
            text = re.sub(r"^```[a-z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
            label = json.loads(text)
            return label
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return {
                "primary_category": "llm_error",
                "secondary_category": "api_failure",
                "subtype": str(e)[:80],
                "confidence": "low",
                "evidence": "",
                "comment": f"LLM labeling failed: {e}",
            }


# ─────────────────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_markdown_report(
    examples: List[Dict],
    output_dir: str,
    run_name: str,
    language: str,
    use_llm: bool,
) -> str:
    total = len(examples)
    scores = [e["chrf_score"] for e in examples]
    mean_score = sum(scores) / max(total, 1)

    # Band counts
    band_counts = Counter(e["score_band"] for e in examples)

    # Category counts (prefer LLM label if available)
    cat_key = "llm_primary_category" if use_llm else "heuristic_primary_category"
    cat_counts = Counter(e.get(cat_key, e["heuristic_primary_category"]) for e in examples)
    sub_counts = Counter()
    for e in examples:
        sub = e.get("llm_secondary_category" if use_llm else "heuristic_secondary_category", "")
        if sub:
            sub_counts[sub] += 1

    # Sort examples by score ascending (worst first)
    sorted_examples = sorted(examples, key=lambda e: e["chrf_score"])

    lines = []
    lines.append(f"# Error Analysis Report — {run_name}")
    lines.append(f"\n**Language**: {language}  |  **Total examples**: {total}  |  **LLM labels**: {'yes' if use_llm else 'heuristic only'}\n")

    lines.append("---\n")
    lines.append("## 1. Score Distribution\n")
    lines.append(f"| Band | Range | Count | % |")
    lines.append(f"|------|-------|-------|---|")
    for band, label in SCORE_BAND_LABELS.items():
        n = band_counts.get(band, 0)
        lines.append(f"| {band} | {label} | {n} | {100*n/max(total,1):.1f}% |")
    lines.append(f"\n**Mean chrF++**: {mean_score:.2f}  |  **Min**: {min(scores):.2f}  |  **Max**: {max(scores):.2f}\n")

    lines.append("---\n")
    lines.append("## 2. Root Cause Distribution\n")
    lines.append(f"| Category | Count | % |")
    lines.append(f"|----------|-------|---|")
    for cat, n in cat_counts.most_common():
        lines.append(f"| {cat} | {n} | {100*n/max(total,1):.1f}% |")

    lines.append(f"\n### 2.1 Subtypes\n")
    lines.append(f"| Subtype | Count |")
    lines.append(f"|---------|-------|")
    for sub, n in sub_counts.most_common(15):
        lines.append(f"| {sub} | {n} |")

    lines.append("\n---\n")
    lines.append("## 3. Worst Examples (chrF++ < 20)\n")
    worst = [e for e in sorted_examples if e["chrf_score"] < SCORE_LOW][:20]
    if not worst:
        lines.append("_No examples below 20 chrF++._\n")
    else:
        for e in worst:
            _add_example_block(lines, e, use_llm)

    lines.append("\n---\n")
    lines.append("## 4. Representative Examples by Error Category\n")

    seen_cats = set()
    for e in sorted_examples:
        cat = e.get(cat_key, e["heuristic_primary_category"])
        if cat in seen_cats or cat in ("no_error",):
            continue
        seen_cats.add(cat)
        lines.append(f"\n### Category: `{cat}`\n")
        _add_example_block(lines, e, use_llm)

    lines.append("\n---\n")
    lines.append("## 5. Best Examples (chrF++ > 60)\n")
    best = [e for e in sorted_examples if e["chrf_score"] > SCORE_HIGH][-10:]
    if not best:
        lines.append("_No examples above 60 chrF++._\n")
    else:
        for e in best:
            _add_example_block(lines, e, use_llm)

    lines.append("\n---\n")
    lines.append("## 6. Heuristic Signal Summary\n")
    signal_keys = [
        "caption_vague", "caption_hallucinated_place", "caption_very_short",
        "repetition", "spanish_leak_count", "pred_empty", "pred_very_short",
        "pred_much_shorter", "pred_much_longer", "likely_not_guarani",
    ]
    lines.append("| Signal | Count | % |")
    lines.append("|--------|-------|---|")
    for key in signal_keys:
        hits = sum(1 for e in examples if e["features"].get(key))
        if hits > 0:
            lines.append(f"| {key} | {hits} | {100*hits/max(total,1):.1f}% |")

    report = "\n".join(lines)
    report_path = os.path.join(output_dir, "error_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    return report_path


def _add_example_block(lines: List[str], e: Dict, use_llm: bool) -> None:
    cat_key = "llm_primary_category" if use_llm else "heuristic_primary_category"
    sub_key = "llm_secondary_category" if use_llm else "heuristic_secondary_category"
    comment_key = "llm_comment" if use_llm else "heuristic_subtype"

    lines.append(f"<details>")
    lines.append(f"<summary><b>Example {e['idx']}</b> — chrF++: <code>{e['chrf_score']}</code> — {e.get(cat_key, '?')}/{e.get(sub_key, '?')}</summary>\n")
    lines.append(f"**Spanish caption**: {e['spanish']}\n")
    lines.append(f"**Prediction (GN)**: {e['prediction']}\n")
    lines.append(f"**Reference (GN)**: {e['reference']}\n")
    if use_llm and e.get("llm_comment"):
        lines.append(f"**Diagnosis**: {e['llm_comment']}\n")
    if e.get("features", {}).get("leaked_tokens"):
        lines.append(f"**Leaked ES tokens**: {e['features']['leaked_tokens']}\n")
    lines.append(f"</details>\n")


# ─────────────────────────────────────────────────────────────────────────────
# Write TSV
# ─────────────────────────────────────────────────────────────────────────────

def write_labeled_tsv(examples: List[Dict], path: str, use_llm: bool) -> None:
    fieldnames = [
        "idx", "chrf_score", "score_band",
        "heuristic_primary_category", "heuristic_secondary_category", "heuristic_subtype",
    ]
    if use_llm:
        fieldnames += [
            "llm_primary_category", "llm_secondary_category", "llm_subtype",
            "llm_confidence", "llm_evidence", "llm_comment",
        ]
    fieldnames += [
        "spanish", "prediction", "reference",
        "caption_vague", "caption_hallucinated_place", "caption_very_short",
        "spanish_leak_ratio", "repetition", "pred_very_short",
        "pred_much_shorter", "pred_much_longer", "len_ratio",
        "likely_not_guarani", "guarani_marker_count",
        "char_overlap", "tok_overlap",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for e in examples:
            row = {k: e.get(k, "") for k in fieldnames}
            for fkey in ["caption_vague", "caption_hallucinated_place", "caption_very_short",
                         "repetition", "pred_very_short", "pred_much_shorter",
                         "pred_much_longer", "likely_not_guarani"]:
                row[fkey] = e["features"].get(fkey, "")
            for fkey in ["spanish_leak_ratio", "len_ratio", "char_overlap", "tok_overlap",
                         "guarani_marker_count"]:
                row[fkey] = e["features"].get(fkey, "")
            writer.writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Dev-set error analysis for AmericasNLP Guaraní pipeline."
    )
    ap.add_argument("--input-spanish",     required=True, help="Spanish captions (one per line).")
    ap.add_argument("--predictions",       required=True, help="Model predictions (one per line).")
    ap.add_argument("--references",        required=True, help="Gold references (one per line).")
    ap.add_argument("--per-example-tsv",   default=None,  help="Per-example TSV from translate script (for extra scores).")
    ap.add_argument("--baseline-preds",    default=None,  help="Optional baseline predictions for comparison.")
    ap.add_argument("--baseline-tsv",      default=None,  help="Optional baseline per-example TSV.")
    ap.add_argument("--output-dir",        required=True, help="Directory for all output files.")
    ap.add_argument("--language",          default="guarani", help="Language name for the report.")
    ap.add_argument("--run-name",          default=None,  help="Run name (defaults to output-dir basename).")
    ap.add_argument("--use-llm",           action="store_true", help="Use Gemini for LLM-assisted labeling.")
    ap.add_argument("--api-key",           default="",    help="Gemini API key (or set GEMINI_API_KEY).")
    ap.add_argument("--llm-model",         default="gemini-2.5-flash", help="Gemini model for labeling.")
    ap.add_argument("--limit",             type=int, default=None, help="Limit to first N examples.")
    ap.add_argument("--sleep",             type=float, default=0.1, help="Sleep between Gemini calls.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    run_name = args.run_name or Path(args.output_dir).name
    ensure_dir(args.output_dir)

    print(f"Loading data...", flush=True)
    inputs    = read_lines(args.input_spanish)
    preds     = read_lines(args.predictions)
    refs      = read_lines(args.references)

    n = min(len(inputs), len(preds), len(refs))
    if args.limit:
        n = min(n, args.limit)
    inputs, preds, refs = inputs[:n], preds[:n], refs[:n]
    print(f"  {n} examples loaded.")

    # Load per-example TSV if available
    per_ex_data: Dict[int, Dict] = {}
    if args.per_example_tsv and os.path.exists(args.per_example_tsv):
        per_ex_data = load_per_example_tsv(args.per_example_tsv)
        print(f"  Per-example TSV loaded: {len(per_ex_data)} rows.")

    baseline_preds = None
    baseline_per_ex = {}
    if args.baseline_preds:
        baseline_preds = read_lines(args.baseline_preds)[:n]
    if args.baseline_tsv and os.path.exists(args.baseline_tsv):
        baseline_per_ex = load_per_example_tsv(args.baseline_tsv)

    # LLM client setup
    llm_client = None
    if args.use_llm:
        api_key = args.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("ERROR: --use-llm requires GEMINI_API_KEY env var or --api-key")
            sys.exit(1)
        try:
            from google import genai as gai
            llm_client = gai.Client(api_key=api_key)
            print(f"  Gemini client ready ({args.llm_model}).")
        except ImportError:
            print("ERROR: google-genai not installed. Run: pip install google-genai --break-system-packages")
            sys.exit(1)

    # ── Per-example analysis ──
    print(f"\nRunning analysis on {n} examples...", flush=True)
    examples = []
    for i in range(n):
        es   = normalize(inputs[i])
        pred = normalize(preds[i])
        ref  = normalize(refs[i])

        # Score: use TSV if available, else compute
        if i in per_ex_data and "chrf_score" in per_ex_data[i]:
            try:
                chrf = float(per_ex_data[i]["chrf_score"])
            except ValueError:
                chrf = compute_chrf_single(pred, ref)
        else:
            chrf = compute_chrf_single(pred, ref)

        features = extract_heuristic_features(es, pred, ref)
        h_cat, h_sub, h_subtype = assign_heuristic_label(chrf, features)

        example = {
            "idx":               i,
            "chrf_score":        chrf,
            "score_band":        score_band(chrf),
            "spanish":           es,
            "prediction":        pred,
            "reference":         ref,
            "features":          features,
            "heuristic_primary_category":   h_cat,
            "heuristic_secondary_category": h_sub,
            "heuristic_subtype": h_subtype,
        }

        # Baseline comparison
        if baseline_preds:
            b_pred = normalize(baseline_preds[i])
            if i in baseline_per_ex and "chrf_score" in baseline_per_ex[i]:
                try:
                    b_chrf = float(baseline_per_ex[i]["chrf_score"])
                except ValueError:
                    b_chrf = compute_chrf_single(b_pred, ref)
            else:
                b_chrf = compute_chrf_single(b_pred, ref)
            example["baseline_pred"]  = b_pred
            example["baseline_chrf"]  = b_chrf
            example["delta_chrf"]     = round(chrf - b_chrf, 2)

        # LLM labeling
        if llm_client:
            label = call_gemini_label(
                client=llm_client,
                model=args.llm_model,
                es=es, pred=pred, ref=ref,
                score=chrf, features=features,
            )
            example["llm_primary_category"]   = label.get("primary_category", "")
            example["llm_secondary_category"]  = label.get("secondary_category", "")
            example["llm_subtype"]             = label.get("subtype", "")
            example["llm_confidence"]          = label.get("confidence", "")
            example["llm_evidence"]            = label.get("evidence", "")
            example["llm_comment"]             = label.get("comment", "")
            time.sleep(args.sleep)

        examples.append(example)

        if (i + 1) % 10 == 0 or i == n - 1:
            print(f"  [{i+1}/{n}] idx={i} score={chrf:.1f} cat={h_cat}", flush=True)

    # ── Write outputs ──
    print("\nWriting outputs...", flush=True)

    # Labeled TSV
    tsv_path = os.path.join(args.output_dir, "all_examples_labeled.tsv")
    write_labeled_tsv(examples, tsv_path, args.use_llm)
    print(f"  TSV → {tsv_path}")

    # JSON summary
    cat_key = "llm_primary_category" if args.use_llm else "heuristic_primary_category"
    sub_key = "llm_secondary_category" if args.use_llm else "heuristic_secondary_category"
    scores = [e["chrf_score"] for e in examples]

    summary = {
        "run_name": run_name,
        "language": args.language,
        "n_examples": n,
        "use_llm": args.use_llm,
        "mean_chrf": round(sum(scores) / max(n, 1), 2),
        "min_chrf":  min(scores),
        "max_chrf":  max(scores),
        "std_chrf":  round(math.sqrt(sum((s - sum(scores)/n)**2 for s in scores) / max(n, 1)), 2),
        "score_bands": dict(Counter(e["score_band"] for e in examples)),
        "category_counts": dict(Counter(e.get(cat_key, "") for e in examples)),
        "subtype_counts":  dict(Counter(e.get(sub_key, "") for e in examples).most_common(20)),
        "heuristic_signals": {
            k: sum(1 for e in examples if e["features"].get(k))
            for k in ["caption_vague", "caption_hallucinated_place", "caption_very_short",
                      "repetition", "pred_very_short", "pred_much_shorter",
                      "pred_much_longer", "likely_not_guarani"]
        },
    }

    json_path = os.path.join(args.output_dir, "error_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  JSON → {json_path}")

    # Markdown report
    md_path = generate_markdown_report(
        examples=examples,
        output_dir=args.output_dir,
        run_name=run_name,
        language=args.language,
        use_llm=args.use_llm,
    )
    print(f"  MD   → {md_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Language:      {args.language}")
    print(f"Run:           {run_name}")
    print(f"Examples:      {n}")
    print(f"Mean chrF++:   {summary['mean_chrf']}")
    print(f"Score bands:   {summary['score_bands']}")
    print(f"\nRoot causes:")
    for cat, cnt in sorted(summary["category_counts"].items(), key=lambda x: -x[1]):
        print(f"  {cat:<30} {cnt:>4}  ({100*cnt/max(n,1):.1f}%)")
    print(f"\nOutputs in: {args.output_dir}")


if __name__ == "__main__":
    main()