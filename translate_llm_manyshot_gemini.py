#!/usr/bin/env python3
"""Many-shot Spanish → Indigenous Language translation using Google Gemini.

Single script replacing translate_llm_manyshot_{guarani,nahuatl,bribri,wixarika,maya}_gemini.py.
Pass --language to select the target language; everything else is identical.

Install:
    pip install google-genai rank-bm25 sacrebleu --break-system-packages

Usage:
    export GEMINI_API_KEY="your-key-here"

    # Guaraní (frozen test config)
    python translate_llm_manyshot_gemini.py \
        --language guarani \
        --train-src  data/parallel/guarani_with_dev/train.es \
        --train-tgt  data/parallel/guarani_with_dev/train.gn \
        --input-spanish outputs/guarani/72b_v3_clean_spanish.txt \
        --reference-file outputs/guarani/dev_references.txt \
        --output-preds  outputs/guarani/gemini_flash_r80_d49_preds.txt \
        --metrics-json  outputs/guarani/gemini_flash_r80_d49_metrics.json \
        --per-example-tsv outputs/guarani/gemini_flash_r80_d49_per_example.tsv \
        --samples-tsv   outputs/guarani/gemini_flash_r80_d49_samples.tsv \
        --model gemini-2.5-flash \
        --num-retrieval 80 \
        --num-dev-examples 49 \
        --dev-example-spanish outputs/guarani/72b_v3_clean_spanish.txt \
        --dev-example-refs   outputs/guarani/dev_references.txt \
        --temperature 0.0

    # Nahuatl
    python translate_llm_manyshot_gemini.py \
        --language nahuatl \
        --train-src  data/parallel/nahuatl_with_dev/train.es \
        --train-tgt  data/parallel/nahuatl_with_dev/train.nah \
        --input-spanish outputs/nahuatl/dev_caption_es_v1.txt \
        --output-preds  outputs/nahuatl/gemini_flash_r80_d20_preds.txt \
        --samples-tsv   outputs/nahuatl/gemini_flash_r80_d20_samples.tsv \
        --num-retrieval 80 --num-dev-examples 20 --temperature 0.0

    # Bribri
    python translate_llm_manyshot_gemini.py \
        --language bribri \
        --train-src  data/parallel/bribri_with_dev/train.es \
        --train-tgt  data/parallel/bribri_with_dev/train.bzd \
        --input-spanish outputs/bribri/dev_caption_es_v1.txt \
        --output-preds  outputs/bribri/gemini_flash_r80_d20_preds.txt \
        --samples-tsv   outputs/bribri/gemini_flash_r80_d20_samples.tsv \
        --num-retrieval 80 --num-dev-examples 20 --temperature 0.0

    # Wixarika
    python translate_llm_manyshot_gemini.py \
        --language wixarika \
        --train-src  data/parallel/wixarika_with_dev/train.es \
        --train-tgt  data/parallel/wixarika_with_dev/train.hch \
        --input-spanish outputs/wixarika/dev_caption_es_v1.txt \
        --output-preds  outputs/wixarika/gemini_flash_r80_d20_preds.txt \
        --samples-tsv   outputs/wixarika/gemini_flash_r80_d20_samples.tsv \
        --num-retrieval 80 --num-dev-examples 20 --temperature 0.0

    # Maya
    python translate_llm_manyshot_gemini.py \
        --language maya \
        --train-src  data/parallel/maya_with_dev/train.es \
        --train-tgt  data/parallel/maya_with_dev/train.maya \
        --input-spanish outputs/maya/dev_caption_es_v1.txt \
        --output-preds  outputs/maya/gemini_flash_r80_d20_preds.txt \
        --samples-tsv   outputs/maya/gemini_flash_r80_d20_samples.tsv \
        --num-retrieval 80 --num-dev-examples 20 --temperature 0.0
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────
# Language registry
# ──────────────────────────────────────────────────────────────

# Each entry:
#   name        : display name used in prompts and summary output
#   system_prompt: full Gemini system instruction
#   tgt_label   : the "TGT:" label used in the few-shot prompt template
#   clean_prefix: regex alternation of leading labels to strip from predictions

LANGUAGE_CONFIGS: Dict[str, Dict] = {
    "guarani": {
        "name": "Guaraní/Jopara",
        "tgt_label": "GN",
        "clean_prefix": r"GN:|Guaran[ií]:",
        "system_prompt": (
            "You are a careful translator from Spanish into Guaraní/Jopara.\n\n"
            "Rules:\n"
            "- Translate into natural Guaraní/Jopara.\n"
            "- Match the style of the provided examples.\n"
            "- Stay concise.\n"
            "- Preserve culturally specific nouns when appropriate.\n"
            "- Output exactly one Guaraní line only.\n"
            "- Do not explain anything.\n"
        ),
    },
    "nahuatl": {
        "name": "Nahuatl",
        "tgt_label": "NAH",
        "clean_prefix": r"NAH:|Nahuatl:",
        "system_prompt": (
            "You are a careful translator from Spanish into Nahuatl.\n\n"
            "Rules:\n"
            "- Translate into natural Nahuatl.\n"
            "- Match the style of the provided examples.\n"
            "- Stay concise.\n"
            "- Preserve culturally specific nouns when appropriate.\n"
            "- Output exactly one Nahuatl line only.\n"
            "- Do not explain anything.\n"
        ),
    },
    "bribri": {
        "name": "Bribri",
        "tgt_label": "BZD",
        "clean_prefix": r"BZD:|Bribri:",
        "system_prompt": (
            "You are a careful translator from Spanish into Bribri.\n\n"
            "Rules:\n"
            "- Translate into natural Bribri.\n"
            "- Match the style of the provided examples.\n"
            "- Stay concise.\n"
            "- Preserve culturally specific nouns when appropriate.\n"
            "- Output exactly one Bribri line only.\n"
            "- Do not explain anything.\n"
        ),
    },
    "wixarika": {
        "name": "Wixarika",
        "tgt_label": "HCH",
        "clean_prefix": r"HCH:|Wixarika:",
        "system_prompt": (
            "You are a careful translator from Spanish into Wixarika (Huichol).\n\n"
            "Rules:\n"
            "- Translate into natural Wixarika.\n"
            "- Output exactly one short caption line only.\n"
            "- Stay close to the Spanish input.\n"
            "- Prefer simple, concrete wording.\n"
            "- Do not repeat words or phrases.\n"
            "- Do not invent details that are not in the Spanish input.\n"
            "- Match the style of the provided examples.\n"
            "- Do not explain anything.\n"
        ),
    },
    "maya": {
        "name": "Maya",
        "tgt_label": "MAYA",
        "clean_prefix": r"MAYA:|Maya:",
        "system_prompt": (
            "You are a careful translator from Spanish into Yucatec Maya.\n\n"
            "Rules:\n"
            "- Translate into natural Yucatec Maya.\n"
            "- Match the style of the provided examples.\n"
            "- Stay concise.\n"
            "- Preserve culturally specific nouns when appropriate.\n"
            "- Output exactly one Maya line only.\n"
            "- Do not explain anything.\n"
        ),
    },
}

SUPPORTED_LANGUAGES = sorted(LANGUAGE_CONFIGS.keys())


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Many-shot Spanish → Indigenous Language translation via Gemini."
    )
    # ── Language ──
    ap.add_argument("--language", required=True, choices=SUPPORTED_LANGUAGES,
                    help=f"Target language. One of: {', '.join(SUPPORTED_LANGUAGES)}")
    # ── Data ──
    ap.add_argument("--train-src", required=True,
                    help="Training Spanish sentences (one per line).")
    ap.add_argument("--train-tgt", required=True,
                    help="Training target-language sentences (one per line).")
    ap.add_argument("--input-spanish", required=True,
                    help="Spanish captions to translate.")
    ap.add_argument("--reference-file", default=None,
                    help="Optional reference file for scoring.")
    # ── Outputs ──
    ap.add_argument("--output-preds", required=True,
                    help="Output predictions (one per line).")
    ap.add_argument("--metrics-json", default=None)
    ap.add_argument("--per-example-tsv", default=None)
    ap.add_argument("--samples-tsv", required=True)
    ap.add_argument("--raw-jsonl", default=None)
    # ── Model ──
    ap.add_argument("--model", default="gemini-2.5-flash",
                    help="Gemini model name.")
    ap.add_argument("--api-key", default="",
                    help="Gemini API key; falls back to GEMINI_API_KEY or GOOGLE_API_KEY env var.")
    # ── Retrieval ──
    ap.add_argument("--num-retrieval", type=int, default=24,
                    help="Number of retrieved train examples per query.")
    ap.add_argument("--retrieval-pool", type=int, default=80,
                    help="BM25 candidate pool size before reranking.")
    ap.add_argument("--use-rerank", action="store_true",
                    help="Rerank BM25 candidates to prefer short caption-like examples.")
    ap.add_argument("--dev-example-spanish", default=None,
                    help="Optional dev Spanish exemplar pool (for leave-one-out dev shots).")
    ap.add_argument("--dev-example-refs", default=None,
                    help="Optional dev target-language exemplar pool.")
    ap.add_argument("--num-dev-examples", type=int, default=0,
                    help="Number of leave-one-out dev exemplars to prepend.")
    # ── Generation ──
    ap.add_argument("--max-tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--thinking-budget", type=int, default=0,
                    help="Thinking token budget. 0 = disable thinking (fastest).")
    # ── Misc ──
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only first N examples (for debugging).")
    ap.add_argument("--sleep", type=float, default=0.15,
                    help="Seconds between API calls.")
    ap.add_argument("--max-retries", type=int, default=6)
    ap.add_argument("--score-script", default="score_outputs.py")
    ap.add_argument("--sample-count", type=int, default=20)
    ap.add_argument("--override-system-prompt", default=None,
                    help="If set, replaces the language config system_prompt entirely.")
    return ap.parse_args()


# ──────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────

def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def write_lines(lines: List[str], path: str) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write((line or "").replace("\n", " ").strip() + "\n")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def clean_prediction(text: str, clean_prefix: str) -> str:
    """Strip leading label prefixes and surrounding quotes from a prediction."""
    text = normalize_text(text)
    # Strip the language-specific label prefix (e.g. "NAH:") and the legacy GN: prefix
    combined = rf"^({clean_prefix}|GN:|Guaran[ií]:)"
    text = re.sub(combined, "", text, flags=re.I).strip()
    text = text.strip('"').strip("'")
    text = re.sub(r"\s+", " ", text)
    return text


def count_tokens(text: str) -> int:
    return len(normalize_text(text).split())


# ──────────────────────────────────────────────────────────────
# Prompt construction
# ──────────────────────────────────────────────────────────────

def build_user_prompt(
    query_es: str,
    retrieved_pairs: List[Tuple[str, str]],
    tgt_label: str,
    dev_pairs: Optional[List[Tuple[str, str]]] = None,
) -> str:
    sections = []

    if dev_pairs:
        sections.append("Gold-style example pairs:\n")
        for es, tgt in dev_pairs:
            sections.append(f"ES: {normalize_text(es)}\n{tgt_label}: {normalize_text(tgt)}\n")

    sections.append("Retrieved parallel examples:\n")
    for es, tgt in retrieved_pairs:
        sections.append(f"ES: {normalize_text(es)}\n{tgt_label}: {normalize_text(tgt)}\n")

    sections.append(f"Now translate this.\nES: {normalize_text(query_es)}\n{tgt_label}:")
    return "\n".join(sections)


# ──────────────────────────────────────────────────────────────
# Reranking
# ──────────────────────────────────────────────────────────────

# Guaraní-specific place names that should be down-weighted in retrieved examples.
# These are irrelevant for other languages but harmless to check against — none of
# these strings appear in Nahuatl/Bribri/Wixarika/Maya training corpora.
PLACE_HINTS = [
    "mercado 4", "mercado cuatro", "corrientes", "luque", "paraguay",
    "san juan", "jesuíta", "jesuita", "residentas",
]

# Guaraní boilerplate phrases in retrieved examples that reduce caption quality.
BOILERPLATE_HINTS = [
    "ko'ápe", "ojehecha", "ohechauka", "ikatu hína", "ta'angápe",
    "en la imagen", "se observa", "la imagen muestra", "posiblemente",
]


def rerank_candidate_score(es: str, tgt: str, bm25_rank: int) -> float:
    es_n = normalize_text(es).lower()
    tgt_n = normalize_text(tgt).lower()
    es_len = count_tokens(es_n)
    tgt_len = count_tokens(tgt_n)
    score = 0.0

    score += max(0, 40 - bm25_rank) * 0.15

    if 4 <= es_len <= 20:
        score += 2.0
    elif es_len <= 28:
        score += 1.0
    else:
        score -= 0.12 * (es_len - 28)

    if 4 <= tgt_len <= 20:
        score += 2.0
    elif tgt_len <= 28:
        score += 1.0
    else:
        score -= 0.12 * (tgt_len - 28)

    for w in PLACE_HINTS:
        if w in es_n:
            score -= 1.0
        if w in tgt_n:
            score -= 1.0

    for w in BOILERPLATE_HINTS:
        if w in es_n:
            score -= 0.8
        if w in tgt_n:
            score -= 0.8

    score -= es_n.count(",") * 0.35
    score -= tgt_n.count(",") * 0.35
    score -= es_n.count(";") * 0.5
    score -= tgt_n.count(";") * 0.5
    score += caption_domain_score(es_n)

    return score


def select_retrieved_pairs(
    es_query: str,
    train_src: List[str],
    train_tgt: List[str],
    bm25,
    num_retrieval: int,
    retrieval_pool: int,
    use_rerank: bool,
) -> Tuple[List[int], List[Tuple[str, str]]]:
    query_tokens = normalize_text(es_query).lower().split()
    scores = bm25.get_scores(query_tokens)
    ranked = sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)

    pool_indices = []
    for j in ranked:
        if normalize_text(train_src[j]).lower() == normalize_text(es_query).lower():
            continue
        pool_indices.append(j)
        if len(pool_indices) >= retrieval_pool:
            break

    if not use_rerank:
        chosen = pool_indices[:num_retrieval]
        return chosen, [(train_src[j], train_tgt[j]) for j in chosen]

    rescored = [
        (rerank_candidate_score(train_src[j], train_tgt[j], rank_pos), j)
        for rank_pos, j in enumerate(pool_indices)
    ]
    rescored.sort(reverse=True)
    chosen = [j for _, j in rescored[:num_retrieval]]
    return chosen, [(train_src[j], train_tgt[j]) for j in chosen]

# Visual/concrete vocabulary that signals a caption-style sentence
VISUAL_HINTS = [
    "mujer", "hombre", "niño", "niña", "persona", "personas",
    "ropa", "camisa", "vestido", "sombrero", "lleva", "usa",
    "mesa", "silla", "casa", "calle", "campo", "agua", "fuego",
    "rojo", "azul", "verde", "blanco", "negro", "amarillo",
    "grande", "pequeño", "sostiende", "carga", "cocina", "come",
    "vende", "trabaja", "camina", "está", "hay", "tiene",
]

def caption_domain_score(es: str) -> float:
    """Bonus score for training examples that look like visual captions."""
    es_lower = es.lower()
    return sum(0.5 for w in VISUAL_HINTS if w in es_lower)

# ──────────────────────────────────────────────────────────────
# Gemini API call
# ──────────────────────────────────────────────────────────────

def call_gemini(
    client,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    thinking_budget: int,
    max_retries: int,
) -> Tuple[str, Dict]:
    """Call Gemini via the google-genai SDK. Returns (text, usage_dict)."""
    from google.genai import types

    gen_config_kwargs: Dict = {
        "temperature": temperature,
        "max_output_tokens": max_tokens,
        "thinking_config": types.ThinkingConfig(thinking_budget=thinking_budget),
    }

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        **gen_config_kwargs,
    )

    last_err = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=user_prompt,
                config=config,
            )

            text = ""
            try:
                if response.text:
                    text = response.text
            except Exception:
                pass
            if not text and response.candidates:
                for part in (response.candidates[0].content.parts or []):
                    # skip thought/reasoning parts, only take final answer
                    if hasattr(part, "thought") and part.thought:
                        continue
                    if hasattr(part, "text") and part.text:
                        text += part.text

            usage: Dict = {}
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                um = response.usage_metadata
                usage = {
                    "prompt_tokens":     getattr(um, "prompt_token_count", None),
                    "completion_tokens": getattr(um, "candidates_token_count", None),
                    "total_tokens":      getattr(um, "total_token_count", None),
                    "thinking_tokens":   getattr(um, "thoughts_token_count", None),
                }

            return normalize_text(text), usage

        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            if "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str:
                sleep_s = min(60, (2 ** (attempt + 1)) + random.random() * 2)
                print(f"  [rate-limited, sleeping {sleep_s:.1f}s]", flush=True)
            else:
                sleep_s = min(30, (2 ** attempt) + random.random())
                print(f"  [error: {e}, retry {attempt+1}/{max_retries}, sleeping {sleep_s:.1f}s]",
                      flush=True)
            time.sleep(sleep_s)

    raise RuntimeError(f"Gemini call failed after {max_retries} retries: {last_err}")


# ──────────────────────────────────────────────────────────────
# Scoring helper
# ──────────────────────────────────────────────────────────────

def maybe_score(args: argparse.Namespace) -> None:
    if not (args.reference_file and args.metrics_json and args.per_example_tsv):
        return
    ensure_parent(args.metrics_json)
    ensure_parent(args.per_example_tsv)
    cmd = [
        sys.executable, args.score_script,
        "--pred-file",      args.output_preds,
        "--ref-file",       args.reference_file,
        "--metrics-json",   args.metrics_json,
        "--per-example-tsv", args.per_example_tsv,
    ]
    subprocess.run(cmd, check=True)


# ──────────────────────────────────────────────────────────────
# Output helpers
# ──────────────────────────────────────────────────────────────

def write_samples_tsv(
    path: str,
    rows: List[Dict],
    references: Optional[List[str]],
    sample_count: int,
) -> None:
    ensure_parent(path)
    fieldnames = [
        "idx", "spanish", "prediction", "reference",
        "retrieved_indices", "dev_example_indices",
        "prompt_tokens", "completion_tokens", "total_tokens", "thinking_tokens",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for r in rows[:sample_count]:
            writer.writerow({
                "idx":                 r["idx"],
                "spanish":             r["spanish"],
                "prediction":          r["prediction"],
                "reference":           "" if references is None else references[r["idx"]],
                "retrieved_indices":   ",".join(map(str, r["retrieved_indices"])),
                "dev_example_indices": ",".join(map(str, r["dev_example_indices"])),
                "prompt_tokens":       r.get("usage", {}).get("prompt_tokens"),
                "completion_tokens":   r.get("usage", {}).get("completion_tokens"),
                "total_tokens":        r.get("usage", {}).get("total_tokens"),
                "thinking_tokens":     r.get("usage", {}).get("thinking_tokens"),
            })


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    cfg = LANGUAGE_CONFIGS[args.language]
    lang_name    = cfg["name"]
    tgt_label    = cfg["tgt_label"]
    clean_prefix = cfg["clean_prefix"]
    system_prompt = args.override_system_prompt if args.override_system_prompt else cfg["system_prompt"]

    print("=" * 60)
    print(f"Gemini Many-Shot Translation  —  Spanish → {lang_name}")
    print("=" * 60)
    print(f"Language:        {args.language}  ({lang_name})")
    print(f"Model:           {args.model}")
    print(f"Retrieval:       {args.num_retrieval} (pool={args.retrieval_pool}, rerank={args.use_rerank})")
    print(f"Dev examples:    {args.num_dev_examples}")
    print(f"Temperature:     {args.temperature}")
    print(f"Thinking budget: {args.thinking_budget}")
    print(flush=True)

    # ── Imports ──
    try:
        from google import genai
    except ImportError:
        print("ERROR: Missing google-genai. Install with:")
        print("  pip install google-genai --break-system-packages")
        sys.exit(1)

    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("ERROR: Missing rank-bm25. Install with:")
        print("  pip install rank-bm25 --break-system-packages")
        sys.exit(1)

    # ── API key ──
    api_key = (
        args.api_key
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY env var or pass --api-key")
        sys.exit(1)

    # ── Load data ──
    train_src = read_lines(args.train_src)
    train_tgt = read_lines(args.train_tgt)
    if len(train_src) != len(train_tgt):
        raise ValueError(f"Train mismatch: {len(train_src)} src vs {len(train_tgt)} tgt")

    inputs = read_lines(args.input_spanish)
    refs   = read_lines(args.reference_file) if args.reference_file else None
    if refs is not None and len(refs) != len(inputs):
        raise ValueError(f"Reference mismatch: {len(refs)} refs vs {len(inputs)} inputs")

    dev_es  = read_lines(args.dev_example_spanish) if args.dev_example_spanish else None
    dev_tgt = read_lines(args.dev_example_refs)    if args.dev_example_refs    else None
    if (dev_es is None) ^ (dev_tgt is None):
        raise ValueError("Provide both --dev-example-spanish and --dev-example-refs together")
    if dev_es is not None and len(dev_es) != len(dev_tgt):
        raise ValueError(f"Dev exemplar mismatch: {len(dev_es)} vs {len(dev_tgt)}")

    if args.limit is not None:
        inputs = inputs[:args.limit]
        if refs is not None:
            refs = refs[:args.limit]

    # ── Build BM25 index ──
    print(f"Building BM25 index over {len(train_src)} training pairs...", flush=True)
    tokenized_corpus = [normalize_text(x).lower().split() for x in train_src]
    bm25 = BM25Okapi(tokenized_corpus)

    # ── Create Gemini client ──
    client = genai.Client(api_key=api_key)
    print(f"Gemini client ready (model={args.model})", flush=True)

    # ── Translate ──
    predictions: List[str] = []
    raw_rows: List[Dict] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    print(f"\nTranslating {len(inputs)} examples...\n", flush=True)

    for i, es in enumerate(inputs):
        retrieved_indices, retrieved_pairs = select_retrieved_pairs(
            es_query=es,
            train_src=train_src,
            train_tgt=train_tgt,
            bm25=bm25,
            num_retrieval=args.num_retrieval,
            retrieval_pool=args.retrieval_pool,
            use_rerank=args.use_rerank,
        )

        # Leave-one-out dev exemplars
        dev_pairs: List[Tuple[str, str]] = []
        dev_example_indices: List[int] = []
        if dev_es is not None and args.num_dev_examples > 0:
            dev_ranked = sorted(
                range(len(dev_es)),
                key=lambda j: len(
                    set(normalize_text(es).lower().split()) &
                    set(normalize_text(dev_es[j]).lower().split())
                ),
                reverse=True,
            )
            for j in dev_ranked:
                if normalize_text(dev_es[j]).lower() == normalize_text(es).lower():
                    continue
                dev_example_indices.append(j)
                dev_pairs.append((dev_es[j], dev_tgt[j]))
                if len(dev_pairs) >= args.num_dev_examples:
                    break

        user_prompt = build_user_prompt(
            query_es=es,
            retrieved_pairs=retrieved_pairs,
            tgt_label=tgt_label,
            dev_pairs=dev_pairs or None,
        )

        pred, usage = call_gemini(
            client=client,
            model=args.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            thinking_budget=args.thinking_budget,
            max_retries=args.max_retries,
        )
        pred = clean_prediction(pred, clean_prefix)
        predictions.append(pred)

        pt = usage.get("prompt_tokens") or 0
        ct = usage.get("completion_tokens") or 0
        total_prompt_tokens     += pt
        total_completion_tokens += ct

        raw_rows.append({
            "idx":                 i,
            "spanish":             normalize_text(es),
            "prediction":          pred,
            "reference":           None if refs is None else normalize_text(refs[i]),
            "retrieved_indices":   retrieved_indices,
            "dev_example_indices": dev_example_indices,
            "usage":               usage,
            "model":               args.model,
            "language":            args.language,
            "prompt":              user_prompt,
        })

        print(f"[{i+1:03d}/{len(inputs):03d}] tokens={pt}→{ct}  {pred[:100]}", flush=True)
        time.sleep(args.sleep)

    # ── Write outputs ──
    write_lines(predictions, args.output_preds)

    if args.raw_jsonl:
        ensure_parent(args.raw_jsonl)
        with open(args.raw_jsonl, "w", encoding="utf-8") as f:
            for row in raw_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_samples_tsv(
        path=args.samples_tsv,
        rows=raw_rows,
        references=refs,
        sample_count=args.sample_count,
    )

    maybe_score(args)

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Language:             {args.language}  ({lang_name})")
    print(f"Model:                {args.model}")
    print(f"Examples:             {len(inputs)}")
    print(f"Retrieval:            {args.num_retrieval} (rerank={args.use_rerank})")
    print(f"Dev examples:         {args.num_dev_examples}")
    print(f"Total prompt tokens:  {total_prompt_tokens:,}")
    print(f"Total output tokens:  {total_completion_tokens:,}")

    # Gemini 2.5 Flash pricing (2025): $0.15/M input, $0.60/M output
    est_cost = (total_prompt_tokens * 0.15 / 1e6) + (total_completion_tokens * 0.60 / 1e6)
    print(f"Est. cost (Flash):    ${est_cost:.4f}")

    print(f"\nPredictions  → {args.output_preds}")
    if args.metrics_json:
        try:
            with open(args.metrics_json) as f:
                metrics = json.load(f)
            print(f"Metrics      → {args.metrics_json}")
            print(f"  chrF++  = {metrics.get('chrfpp', '?')}")
            print(f"  chrF    = {metrics.get('chrf', '?')}")
            print(f"  BLEU    = {metrics.get('bleu', '?')}")
        except Exception:
            print(f"Metrics      → {args.metrics_json} (check file)")
    if args.per_example_tsv:
        print(f"Per-example  → {args.per_example_tsv}")
    print(f"Samples      → {args.samples_tsv}")
    if args.raw_jsonl:
        print(f"Raw JSONL    → {args.raw_jsonl}")


if __name__ == "__main__":
    main()