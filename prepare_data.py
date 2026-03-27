#!/usr/bin/env python3
"""Prepare combined parallel training data for MT models.

Combines:
  1. AmericasNLP parallel data (from 2021/2023 shared tasks)
  2. MultiScript30K synthetic Es↔Gn pairs (from multi30k-extension repo)
  3. AmericasNLP 2026 dev set (allowed for training per updated rules)

The MultiScript30K data format:
  multi30k-extension/
    multi30k-dataset-en-es/   → has train.es, val.es, test_2016_flickr.es, etc.
    multi30k-dataset-en-gn/   → has train.gn, val.gn, test_2016_flickr.gn, etc.

The AmericasNLP 2023 data format:
  americasnlp2023/data/
    gn/train.es, train.gn     (Guaraní)
    hch/train.es, train.hch   (Wixárika)

This script produces:
  data/parallel/{lang}/
    train.es, train.{lang}    (combined training data)
    val.es, val.{lang}        (validation data, kept separate for eval)

Usage:
    # Guaraní: combine AmericasNLP 2023 + MultiScript30K + 2026 dev
    python prepare_data.py \
        --language guarani \
        --americasnlp-dir data/americasnlp2023/data/gn \
        --multiscript-es-dir data/multi30k-extension/multi30k-dataset-en-es \
        --multiscript-tgt-dir data/multi30k-extension/multi30k-dataset-en-gn \
        --dev2026-jsonl data/americasnlp2026/data/dev/guarani/guarani.jsonl \
        --output-dir data/parallel/guarani \
        --val-ratio 0.05

    # Wixárika: only AmericasNLP + 2026 dev (no MultiScript30K for Wixárika)
    python prepare_data.py \
        --language wixarika \
        --americasnlp-dir data/americasnlp2023/data/hch \
        --dev2026-jsonl data/americasnlp2026/data/dev/wixarika/wixarika.jsonl \
        --output-dir data/parallel/wixarika
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

LOGGER = logging.getLogger("prepare_data")

# MultiScript30K file naming convention
MULTI30K_SPLITS = ["train", "val", "test_2016_flickr", "test_2017_flickr",
                   "test_2017_mscoco", "test_2018_flickr"]

# AmericasNLP 2023 language codes → file extensions
LANG_TO_EXT = {
    "guarani": "gn",
    "wixarika": "hch",
    "bribri": "bzd",
    "maya": "yua",       # Yucatec Maya
    "quechua": "quy",
    "aymara": "aym",
}

# AmericasNLP 2026 JSONL field containing target caption
TARGET_FIELD = "target_caption"
# In the 2026 task, pilot has spanish_caption; dev does NOT
SPANISH_FIELD = "spanish_caption"


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def read_lines(path: str) -> List[str]:
    """Read non-empty lines from a text file."""
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]
    return lines


def write_lines(lines: List[str], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def load_parallel_from_files(
    src_path: str, tgt_path: str, label: str
) -> List[Tuple[str, str]]:
    """Load parallel pairs from two aligned text files."""
    if not os.path.exists(src_path) or not os.path.exists(tgt_path):
        LOGGER.warning("Missing file(s): %s or %s — skipping", src_path, tgt_path)
        return []
    src = read_lines(src_path)
    tgt = read_lines(tgt_path)
    if len(src) != len(tgt):
        LOGGER.warning(
            "%s: line count mismatch (%d vs %d) — skipping",
            label, len(src), len(tgt),
        )
        return []
    pairs = [(s.strip(), t.strip()) for s, t in zip(src, tgt)
             if s.strip() and t.strip()]
    LOGGER.info("Loaded %d pairs from %s", len(pairs), label)
    return pairs


def load_americasnlp_data(
    data_dir: str, language: str
) -> List[Tuple[str, str]]:
    """Load parallel data from AmericasNLP 2021/2023 format.

    Expected files: {data_dir}/train.es, {data_dir}/train.{ext}
    Also tries: dev.es/dev.{ext}
    """
    ext = LANG_TO_EXT.get(language, language)
    pairs = []

    for split in ["train", "dev"]:
        src_path = os.path.join(data_dir, f"{split}.es")
        tgt_path = os.path.join(data_dir, f"{split}.{ext}")
        pairs.extend(load_parallel_from_files(
            src_path, tgt_path, f"AmericasNLP-{split}"
        ))

    return pairs


def load_multiscript30k_data(
    es_dir: str, tgt_dir: str, tgt_ext: str
) -> List[Tuple[str, str]]:
    """Load parallel Es↔Target from MultiScript30K directories.

    The MultiScript30K repo stores data as:
      multi30k-dataset-en-es/train.es
      multi30k-dataset-en-gn/train.gn
    with splits: train, val, test_2016_flickr, etc.

    We use ALL splits as synthetic training data (since it's all
    machine-generated from NLLB-200 anyway — no held-out concern).
    """
    pairs = []

    for split in MULTI30K_SPLITS:
        src_path = os.path.join(es_dir, f"{split}.es")
        tgt_path = os.path.join(tgt_dir, f"{split}.{tgt_ext}")
        pairs.extend(load_parallel_from_files(
            src_path, tgt_path, f"MultiScript30K-{split}"
        ))

    return pairs


def load_dev2026_as_parallel(
    jsonl_path: str,
    spanish_field: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """Load AmericasNLP 2026 dev JSONL as parallel pairs.

    The dev set has target_caption (indigenous language) but NO spanish_caption.
    If Spanish captions were generated by our VLM pipeline, we can pass those
    via --generated-spanish-file. Otherwise, we skip this source.

    For the pilot set, spanish_caption IS available.
    """
    if not jsonl_path or not os.path.exists(jsonl_path):
        return []

    pairs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            target = row.get(TARGET_FIELD, "").strip()
            spanish = row.get(spanish_field or SPANISH_FIELD, "").strip()

            if spanish and target:
                pairs.append((spanish, target))

    LOGGER.info("Loaded %d pairs from 2026 JSONL: %s", len(pairs), jsonl_path)
    return pairs


def load_generated_spanish_pairs(
    jsonl_path: str,
    spanish_txt_path: str,
) -> List[Tuple[str, str]]:
    """Pair generated Spanish captions with target captions from 2026 dev JSONL.

    Use this when dev set has no Spanish captions but you've already
    run caption_images.py to generate them.
    """
    if not os.path.exists(jsonl_path) or not os.path.exists(spanish_txt_path):
        return []

    targets = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            targets.append(row.get(TARGET_FIELD, "").strip())

    spanish = read_lines(spanish_txt_path)

    if len(spanish) != len(targets):
        LOGGER.warning(
            "Generated Spanish (%d) != JSONL targets (%d) — skipping",
            len(spanish), len(targets),
        )
        return []

    pairs = [(s.strip(), t) for s, t in zip(spanish, targets)
             if s.strip() and t.strip()]
    LOGGER.info("Loaded %d pairs from generated Spanish + 2026 targets", len(pairs))
    return pairs


def deduplicate_pairs(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Remove exact duplicate pairs."""
    seen = set()
    unique = []
    for src, tgt in pairs:
        key = (src, tgt)
        if key not in seen:
            seen.add(key)
            unique.append((src, tgt))
    removed = len(pairs) - len(unique)
    if removed > 0:
        LOGGER.info("Removed %d duplicate pairs", removed)
    return unique


def split_train_val(
    pairs: List[Tuple[str, str]],
    val_ratio: float = 0.05,
    seed: int = 42,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Split pairs into train and validation sets."""
    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    return shuffled[n_val:], shuffled[:n_val]


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--language", required=True,
                        choices=sorted(LANG_TO_EXT.keys()))
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for combined parallel files.")

    # AmericasNLP 2021/2023 data
    parser.add_argument("--americasnlp-dir", default=None,
                        help="Dir with train.es/train.{ext} from AmericasNLP 2023.")

    # MultiScript30K synthetic data
    parser.add_argument("--multiscript-es-dir", default=None,
                        help="MultiScript30K Spanish dir (multi30k-dataset-en-es).")
    parser.add_argument("--multiscript-tgt-dir", default=None,
                        help="MultiScript30K target dir (multi30k-dataset-en-gn).")

    # AmericasNLP 2026 dev data
    parser.add_argument("--dev2026-jsonl", default=None,
                        help="2026 dev JSONL (has target_caption).")
    parser.add_argument("--pilot2026-jsonl", default=None,
                        help="2026 pilot JSONL (has both spanish_caption + target_caption).")
    parser.add_argument("--generated-spanish-file", default=None,
                        help="VLM-generated Spanish captions for dev images (one per line).")

    # Additional parallel files
    parser.add_argument("--extra-src", action="append", default=[],
                        help="Extra source (Spanish) file to include.")
    parser.add_argument("--extra-tgt", action="append", default=[],
                        help="Extra target file to include (must match --extra-src).")

    # Options
    parser.add_argument("--val-ratio", type=float, default=0.05,
                        help="Fraction of data to hold out as validation.")
    parser.add_argument("--existing-val-src", default=None,
                        help="Use this file as validation source instead of splitting.")
    parser.add_argument("--existing-val-tgt", default=None,
                        help="Use this file as validation target instead of splitting.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")

    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_level)

    ext = LANG_TO_EXT[args.language]
    all_pairs: List[Tuple[str, str]] = []

    # --- Source 1: AmericasNLP 2021/2023 parallel data ---
    if args.americasnlp_dir:
        all_pairs.extend(load_americasnlp_data(args.americasnlp_dir, args.language))

    # --- Source 2: MultiScript30K synthetic data ---
    if args.multiscript_es_dir and args.multiscript_tgt_dir:
        all_pairs.extend(load_multiscript30k_data(
            args.multiscript_es_dir, args.multiscript_tgt_dir, ext
        ))

    # --- Source 3: AmericasNLP 2026 pilot (has Spanish captions) ---
    if args.pilot2026_jsonl:
        all_pairs.extend(load_dev2026_as_parallel(
            args.pilot2026_jsonl, SPANISH_FIELD
        ))

    # --- Source 4: AmericasNLP 2026 dev + generated Spanish ---
    if args.dev2026_jsonl and args.generated_spanish_file:
        all_pairs.extend(load_generated_spanish_pairs(
            args.dev2026_jsonl, args.generated_spanish_file
        ))

    # --- Source 5: Extra parallel files ---
    if args.extra_src and args.extra_tgt:
        if len(args.extra_src) != len(args.extra_tgt):
            raise ValueError("--extra-src and --extra-tgt counts must match")
        for src_f, tgt_f in zip(args.extra_src, args.extra_tgt):
            all_pairs.extend(load_parallel_from_files(src_f, tgt_f, f"extra:{src_f}"))

    LOGGER.info("Total raw pairs: %d", len(all_pairs))
    all_pairs = deduplicate_pairs(all_pairs)
    LOGGER.info("After dedup: %d", len(all_pairs))

    if not all_pairs:
        LOGGER.error("No parallel data found! Check your paths.")
        return

    # --- Split into train/val ---
    if args.existing_val_src and args.existing_val_tgt:
        # Use provided validation files; all collected data goes to train
        train_pairs = all_pairs
        val_src = read_lines(args.existing_val_src)
        val_tgt = read_lines(args.existing_val_tgt)
        val_pairs = list(zip(val_src, val_tgt))
        LOGGER.info("Using existing val set: %d pairs", len(val_pairs))
    else:
        train_pairs, val_pairs = split_train_val(
            all_pairs, args.val_ratio, args.seed
        )

    LOGGER.info("Train: %d pairs, Val: %d pairs", len(train_pairs), len(val_pairs))

    # --- Write output ---
    os.makedirs(args.output_dir, exist_ok=True)

    write_lines([p[0] for p in train_pairs],
                os.path.join(args.output_dir, "train.es"))
    write_lines([p[1] for p in train_pairs],
                os.path.join(args.output_dir, f"train.{ext}"))
    write_lines([p[0] for p in val_pairs],
                os.path.join(args.output_dir, "val.es"))
    write_lines([p[1] for p in val_pairs],
                os.path.join(args.output_dir, f"val.{ext}"))

    # --- Stats summary ---
    stats = {
        "language": args.language,
        "extension": ext,
        "total_train": len(train_pairs),
        "total_val": len(val_pairs),
        "sources": {
            "americasnlp": bool(args.americasnlp_dir),
            "multiscript30k": bool(args.multiscript_es_dir),
            "pilot2026": bool(args.pilot2026_jsonl),
            "dev2026_generated": bool(args.dev2026_jsonl and args.generated_spanish_file),
            "extra_files": len(args.extra_src),
        },
    }

    stats_path = os.path.join(args.output_dir, "data_sources.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    LOGGER.info("Wrote parallel data to %s", args.output_dir)
    LOGGER.info("Stats: %s", stats)

    print(f"\n{'='*50}")
    print(f"DATA PREPARED: {args.language}")
    print(f"{'='*50}")
    print(f"  Train: {len(train_pairs)} pairs")
    print(f"  Val:   {len(val_pairs)} pairs")
    print(f"  Output: {args.output_dir}/")
    print(f"  Files:  train.es, train.{ext}, val.es, val.{ext}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
