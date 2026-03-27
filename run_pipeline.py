#!/usr/bin/env python3
"""End-to-end pipeline: Image → Spanish Caption → Indigenous Language Caption → Score.

Usage:
    # Full pipeline for Guaraní dev set
    python run_pipeline.py \
        --input-jsonl data/dev/guarani/guarani.jsonl \
        --base-path data/dev/guarani \
        --language guarani \
        --vlm-model Qwen/Qwen3-VL-8B-Instruct \
        --mt-model-path models/guarani_mbart50 \
        --output-dir outputs/guarani \
        --quantize 4bit

    # Skip captioning (reuse existing Spanish captions)
    python run_pipeline.py \
        --input-jsonl data/dev/guarani/guarani.jsonl \
        --base-path data/dev/guarani \
        --language guarani \
        --mt-model-path models/guarani_mbart50 \
        --output-dir outputs/guarani \
        --skip-captioning \
        --spanish-captions-file outputs/guarani/spanish_captions.txt
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

LOGGER = logging.getLogger("run_pipeline")


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def run_step(cmd: List[str], step_name: str):
    """Run a subprocess and handle errors."""
    LOGGER.info("=== %s ===", step_name)
    LOGGER.info("Command: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        LOGGER.error("%s failed with return code %d", step_name, result.returncode)
        sys.exit(result.returncode)
    LOGGER.info("%s completed successfully", step_name)


def extract_references_from_jsonl(
    jsonl_path: str,
    output_path: str,
    field: str = "target_caption",
):
    """Extract reference captions from JSONL for evaluation."""
    refs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            refs.append(row.get(field, ""))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ref in refs:
            f.write(ref.replace("\n", " ").strip() + "\n")

    LOGGER.info("Extracted %d references to %s", len(refs), output_path)
    return refs


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--input-jsonl", required=True,
                        help="Input JSONL file with image paths and references.")
    parser.add_argument("--base-path", default=None,
                        help="Base path for resolving image paths.")
    parser.add_argument("--language", required=True,
                        choices=["guarani", "wixarika", "bribri", "maya", "generic"])
    parser.add_argument("--output-dir", required=True)

    # Captioning options
    parser.add_argument("--vlm-model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--quantize", choices=["4bit", "8bit"], default=None)
    parser.add_argument("--skip-captioning", action="store_true",
                        help="Skip captioning step; use existing captions.")
    parser.add_argument("--spanish-captions-file", default=None,
                        help="Pre-existing Spanish captions (one per line).")

    # Translation options
    parser.add_argument("--mt-model-path", required=True,
                        help="Path to trained MT model.")
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)

    # Evaluation options
    parser.add_argument("--reference-field", default="target_caption",
                        help="JSONL field containing target language reference.")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip evaluation (no references available, e.g. test set).")

    parser.add_argument("--log-level", default="INFO")

    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_level)

    os.makedirs(args.output_dir, exist_ok=True)
    script_dir = Path(__file__).parent

    # File paths
    captions_jsonl = os.path.join(args.output_dir, "captions.jsonl")
    spanish_txt = os.path.join(args.output_dir, "spanish_captions.txt")
    predictions_txt = os.path.join(args.output_dir, "predictions.txt")
    references_txt = os.path.join(args.output_dir, "references.txt")
    metrics_json = os.path.join(args.output_dir, "metrics.json")
    per_example_tsv = os.path.join(args.output_dir, "per_example.tsv")

    # -----------------------------------------------------------------------
    # Step 1: Caption images → Spanish
    # -----------------------------------------------------------------------
    if args.skip_captioning:
        if args.spanish_captions_file:
            spanish_txt = args.spanish_captions_file
        LOGGER.info("Skipping captioning, using: %s", spanish_txt)
    else:
        caption_cmd = [
            sys.executable, str(script_dir / "caption_images.py"),
            "--input-jsonl", args.input_jsonl,
            "--language", args.language,
            "--model-name", args.vlm_model,
            "--output-jsonl", captions_jsonl,
            "--output-txt", spanish_txt,
        ]
        if args.base_path:
            caption_cmd.extend(["--base-path", args.base_path])
        if args.quantize:
            caption_cmd.extend(["--quantize", args.quantize])

        run_step(caption_cmd, "Step 1: Image Captioning")

    # -----------------------------------------------------------------------
    # Step 2: Translate Spanish → Target Language
    # -----------------------------------------------------------------------
    translate_cmd = [
        sys.executable, str(script_dir / "translate_mt.py"),
        "predict",
        "--language", args.language,
        "--model-path", args.mt_model_path,
        "--input-file", spanish_txt,
        "--output-file", predictions_txt,
        "--batch-size", str(args.batch_size),
        "--num-beams", str(args.num_beams),
    ]
    run_step(translate_cmd, "Step 2: Translation")

    # -----------------------------------------------------------------------
    # Step 3: Evaluate
    # -----------------------------------------------------------------------
    if not args.skip_eval:
        # Extract references from JSONL
        extract_references_from_jsonl(
            args.input_jsonl, references_txt, args.reference_field
        )

        score_cmd = [
            sys.executable, str(script_dir / "score_outputs.py"),
            "--pred-file", predictions_txt,
            "--ref-file", references_txt,
            "--metrics-json", metrics_json,
            "--per-example-tsv", per_example_tsv,
        ]
        run_step(score_cmd, "Step 3: Evaluation")

        # Print final results
        with open(metrics_json, "r") as f:
            metrics = json.load(f)
        print("\n" + "=" * 50)
        print(f"RESULTS for {args.language.upper()}")
        print("=" * 50)
        print(f"  ChrF++: {metrics['chrfpp']:.2f}")
        print(f"  ChrF:   {metrics['chrf']:.2f}")
        print(f"  BLEU:   {metrics['bleu']:.2f}")
        print(f"  N:      {metrics['num_examples']}")
        print("=" * 50)
    else:
        LOGGER.info("Skipping evaluation (no references).")
        print(f"\nPredictions saved to: {predictions_txt}")


if __name__ == "__main__":
    main()
