#!/usr/bin/env python3
"""Format predictions for AmericasNLP 2026 submission.

Takes the pipeline output and formats it into the expected submission format:
a JSONL file with id + generated caption fields.

Usage:
    python prepare_submission.py \
        --input-jsonl data/dev/guarani/guarani.jsonl \
        --predictions-file outputs/guarani/predictions.txt \
        --output-jsonl submissions/guarani_v1.jsonl \
        --team-name "LSU-MR-Lab" \
        --version 1
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from typing import List

LOGGER = logging.getLogger("prepare_submission")


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True,
                        help="Original task JSONL (for IDs).")
    parser.add_argument("--predictions-file", required=True,
                        help="Generated captions (one per line).")
    parser.add_argument("--output-jsonl", required=True,
                        help="Output submission JSONL.")
    parser.add_argument("--id-field", default="id",
                        help="ID field in input JSONL.")
    parser.add_argument("--caption-field", default="predicted_caption",
                        help="Field name for generated caption in output.")
    
    parser.add_argument("--team-name", default="LSU-MR-Lab")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_level)

    rows = load_jsonl(args.input_jsonl)
    predictions = read_lines(args.predictions_file)

    if len(rows) != len(predictions):
        raise ValueError(
            f"Mismatch: {len(rows)} input rows vs {len(predictions)} predictions"
        )

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)

    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for row, pred in zip(rows, predictions):
            entry = dict(row)
            entry[args.caption_field] = pred.strip()
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    LOGGER.info(
        "Wrote %d predictions to %s (team=%s, v%d)",
        len(predictions), args.output_jsonl, args.team_name, args.version,
    )
    print(f"Submission file: {args.output_jsonl}")
    print(f"  Examples: {len(predictions)}")
    print(f"  Team: {args.team_name}")
    print(f"  Version: {args.version}")


if __name__ == "__main__":
    main()
