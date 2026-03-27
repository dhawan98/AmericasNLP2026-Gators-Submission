#!/usr/bin/env python3
"""Step 3: Score system outputs with BLEU, chrF, and chrF++.

Usage:
    python score_outputs.py \
        --pred-file outputs/guarani/predictions.txt \
        --ref-file data/dev/guarani/references.txt \
        --metrics-json outputs/guarani/metrics.json \
        --per-example-tsv outputs/guarani/per_example.tsv
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from typing import Dict, List

LOGGER = logging.getLogger("score_outputs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-file", required=True)
    parser.add_argument("--ref-file", required=True)
    parser.add_argument("--pred-format", choices=["txt", "jsonl"], default="txt")
    parser.add_argument("--ref-format", choices=["txt", "jsonl"], default="txt")
    parser.add_argument("--pred-field", default="prediction")
    parser.add_argument("--ref-field", default="reference")
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--per-example-tsv", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def read_txt(path: str) -> List[Dict[str, str]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            rows.append({"id": str(idx), "text": line.rstrip("\n")})
    return rows


def read_jsonl(path: str, text_field: str, id_field: str) -> List[Dict[str, str]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            obj = json.loads(line)
            rows.append({
                "id": str(obj.get(id_field, idx)),
                "text": str(obj[text_field]),
            })
    return rows


def load_rows(path: str, fmt: str, text_field: str, id_field: str):
    if fmt == "txt":
        return read_txt(path)
    return read_jsonl(path, text_field=text_field, id_field=id_field)


def save_json(data: Dict, path: str) -> None:
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    pred_rows = load_rows(args.pred_file, args.pred_format,
                          args.pred_field, args.id_field)
    ref_rows = load_rows(args.ref_file, args.ref_format,
                         args.ref_field, args.id_field)

    if len(pred_rows) != len(ref_rows):
        raise ValueError(
            f"Mismatched: {len(pred_rows)} predictions vs {len(ref_rows)} references"
        )

    predictions = [row["text"] for row in pred_rows]
    references = [row["text"] for row in ref_rows]

    from sacrebleu.metrics import BLEU, CHRF

    bleu_metric = BLEU()
    chrf_metric = CHRF(word_order=0)
    chrfpp_metric = CHRF(word_order=2)

    corpus_metrics = {
        "num_examples": len(predictions),
        "bleu": bleu_metric.corpus_score(predictions, [references]).score,
        "chrf": chrf_metric.corpus_score(predictions, [references]).score,
        "chrfpp": chrfpp_metric.corpus_score(predictions, [references]).score,
    }
    save_json(corpus_metrics, args.metrics_json)
    LOGGER.info("Corpus metrics: %s", corpus_metrics)

    if args.per_example_tsv:
        import os
        os.makedirs(os.path.dirname(args.per_example_tsv) or ".", exist_ok=True)
        with open(args.per_example_tsv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["id", "prediction", "reference", "chrfpp_sentence"],
                delimiter="\t",
            )
            writer.writeheader()
            for pred_row, ref_row in zip(pred_rows, ref_rows):
                sent_score = chrfpp_metric.sentence_score(
                    pred_row["text"], [ref_row["text"]]
                ).score
                writer.writerow({
                    "id": pred_row["id"],
                    "prediction": pred_row["text"],
                    "reference": ref_row["text"],
                    "chrfpp_sentence": f"{sent_score:.4f}",
                })
        LOGGER.info("Per-example scores: %s", args.per_example_tsv)


if __name__ == "__main__":
    main()