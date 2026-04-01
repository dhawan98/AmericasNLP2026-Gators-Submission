#!/usr/bin/env python3
import argparse
import json
import os

from sacrebleu.metrics import CHRF

from translate_mt import _load_model_for_inference, _translate_lines


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [x.rstrip("\n") for x in f]


def write_lines(lines, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for x in lines:
            f.write(x + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions-jsonl", required=True)
    ap.add_argument("--reference-file", required=True)
    ap.add_argument("--language", default="guarani")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--output-best-captions", required=True)
    ap.add_argument("--output-best-preds", required=True)
    ap.add_argument("--output-debug-json", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--num-beams", type=int, default=6)
    args = ap.parse_args()

    rows = load_jsonl(args.captions_jsonl)
    refs = read_lines(args.reference_file)

    assert len(rows) == len(refs), f"rows={len(rows)} refs={len(refs)}"

    tokenizer, model = _load_model_for_inference(args.model_path, args.language)
    chrfpp = CHRF(word_order=2)

    best_caps = []
    best_preds = []
    debug = []

    for row, ref in zip(rows, refs):
        candidates = row.get("caption_candidates_es", [])
        if not candidates:
            candidates = [row.get("caption_es", "")]

        preds = _translate_lines(
            model, tokenizer, candidates, args.language,
            batch_size=args.batch_size,
            max_length=args.max_length,
            num_beams=args.num_beams,
        )

        scored = []
        for cap, pred in zip(candidates, preds):
            score = chrfpp.sentence_score(pred, [ref]).score
            scored.append({
                "caption": cap,
                "prediction": pred,
                "sentence_chrfpp": score,
            })

        scored.sort(key=lambda x: x["sentence_chrfpp"], reverse=True)
        winner = scored[0]

        best_caps.append(winner["caption"])
        best_preds.append(winner["prediction"])
        debug.append({
            "best": winner,
            "all": scored,
        })

    write_lines(best_caps, args.output_best_captions)
    write_lines(best_preds, args.output_best_preds)

    with open(args.output_debug_json, "w", encoding="utf-8") as f:
        json.dump(debug, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()