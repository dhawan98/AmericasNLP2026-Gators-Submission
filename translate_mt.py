#!/usr/bin/env python3
"""Step 2: Train and run machine translation (Spanish → Indigenous Language).

Subcommands:
  train   — Fine-tune a seq2seq model from parallel source/target files
  predict — Translate a plain text file (one line per example)
  eval    — Translate + score on labeled data

Supports mBART-50 (your existing approach) and NLLB-200 (better low-resource coverage).

Usage:
    # Train mBART50 on Guaraní
    python translate_mt.py train \
        --language guarani \
        --train-src data/combined_train.es --train-tgt data/combined_train.gn \
        --val-src data/combined_val.es --val-tgt data/combined_val.gn \
        --output-dir models/guarani_mbart50 \
        --fp16 --num-train-epochs 15

    # Train NLLB-200 on Guaraní
    python translate_mt.py train \
        --language guarani \
        --model-name facebook/nllb-200-distilled-600M \
        --train-src data/combined_train.es --train-tgt data/combined_train.gn \
        --val-src data/combined_val.es --val-tgt data/combined_val.gn \
        --output-dir models/guarani_nllb200 \
        --fp16

    # Predict
    python translate_mt.py predict \
        --language guarani \
        --model-path models/guarani_mbart50 \
        --input-file outputs/guarani/spanish_captions.txt \
        --output-file outputs/guarani/target_captions.txt

    # Evaluate on dev set
    python translate_mt.py eval \
        --language guarani \
        --model-path models/guarani_mbart50 \
        --input-file outputs/guarani/spanish_captions.txt \
        --reference-file data/dev/guarani/references.txt \
        --output-file outputs/guarani/predictions.txt \
        --metrics-json outputs/guarani/metrics.json
"""
from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import re
import subprocess
import unicodedata
from typing import Dict, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger("translate_mt")

# ---------------------------------------------------------------------------
# Language configurations
# ---------------------------------------------------------------------------
# NOTE: For mBART-50, Guaraní is NOT in the original 50 languages, so tgt_lang
# is set to None (no forced BOS token). For NLLB-200, Guaraní IS supported
# as "grn_Latn". The code auto-detects the model type.
# ---------------------------------------------------------------------------

LANGUAGE_CONFIGS: Dict[str, Dict] = {
    "guarani": {
        "mbart": {"src_lang": "es_XX", "tgt_lang": None},
        "nllb":  {"src_lang": "spa_Latn", "tgt_lang": "grn_Latn"},
        "special_tokens": ["\u0303", "ã", "ẽ", "ĩ", "õ", "ũ", "ch", "mb", "ng",
                           "ỹ", "g̃"],  # Added missing Guaraní chars
        "digraphs": {"c h": "ch", "m b": "mb", "n g": "ng", "n d": "nd",
                     "n t": "nt"},
        "keep_chars": r"áéíóúñãẽĩõũỹg̃",
    },
    "wixarika": {
        "mbart": {"src_lang": "es_XX", "tgt_lang": None},
        "nllb":  {"src_lang": "spa_Latn", "tgt_lang": None},  # Not in NLLB
        "special_tokens": ["+", "ɨ"],  # Wixárika uses + for ɨ
        "digraphs": {},
        "keep_chars": r"áéíóúñ+ɨ",
    },
    "bribri": {
        "mbart": {"src_lang": "es_XX", "tgt_lang": None},
        "nllb":  {"src_lang": "spa_Latn", "tgt_lang": None},  # Not in NLLB
        "special_tokens": ["ë", "ö", "ë̀", "ö̀", "à", "è", "ì", "ò", "ù"],
        "digraphs": {},
        "keep_chars": r"áéíóúñëöàèìòù",
    },
    "maya": {
        "mbart": {"src_lang": "es_XX", "tgt_lang": None},
        "nllb":  {"src_lang": "spa_Latn", "tgt_lang": None},  # Yucatec Maya not in NLLB
        "special_tokens": ["ts'", "ch'", "k'", "p'", "t'"],
        "digraphs": {"t s": "ts", "c h": "ch"},
        "keep_chars": r"áéíóúñ'",
    },
    "generic": {
        "mbart": {"src_lang": None, "tgt_lang": None},
        "nllb":  {"src_lang": None, "tgt_lang": None},
        "special_tokens": [],
        "digraphs": {},
        "keep_chars": r"áéíóú",
    },
}


def detect_model_family(model_name: str) -> str:
    """Detect whether a model is mBART or NLLB family."""
    name_lower = model_name.lower()
    if "nllb" in name_lower:
        return "nllb"
    if "mbart" in name_lower:
        return "mbart"
    # Default: check for known patterns
    if "m2m" in name_lower:
        return "nllb"  # Similar API
    return "mbart"  # Default fallback


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

def preprocess_text(
    text: str,
    language: str = "generic",
    lowercase: bool = True,
    remove_special_chars: bool = True,
    normalize_unicode: bool = True,
) -> str:
    """Clean and normalize text for translation."""
    if normalize_unicode:
        text = unicodedata.normalize("NFKC", text)
    if lowercase:
        text = text.lower()
    if remove_special_chars:
        keep = LANGUAGE_CONFIGS.get(language, LANGUAGE_CONFIGS["generic"])["keep_chars"]
        pattern = rf"[^\w\s.,'\"{keep}]+"
        text = re.sub(pattern, "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def merge_digraphs(text: str, language: str) -> str:
    """Merge tokenized digraphs back together for a specific language."""
    digraphs = LANGUAGE_CONFIGS.get(language, {}).get("digraphs", {})
    for key, value in digraphs.items():
        text = text.replace(key, value)
    return text


def postprocess_translation(text: str, language: str) -> str:
    """Apply language-specific post-processing to translated text."""
    text = merge_digraphs(text, language)
    return text


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_corpus_metrics(
    predictions: List[str], references: List[str]
) -> Dict[str, float]:
    """Compute BLEU, chrF, and chrF++ scores."""
    from sacrebleu.metrics import BLEU, CHRF

    bleu_metric = BLEU()
    chrf_metric = CHRF(word_order=0)
    chrfpp_metric = CHRF(word_order=2)

    return {
        "bleu": bleu_metric.corpus_score(predictions, [references]).score,
        "chrf": chrf_metric.corpus_score(predictions, [references]).score,
        "chrfpp": chrfpp_metric.corpus_score(predictions, [references]).score,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_parallel_data(
    source_file: str,
    target_file: str,
    language: str,
    lowercase: bool = True,
    remove_special_chars: bool = True,
    normalize_unicode: bool = True,
    length_ratio_threshold: float = 2.5,
    plots_dir: Optional[str] = None,
) -> Tuple[List[Dict[str, str]], Dict[str, float]]:
    """Load and filter parallel data with length-ratio filtering."""
    with open(source_file, "r", encoding="utf-8") as sf:
        source_lines = sf.readlines()
    with open(target_file, "r", encoding="utf-8") as tf:
        target_lines = tf.readlines()

    if len(source_lines) != len(target_lines):
        raise ValueError(
            f"Line count mismatch: {source_file} has {len(source_lines)}, "
            f"{target_file} has {len(target_lines)}"
        )

    source_lengths, target_lengths, ratios = [], [], []
    data: List[Dict[str, str]] = []

    for src, tgt in zip(source_lines, target_lines):
        src_clean = preprocess_text(src, "generic", lowercase,
                                     remove_special_chars, normalize_unicode)
        tgt_clean = preprocess_text(tgt, language, lowercase,
                                     remove_special_chars, normalize_unicode)
        tgt_clean = postprocess_translation(tgt_clean, language)

        if not src_clean or not tgt_clean:
            continue

        src_len = len(src_clean.split())
        tgt_len = len(tgt_clean.split())
        if src_len == 0:
            continue

        ratio = tgt_len / src_len
        source_lengths.append(src_len)
        target_lengths.append(tgt_len)
        ratios.append(ratio)

        if 1 / length_ratio_threshold <= ratio <= length_ratio_threshold:
            data.append({"source_text": src_clean, "target_text": tgt_clean})

    if plots_dir:
        _plot_diagnostics(source_lengths, target_lengths, ratios,
                          length_ratio_threshold, plots_dir)

    stats = {
        "total_pairs": len(source_lines),
        "valid_pairs": len(data),
        "filtered_out": len(source_lines) - len(data),
        "avg_source_length": float(np.mean(source_lengths)) if source_lengths else 0,
        "avg_target_length": float(np.mean(target_lengths)) if target_lengths else 0,
    }
    LOGGER.info("Data stats: %s", stats)
    return data, stats


def _plot_diagnostics(src_lens, tgt_lens, ratios, threshold, out_dir):
    """Save length distribution plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.hist(src_lens, bins=30, alpha=0.5, label="Source (Spanish)")
    ax1.hist(tgt_lens, bins=30, alpha=0.5, label="Target")
    ax1.set_xlabel("Words")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Length Distribution")
    ax1.legend()
    ax1.grid(True)

    ax2.hist(ratios, bins=30, alpha=0.7)
    ax2.axvline(1 / threshold, color="red", ls="--", label="Threshold")
    ax2.axvline(threshold, color="red", ls="--")
    ax2.set_xlabel("Target/Source Length Ratio")
    ax2.set_title("Length Ratio Distribution")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "data_diagnostics.png"), dpi=100)
    plt.close()


# ---------------------------------------------------------------------------
# Model loading (training)
# ---------------------------------------------------------------------------

def load_tokenizer_and_model(model_name: str, language: str):
    """Load tokenizer and model, handling mBART vs NLLB differences."""
    from transformers import (
        AutoModelForSeq2SeqLM, AutoTokenizer,
        MBart50Tokenizer, MBartForConditionalGeneration,
    )

    family = detect_model_family(model_name)
    config = LANGUAGE_CONFIGS[language]
    lang_config = config.get(family, config.get("mbart", {}))

    if family == "mbart":
        tokenizer = MBart50Tokenizer.from_pretrained(model_name, use_fast=False)
        model = MBartForConditionalGeneration.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=False, trust_remote_code=True
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, trust_remote_code=True
        )

    # Set source language
    src_lang = lang_config.get("src_lang")
    if src_lang and hasattr(tokenizer, "lang_code_to_id"):
        if src_lang in tokenizer.lang_code_to_id:
            tokenizer.src_lang = src_lang
        else:
            LOGGER.warning("src_lang %s not in tokenizer vocab", src_lang)

    # Set target language and forced BOS token
    tgt_lang = lang_config.get("tgt_lang")
    forced_bos_token_id = None
    if tgt_lang and hasattr(tokenizer, "lang_code_to_id"):
        if tgt_lang in tokenizer.lang_code_to_id:
            tokenizer.tgt_lang = tgt_lang
            forced_bos_token_id = tokenizer.lang_code_to_id.get(tgt_lang)
        else:
            LOGGER.warning("tgt_lang %s not in tokenizer vocab", tgt_lang)

    # Add language-specific tokens
    extra_tokens = config["special_tokens"]
    if extra_tokens:
        num_added = tokenizer.add_tokens(extra_tokens)
        if num_added > 0:
            model.resize_token_embeddings(len(tokenizer))
            LOGGER.info("Added %d special tokens for %s", num_added, language)

    return tokenizer, model, forced_bos_token_id


# ---------------------------------------------------------------------------
# Tokenization + training
# ---------------------------------------------------------------------------

def tokenize_dataset(dataset, tokenizer, src_max_len: int, tgt_max_len: int):
    def fn(examples):
        model_inputs = tokenizer(
            examples["source_text"],
            max_length=src_max_len,
            truncation=True,
        )
        labels = tokenizer(
            text_target=examples["target_text"],
            max_length=tgt_max_len,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(fn, batched=True, remove_columns=["source_text", "target_text"])


def build_compute_metrics(tokenizer, language: str):
    def compute_metrics(eval_preds):
        predictions, labels = eval_preds
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [
            postprocess_translation(preprocess_text(x, language), language)
            for x in decoded_preds
        ]
        decoded_labels = [
            postprocess_translation(preprocess_text(x, language), language)
            for x in decoded_labels
        ]

        return compute_corpus_metrics(decoded_preds, decoded_labels)

    return compute_metrics


def save_json(data: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def plot_learning_curve(log_history, out_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train_losses = [e["loss"] for e in log_history if "loss" in e]
    eval_losses = [e["eval_loss"] for e in log_history if "eval_loss" in e]

    plt.figure(figsize=(10, 6))
    if train_losses:
        plt.plot(train_losses, label="Train loss", marker=".")
    if eval_losses:
        plt.plot(eval_losses, label="Val loss", marker=".")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()


def _build_training_args(args, output_dir: str):
    from transformers import Seq2SeqTrainingArguments

    kwargs = {
        "output_dir": output_dir,
        "save_strategy": "steps",
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "predict_with_generate": True,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "save_total_limit": args.save_total_limit,
        "logging_dir": os.path.join(output_dir, "logs"),
        "logging_steps": args.logging_steps,
        "load_best_model_at_end": True,
        "metric_for_best_model": args.metric_for_best_model,
        "greater_is_better": True,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "label_smoothing_factor": args.label_smoothing_factor,
        "generation_max_length": args.generation_max_length,
        "generation_num_beams": args.generation_num_beams,
        "seed": args.seed,
        "run_name": args.run_name,
        "report_to": "none",
        "dataloader_num_workers": 2,
    }

    # Handle API changes between transformers versions
    sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"

    return Seq2SeqTrainingArguments(**kwargs)


def _build_trainer(model, training_args, train_dataset, eval_dataset,
                   tokenizer, data_collator, compute_metrics, callbacks):
    from transformers import Seq2SeqTrainer

    kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
        "callbacks": callbacks,
    }

    sig = inspect.signature(Seq2SeqTrainer.__init__)
    if "processing_class" in sig.parameters:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sig.parameters:
        kwargs["tokenizer"] = tokenizer

    return Seq2SeqTrainer(**kwargs)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def train_command(args):
    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = args.plots_dir or os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    train_data, train_stats = load_parallel_data(
        args.train_src, args.train_tgt, args.language,
        args.lowercase, args.remove_special_chars, args.normalize_unicode,
        args.length_ratio_threshold, plots_dir,
    )
    val_data, val_stats = load_parallel_data(
        args.val_src, args.val_tgt, args.language,
        args.lowercase, args.remove_special_chars, args.normalize_unicode,
        args.length_ratio_threshold,
    )

    save_json({"train": train_stats, "val": val_stats},
              os.path.join(args.output_dir, "data_stats.json"))

    from datasets import Dataset
    from transformers import DataCollatorForSeq2Seq, EarlyStoppingCallback

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    tokenizer, model, forced_bos_token_id = load_tokenizer_and_model(
        args.model_name, args.language
    )
    tokenized_train = tokenize_dataset(train_dataset, tokenizer,
                                        args.source_max_length, args.target_max_length)
    tokenized_val = tokenize_dataset(val_dataset, tokenizer,
                                      args.source_max_length, args.target_max_length)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = _build_training_args(args, args.output_dir)

    if forced_bos_token_id is not None:
        model.config.forced_bos_token_id = forced_bos_token_id

    trainer = _build_trainer(
        model=model,
        training_args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer, args.language),
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience
        )],
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Final evaluation on val set
    predictions, labels, metrics = trainer.predict(tokenized_val)
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [
        postprocess_translation(preprocess_text(x, args.language), args.language)
        for x in decoded_preds
    ]
    decoded_labels = [
        postprocess_translation(preprocess_text(x, args.language), args.language)
        for x in decoded_labels
    ]

    pred_file = os.path.join(args.output_dir, "predictions.txt")
    ref_file = os.path.join(args.output_dir, "references.txt")
    _write_lines(decoded_preds, pred_file)
    _write_lines(decoded_labels, ref_file)

    final_metrics = {
        **metrics,
        **compute_corpus_metrics(decoded_preds, decoded_labels),
    }
    save_json(final_metrics, os.path.join(args.output_dir, "metrics.json"))
    plot_learning_curve(
        trainer.state.log_history,
        os.path.join(args.output_dir, "learning_curve.png"),
    )

    LOGGER.info("Training complete. Model saved to %s", args.output_dir)
    LOGGER.info("Final metrics: %s", final_metrics)


def predict_command(args):
    tokenizer, model = _load_model_for_inference(args.model_path, args.language)
    source_lines = _read_lines(args.input_file)
    predictions = _translate_lines(
        model, tokenizer, source_lines, args.language,
        args.batch_size, args.max_length, args.num_beams,
    )
    _write_lines(predictions, args.output_file)
    LOGGER.info("Saved %d translations to %s", len(predictions), args.output_file)


def eval_command(args):
    tokenizer, model = _load_model_for_inference(args.model_path, args.language)
    source_lines = _read_lines(args.input_file)
    references = [
        postprocess_translation(preprocess_text(x, args.language), args.language)
        for x in _read_lines(args.reference_file)
    ]

    predictions = _translate_lines(
        model, tokenizer, source_lines, args.language,
        args.batch_size, args.max_length, args.num_beams,
    )
    _write_lines(predictions, args.output_file)

    metrics = compute_corpus_metrics(predictions, references)
    metrics["num_examples"] = len(predictions)
    save_json(metrics, args.metrics_json)
    LOGGER.info("Metrics: %s", metrics)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def _write_lines(lines: List[str], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def _load_model_for_inference(model_path: str, language: str):
    from transformers import (
        AutoModelForSeq2SeqLM, AutoTokenizer,
        MBart50Tokenizer, MBartForConditionalGeneration,
    )

    family = detect_model_family(model_path)
    config = LANGUAGE_CONFIGS[language]
    lang_config = config.get(family, config.get("mbart", {}))

    if family == "mbart":
        tokenizer = MBart50Tokenizer.from_pretrained(model_path, use_fast=False)
        model = MBartForConditionalGeneration.from_pretrained(model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=False, trust_remote_code=True
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path, trust_remote_code=True
        )

    src_lang = lang_config.get("src_lang")
    tgt_lang = lang_config.get("tgt_lang")

    if src_lang and hasattr(tokenizer, "lang_code_to_id"):
        if src_lang in tokenizer.lang_code_to_id:
            tokenizer.src_lang = src_lang

    forced_bos_token_id = None
    if tgt_lang and hasattr(tokenizer, "lang_code_to_id"):
        if tgt_lang in tokenizer.lang_code_to_id:
            tokenizer.tgt_lang = tgt_lang
            forced_bos_token_id = tokenizer.lang_code_to_id.get(tgt_lang)

    if forced_bos_token_id is not None:
        model.config.forced_bos_token_id = forced_bos_token_id

    return tokenizer, model


def _translate_lines(
    model, tokenizer, lines: List[str], language: str,
    batch_size: int, max_length: int, num_beams: int,
) -> List[str]:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    outputs: List[str] = []
    for i in range(0, len(lines), batch_size):
        batch = lines[i:i + batch_size]
        normalized = [preprocess_text(x, "generic") for x in batch]
        encoded = tokenizer(
            normalized,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_length=max_length,
                num_beams=num_beams,
            )

        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        decoded = [
            postprocess_translation(preprocess_text(x, language), language)
            for x in decoded
        ]
        outputs.extend(decoded)

    return outputs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_shared_args(parser):
    parser.add_argument("--language", default="guarani",
                        choices=sorted(LANGUAGE_CONFIGS.keys()))
    parser.add_argument("--model-name", default="facebook/mbart-large-50")
    parser.add_argument("--source-max-length", type=int, default=128)
    parser.add_argument("--target-max-length", type=int, default=128)
    parser.add_argument("--length-ratio-threshold", type=float, default=2.5)
    parser.add_argument("--lowercase", action="store_true", default=True)
    parser.add_argument("--no-lowercase", dest="lowercase", action="store_false")
    parser.add_argument("--remove-special-chars", action="store_true", default=True)
    parser.add_argument("--keep-special-chars", dest="remove_special_chars",
                        action="store_false")
    parser.add_argument("--normalize-unicode", action="store_true", default=True)
    parser.add_argument("--no-normalize-unicode", dest="normalize_unicode",
                        action="store_false")
    parser.add_argument("--log-level", default="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- train ---
    tp = subparsers.add_parser("train")
    _add_shared_args(tp)
    tp.add_argument("--train-src", required=True)
    tp.add_argument("--train-tgt", required=True)
    tp.add_argument("--val-src", required=True)
    tp.add_argument("--val-tgt", required=True)
    tp.add_argument("--output-dir", required=True)
    tp.add_argument("--plots-dir", default=None)
    tp.add_argument("--learning-rate", type=float, default=3e-5)
    tp.add_argument("--weight-decay", type=float, default=0.01)
    tp.add_argument("--warmup-steps", type=int, default=800)
    tp.add_argument("--num-train-epochs", type=float, default=15.0)
    tp.add_argument("--train-batch-size", type=int, default=8)
    tp.add_argument("--eval-batch-size", type=int, default=8)
    tp.add_argument("--gradient-accumulation-steps", type=int, default=4)
    tp.add_argument("--eval-steps", type=int, default=500)
    tp.add_argument("--save-steps", type=int, default=500)
    tp.add_argument("--logging-steps", type=int, default=100)
    tp.add_argument("--save-total-limit", type=int, default=3)
    tp.add_argument("--label-smoothing-factor", type=float, default=0.1)
    tp.add_argument("--lr-scheduler-type", default="cosine")
    tp.add_argument("--early-stopping-patience", type=int, default=3)
    tp.add_argument("--fp16", action="store_true")
    tp.add_argument("--bf16", action="store_true")
    tp.add_argument("--seed", type=int, default=42)
    tp.add_argument("--metric-for-best-model", default="chrfpp")
    tp.add_argument("--generation-max-length", type=int, default=128)
    tp.add_argument("--generation-num-beams", type=int, default=4)
    tp.add_argument("--run-name", default=None)

    # --- predict ---
    pp = subparsers.add_parser("predict")
    _add_shared_args(pp)
    pp.add_argument("--model-path", required=True)
    pp.add_argument("--input-file", required=True)
    pp.add_argument("--output-file", required=True)
    pp.add_argument("--batch-size", type=int, default=16)
    pp.add_argument("--num-beams", type=int, default=4)
    pp.add_argument("--max-length", type=int, default=128)

    # --- eval ---
    ep = subparsers.add_parser("eval")
    _add_shared_args(ep)
    ep.add_argument("--model-path", required=True)
    ep.add_argument("--input-file", required=True)
    ep.add_argument("--reference-file", required=True)
    ep.add_argument("--output-file", required=True)
    ep.add_argument("--metrics-json", required=True)
    ep.add_argument("--batch-size", type=int, default=16)
    ep.add_argument("--num-beams", type=int, default=4)
    ep.add_argument("--max-length", type=int, default=128)

    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main():
    args = parse_args()
    setup_logging(args.log_level)

    if args.command == "train":
        train_command(args)
    elif args.command == "predict":
        predict_command(args)
    elif args.command == "eval":
        eval_command(args)


if __name__ == "__main__":
    main()