#!/usr/bin/env python3
"""Step 1: Caption task images into Spanish using a Vision-Language Model.

Usage:
    # Basic usage with Qwen3-VL on dev data
    python caption_images.py \
        --input-jsonl data/dev/guarani/guarani.jsonl \
        --base-path data/dev/guarani \
        --language guarani \
        --model-name Qwen/Qwen3-VL-8B-Instruct \
        --output-jsonl outputs/guarani/captions.jsonl \
        --output-txt outputs/guarani/spanish_captions.txt

    # With 4-bit quantization for L4 GPU (24GB VRAM)
    python caption_images.py \
        --input-jsonl data/dev/guarani/guarani.jsonl \
        --base-path data/dev/guarani \
        --language guarani \
        --model-name Qwen/Qwen3-VL-8B-Instruct \
        --output-jsonl outputs/guarani/captions.jsonl \
        --quantize 4bit
"""
from __future__ import annotations
import re
import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from PIL import Image
from tqdm import tqdm

LOGGER = logging.getLogger("caption_images")

# ---------------------------------------------------------------------------
# Language-specific prompts
# ---------------------------------------------------------------------------
# IMPORTANT: The shared task does NOT provide Spanish captions in dev/test.
# These prompts ask the VLM to generate a LITERAL Spanish description of
# what is visible in the image. Cultural context helps the VLM recognize
# artifacts but the caption itself should stay grounded in what's visible.
# ---------------------------------------------------------------------------

PROMPTS: Dict[str, str] = {
    "generic": (
        "Describe this image in Spanish with one short, literal caption (2-3 sentences max). "
        "Mention only clearly visible people, objects, actions, clothing, and setting. "
        "Do not speculate or add interpretation beyond what is visually obvious."
    ),
    "guarani": (
        "Escribe UNA sola leyenda breve en español para esta imagen (2-3 oraciones máximo). "
        "Sé literal y concreto. Describe solo lo visible: personas, objetos, ropa, acción y entorno. "
        "No inventes contexto. Si hay elementos culturales guaraníes visibles "
        "(artesanía, instrumentos, vestimenta tradicional, yerba mate, naturaleza), "
        "menciónalos solo si son claramente visibles."
    ),
    "wixarika": (
        "Escribe UNA sola leyenda breve en español para esta imagen (2-3 oraciones máximo). "
        "Sé literal y concreto. Describe solo lo visible: personas, objetos, ropa, acción y entorno. "
        "Si hay elementos culturales wixárika visibles (arte de estambre, chaquira, "
        "nierika, ojo de Dios, trajes bordados, arquitectura tradicional), "
        "menciónalos solo si son claramente visibles. No especules."
    ),
    "bribri": (
        "Escribe UNA sola leyenda breve en español para esta imagen (2-3 oraciones máximo). "
        "Sé literal y concreto. Describe solo lo visible: personas, objetos, ropa, acción y entorno. "
        "Si hay elementos culturales bribri visibles (artesanía, naturaleza tropical, "
        "arquitectura de madera, cacao, selva), menciónalos solo si son claramente visibles."
    ),
    "maya": (
        "Escribe UNA sola leyenda breve en español para esta imagen (2-3 oraciones máximo). "
        "Sé literal y concreto. Describe solo lo visible: personas, objetos, ropa, acción y entorno. "
        "Si hay elementos culturales mayas visibles (ruinas, pirámides, glifos, bordados, "
        "huipil, milpa, hamacas, cenotes), menciónalos solo si son claramente visibles."
    ),
    "nahuatl": (
        "Escribe UNA sola leyenda breve en español para esta imagen (2-3 oraciones máximo). "
        "Sé literal y concreto. Describe solo lo visible: personas, objetos, ropa, acción y entorno. "
        "Si hay elementos culturales nahuas visibles (textiles, bordados, mercado, cerámica, "
        "iglesia, danza, instrumentos, maíz, metate, comal, arquitectura comunitaria), "
        "menciónalos solo si son claramente visibles. No especules."
    ),
}

# Fields where the image path might be stored in different JSONL formats
IMAGE_KEYS = [
    "image", "image_path", "image_file", "img",
    "file_name", "filename", "path",
]


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at line {line_no} in {path}: {e}") from e
    return rows


def dump_jsonl(rows: Iterable[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def dump_txt(lines: Iterable[str], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            # Flatten multi-line captions to single line
            f.write((line or "").replace("\n", " ").strip() + "\n")


def read_prompt(language: str, prompt_file: Optional[str]) -> str:
    if prompt_file:
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    return PROMPTS.get(language.lower(), PROMPTS["generic"])

def clean_caption_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^(¡Claro!|Aquí tienes.*?:)\s*", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    parts = [p.strip() for p in parts if p.strip()]
    return " ".join(parts[:2]).strip()


BAD_VAGUE_TERMS = [
    "parece", "probablemente", "podría", "tal vez", "quizá",
    "hermosa escena", "tradicional escena", "cultura vibrante",
    "evoca", "simboliza", "representa", "posiblemente",
    "probablemente en", "podría ser", "como si fuera",
]

BAD_LOCATION_TERMS = [
    "mercado 4", "mercado de las yuyos", "plaza 25 de mayo", "plaza 25 de abril",
    "caacupé", "corrientes", "concepción", "ybyty curupay",
    "parque 12 de octubre", "parque 3 de febrero",
]

VISIBLE_OBJECT_TERMS = [
    "persona", "hombre", "mujer", "niño", "niña",
    "mesa", "silla", "plato", "vaso", "taza", "pan",
    "flor", "árbol", "pájaro", "sombrero", "vestido",
    "cesta", "madera", "barro", "tejido", "escultura",
    "casa", "choza", "fuego", "calle", "mercado",
]

def score_caption_candidate(text: str) -> tuple:
    t = text.lower().strip()
    words = t.split()
    length_words = len(words)
    sentence_count = max(1, len(re.findall(r"[.!?]", t)))

    vague_hits = sum(1 for w in BAD_VAGUE_TERMS if w in t)
    location_hits = sum(1 for w in BAD_LOCATION_TERMS if w in t)
    visible_hits = sum(1 for w in VISIBLE_OBJECT_TERMS if w in t)

    score = 0.0
    score += 1.25 * visible_hits
    score -= 2.5 * vague_hits
    score -= 4.0 * location_hits
    score -= 0.30 * max(0, length_words - 32)

    if sentence_count > 2:
        score -= 2.0 * (sentence_count - 2)

    return (score, -length_words)

def debug_candidate_scores(candidates: List[str]) -> List[tuple]:
    scored = []
    for c in candidates:
        scored.append((score_caption_candidate(c), c))
    scored.sort(reverse=True)
    return scored
# ---------------------------------------------------------------------------
# Image path resolution
# ---------------------------------------------------------------------------

def infer_image_key(row: dict, explicit_key: Optional[str] = None) -> str:
    if explicit_key:
        if explicit_key not in row:
            raise KeyError(
                f"Requested image field '{explicit_key}' not found in row keys: {list(row.keys())}"
            )
        return explicit_key
    for key in IMAGE_KEYS:
        if key in row and row[key]:
            return key
    raise KeyError(f"Could not infer image field. Available keys: {list(row.keys())}")


def resolve_image_path(
    raw_path: str,
    input_jsonl: str,
    base_path: Optional[str] = None,
) -> Path:
    """Resolve image path trying multiple strategies."""
    raw_path = str(raw_path).strip()
    jsonl_parent = Path(input_jsonl).resolve().parent
    base = Path(base_path).resolve() if base_path else None
    raw = Path(raw_path)
    basename = raw.name

    candidates: List[Path] = []

    # Absolute path
    if raw.is_absolute():
        candidates.append(raw)

    # Relative to CWD
    candidates.append(Path(raw_path).resolve())

    # Relative to base_path
    if base is not None:
        candidates.append((base / raw).resolve())
        candidates.append((base / basename).resolve())
        candidates.append((base / "images" / basename).resolve())

    # Relative to JSONL parent
    candidates.append((jsonl_parent / raw).resolve())
    candidates.append((jsonl_parent / basename).resolve())
    candidates.append((jsonl_parent / "images" / basename).resolve())

    # Strip leading "data/" prefix
    parts = list(raw.parts)
    if parts and parts[0] == "data":
        stripped = Path(*parts[1:])
        if base is not None:
            candidates.append((base / stripped).resolve())
        candidates.append((jsonl_parent / stripped).resolve())

    # Strip split dirs
    for anchor in ["train", "pilot", "dev", "test"]:
        if anchor in parts:
            idx = parts.index(anchor)
            after = parts[idx + 1:]
            if after:
                stripped = Path(*after)
                if base is not None:
                    candidates.append((base / stripped).resolve())
                candidates.append((jsonl_parent / stripped).resolve())

    # Language subdirs
    if base is not None:
        for lang_dir in ["guarani", "wixarika", "bribri", "maya", "yucatec_maya", "nahuatl"]:
            candidates.append((base / lang_dir / "images" / basename).resolve())
            candidates.append((base / lang_dir / basename).resolve())

    # Deduplicate preserving order
    seen = set()
    unique = []
    for p in candidates:
        s = str(p)
        if s not in seen:
            seen.add(s)
            unique.append(p)

    for cand in unique:
        if cand.exists():
            return cand

    tried = "\n".join(f"  - {p}" for p in unique[:30])
    raise FileNotFoundError(
        f"Image not found for raw path: {raw_path}\n"
        f"JSONL: {input_jsonl}\n"
        f"base_path: {base_path}\n"
        f"Tried:\n{tried}"
    )


# ---------------------------------------------------------------------------
# Model loading with quantization support
# ---------------------------------------------------------------------------

def load_model_and_processor(model_name: str, quantize: Optional[str] = None):
    """Load VLM with optional quantization for GPU memory constraints.

    Args:
        model_name: HuggingFace model name
        quantize: None, "4bit", or "8bit" for bitsandbytes quantization
    """
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info("Using device=%s model=%s quantize=%s", device, model_name, quantize)

    load_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto" if device == "cuda" else None,
    }

    if quantize == "4bit" and device == "cuda":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        LOGGER.info("Using 4-bit quantization (recommended for L4 24GB)")
    elif quantize == "8bit" and device == "cuda":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        LOGGER.info("Using 8-bit quantization")
    elif device == "cuda":
        load_kwargs["torch_dtype"] = torch.float16

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(model_name, **load_kwargs)
    for attr in ["temperature", "top_p", "top_k"]:
        if hasattr(model.generation_config, attr):
            setattr(model.generation_config, attr, None)

    return model, processor, device


# ---------------------------------------------------------------------------
# Caption generation
# ---------------------------------------------------------------------------

def generate_caption(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    device: str,
    max_new_tokens: int = 96,
    num_beams: int = 6,
    num_return_sequences: int = 1,
) -> List[str]:
    import torch

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    inputs = processor(text=[text], images=[image], return_tensors="pt")

    if device == "cuda":
        inputs = {
            k: v.to(model.device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            max_new_tokens=max_new_tokens,
            no_repeat_ngram_size=3,
            repetition_penalty=1.1,
            length_penalty=0.9,
        )

    # Remove prompt tokens
    # Remove prompt tokens
    input_len = inputs["input_ids"].shape[1]
    generated_only = generated_ids[:, input_len:]

    output_texts = processor.batch_decode(
        generated_only,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    output_texts = [" ".join(x.split()).strip() for x in output_texts]
    return output_texts
def verify_caption(
    model,
    processor,
    image: Image.Image,
    caption: str,
    device: str,
    max_new_tokens: int = 64,
) -> str:
    import torch

    verify_prompt = f"""
Analiza esta imagen y la siguiente descripción en español.

Indica si la descripción contiene detalles que NO están claramente sustentados por la imagen.

Responde SOLO con una de estas dos opciones:
OK
o
UNSUPPORTED: <lista breve de palabras o frases no sustentadas>

Descripción:
{caption}
""".strip()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": verify_prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(text=[text], images=[image], return_tensors="pt")

    if device == "cuda":
        inputs = {
            k: v.to(model.device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            num_return_sequences=1,
            max_new_tokens=max_new_tokens,
        )

    input_len = inputs["input_ids"].shape[1]
    generated_only = generated_ids[:, input_len:]

    out = processor.batch_decode(
        generated_only,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0].strip()

    return " ".join(out.split())

def make_literal_fallback_prompt(original_prompt: str) -> str:
    return original_prompt + "\n\nMUY IMPORTANTE:\n- Describe solo objetos, personas, acciones y entorno claramente visibles.\n- No nombres lugares, monumentos, santos, especies, comidas o artefactos específicos si no estás completamente seguro.\n- Si dudas, usa una descripción visual general."

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Caption task images into Spanish using a VLM."
    )
    parser.add_argument("--input-jsonl", required=True,
                        help="Input task JSONL file.")
    parser.add_argument("--base-path", default=None,
                        help="Base path to resolve relative image paths.")
    parser.add_argument("--image-field", default=None,
                        help="Explicit JSON field holding image path.")
    parser.add_argument("--language", default="generic",
                        choices=sorted(PROMPTS.keys()),
                        help="Language profile for prompt selection.")
    parser.add_argument("--prompt-file", default=None,
                        help="Optional text file with a custom prompt.")
    parser.add_argument("--model-name", required=True,
                        help="HF model name, e.g. Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--output-jsonl", required=True,
                        help="Output JSONL with Spanish captions added.")
    parser.add_argument("--output-txt", default=None,
                        help="Optional plain text output (one caption per line).")
    parser.add_argument("--caption-field", default="caption_es",
                        help="Field name to store generated Spanish caption.")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--quantize", choices=["4bit", "8bit"], default=None,
                        help="Quantization for limited GPU memory (L4: use 4bit).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N rows (for debugging).")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--num-beams", type=int, default=6)
    parser.add_argument("--num-return-sequences", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    LOGGER.info("Loading input rows from %s", args.input_jsonl)
    rows = load_jsonl(args.input_jsonl)
    if args.limit is not None:
        rows = rows[:args.limit]
    LOGGER.info("Loaded %d rows", len(rows))

    prompt = read_prompt(args.language, args.prompt_file)
    LOGGER.info("Using prompt: %s", prompt[:100] + "...")

    model, processor, device = load_model_and_processor(
        args.model_name, args.quantize
    )

    captioned_rows: List[dict] = []
    caption_lines: List[str] = []
    errors: List[str] = []

    for i, row in enumerate(tqdm(rows, desc="Captioning")):
        candidates = []
        try:
            image_key = infer_image_key(row, args.image_field)
            raw_image_path = row[image_key]
            image_path = resolve_image_path(
                raw_path=raw_image_path,
                input_jsonl=args.input_jsonl,
                base_path=args.base_path,
            )

            image = Image.open(image_path).convert("RGB")
            raw_candidates = generate_caption(
                model=model,
                processor=processor,
                image=image,
                prompt=prompt,
                device=device,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                num_return_sequences=args.num_return_sequences,
            )
            if i < 5:
                LOGGER.info("Row %d raw candidates: %s", i, raw_candidates)

            candidates = [x.strip() for x in raw_candidates if x.strip()]
            caption = candidates[0] if candidates else ""

            if caption:
                verification = verify_caption(
                    model=model,
                    processor=processor,
                    image=image,
                    caption=caption,
                    device=device,
                )
                if verification.startswith("UNSUPPORTED:"):
                    fallback_prompt = make_literal_fallback_prompt(prompt)
                    fallback_candidates = generate_caption(
                        model=model,
                        processor=processor,
                        image=image,
                        prompt=fallback_prompt,
                        device=device,
                        max_new_tokens=args.max_new_tokens,
                        num_beams=args.num_beams,
                        num_return_sequences=1,
                    )
                    fallback_candidates = [x.strip() for x in fallback_candidates if x.strip()]
                    if fallback_candidates:
                        caption = fallback_candidates[0]
                        candidates[0] = caption
                    
        except Exception as e:
            LOGGER.error("Failed on row %d: %s", i, e)
            caption = ""
            errors.append(f"Row {i}: {e}")

        new_row = dict(row)
        new_row[args.caption_field] = caption
        new_row["caption_candidates_es"] = candidates if caption else []
        captioned_rows.append(new_row)
        caption_lines.append(caption)

    dump_jsonl(captioned_rows, args.output_jsonl)
    LOGGER.info("Wrote captioned JSONL to %s", args.output_jsonl)

    if args.output_txt:
        dump_txt(caption_lines, args.output_txt)
        LOGGER.info("Wrote caption text file to %s", args.output_txt)

    if errors:
        LOGGER.warning("%d errors occurred:", len(errors))
        for e in errors:
            LOGGER.warning("  %s", e)


if __name__ == "__main__":
    main()