**Visual Captioning for Indigenous Languages**
University of Florida · Dr. Daisy Zhe Wang's Lab

## Overview

System submitted by **Team Gators** to the [AmericasNLP 2026 Shared Task](https://github.com/AmericasNLP/americasnlp2026) on visual captioning for indigenous languages.

**Task:** Given an image, generate a caption in one of 5 indigenous languages.
**Metric:** chrF++

### Results

| Language | Config | Dev chrF++ | Test Examples |
|----------|--------|-----------|---------------|
| Guaraní (grn) | r=80, d=49 | **41.48** | 101 |
| Maya (yua) | r=0, d=49 | **26.29** | 212 |
| Nahuatl (nlv) | r=40, d=20 | **25.67** | 200 |
| Wixarika (hch) | r=40, d=20 | **18.99** | 201 |
| Bribri (bzd) | r=80, d=20 | **11.50** | 267 |

---

## Pipeline
**Stage 1 — VLM Captioning**
- Model: `Qwen/Qwen3-VL-8B-Instruct` (4-bit quantized, GPU required)
- Generates a short literal Spanish caption from the image
- Language-specific prompts in `prompts/`

**Stage 2 — Many-Shot LLM Translation**
- Model: `gemini-2.5-flash` via Gemini API (CPU only)
- BM25 retrieval over parallel training corpus (Spanish side)
- In-context: BM25-retrieved + leave-one-out dev examples
- Temperature: 0.0

---

## Setup

```bash
git clone https://github.com/dhawan98/AmericasNLP2026.git
cd AmericasNLP2026
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY="your_key_here"
```

Get task data:
```bash
git clone https://github.com/AmericasNLP/americasnlp2026.git
```

Parallel corpora (not in repo — available from task organizers):

| Language | Train pairs |
|----------|------------|
| Guaraní | 53,183 (data/parallel/guarani_with_dev/) 30k added from multiscript30k (https://github.com/ufdatastudio/multiscript30k) |
| Nahuatl | 16,145 (mbart/data/nahuatl-spanish/) |
| Bribri | 7,508 (mbart/data/bribri-spanish/) |
| Wixarika | 8,966 (data/parallel/wixarika_with_dev/) |
| Maya | none — pure few-shot |

---

## Reproduce: End-to-End

### Step 1 — Caption images (GPU required)
```bash
python caption_images.py \
    --input-jsonl  americasnlp2026/data/test/{lang}/{lang}.jsonl \
    --base-path    americasnlp2026/data/test/{lang} \
    --language     {lang} \
    --model-name   Qwen/Qwen3-VL-8B-Instruct \
    --prompt-file  prompts/{lang}/caption_es_v1.txt \
    --output-jsonl outputs/{lang}/test_caption_es.jsonl \
    --output-txt   outputs/{lang}/test_clean_spanish.txt \
    --quantize 4bit
```

### Step 2 — Translate captions (best configs)
```bash
bash run_test_submissions.sh --language guarani   # r=80 d=49
bash run_test_submissions.sh --language maya      # r=0  d=49
bash run_test_submissions.sh --language nahuatl   # r=40 d=20
bash run_test_submissions.sh --language bribri    # r=80 d=20
bash run_test_submissions.sh --language wixarika  # r=40 d=20
```

Or with custom settings:
```bash
python translate_llm_manyshot_gemini.py \
    --language       guarani \
    --train-src      data/parallel/guarani_with_dev/train.es \
    --train-tgt      data/parallel/guarani_with_dev/train.gn \
    --input-spanish  outputs/guarani/test_clean_spanish.txt \
    --output-preds   test_submissions/guarani/preds.txt \
    --model          gemini-2.5-flash \
    --num-retrieval  80 \
    --num-dev-examples 49 \
    --dev-example-spanish outputs/guarani/dev_caption_es_v1.txt \
    --dev-example-refs    outputs/guarani/dev_references.txt \
    --temperature    0.0
```

### Step 3 — Format for submission
```bash
python prepare_submission.py \
    --input-jsonl      americasnlp2026/data/test/{lang}/{lang}.jsonl \
    --predictions-file test_submissions/{lang}/preds.txt \
    --output-jsonl     test_submissions/{lang}/submission_{lang}.jsonl \
    --team-name "gators" --version 1
```

### Step 4 — Evaluate on dev
```bash
python score_outputs.py \
    --predictions outputs/{lang}/preds.txt \
    --references  outputs/{lang}/dev_references.txt
```

---
## Hyperparameter Tuning

```bash
bash run_improvements_v2.sh --language nahuatl --experiment retrieval-sweep
```

Key findings:
- **More retrieval helps** for high-resource languages (Guaraní r=80 >> r=20)
- **Zero retrieval is best for Maya** — Gemini's pretrained knowledge is sufficient; tiny corpus adds noise
- **Dev examples matter most** — d=0 drops chrF++ by ~9 points across all languages
- **Temperature 0.0** optimal for all languages at test time

---

## Key Findings

- **LLM >> fine-tuned MT**: Gemini 2.5 Flash many-shot outperforms mBART-50/NLLB by ~20 chrF++ on Guaraní (41.48 vs ~22)
- **Zero retrieval for Maya**: BM25 over a tiny out-of-domain corpus hurts performance
- **VLM quality is the bottleneck**: 54% of Guaraní errors are vision-based (hedged language from captioner)
- **Wixarika corpus mismatch**: Literary training data causes systematic degeneration for visual descriptions
- **Wixarika caption prompt v2 (cultural glossary) hurts**: -2.58 chrF++ vs v1

## Hardware

- Stage 1: NVIDIA B200 GPU, 4-bit quantization, ~3.5s/image
- Stage 2: CPU only, Gemini API
- Cluster: UF HiPerGator

---

## Citation

```bibtex
@inproceedings{gators-americasnlp2026,
  title     = {Cascading VLM Captioning and LLM Translation for Indigenous Language Visual Captioning},
  author    = {Dhawan, Aashish and others},
  booktitle = {Proceedings of AmericasNLP 2026},
  year      = {2026}
}
```

