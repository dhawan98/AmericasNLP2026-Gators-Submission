# AmericasNLP 2026 — Cultural Image Captioning for Indigenous Languages

## Challenge Summary

**Task:** Given culturally situated images, generate captions in Indigenous languages.  
**Pipeline:** Image → Spanish caption (VLM) → Indigenous language caption (MT)  
**Metric:** ChrF++ (Stage 1); human judges for top-5 (Stage 2)  
**Languages:** Wixárika (pilot), Bribri, Guaraní, Maya (dev), possibly surprise languages  
**Data:** ~50 dev examples per language with images + target captions. NO Spanish captions in dev/test.

## Key Dates (all AoE)

| Date | Milestone | Status |
|------|-----------|--------|
| Feb 20, 2025 | Pilot data + baseline | ✅ Released |
| Mar 1, 2025 | Dev sets (50 examples) | ✅ Released (Bribri, Guaraní, Maya, Wixárika) |
| Apr 1, 2025 | Surprise languages | ⏳ Check repo |
| Apr 20, 2025 | Test sets released | ⏳ |
| May 1, 2025 | **Submission deadline** | 🎯 |
| Jul 3-4, 2025 | Workshop at ACL 2026 (San Diego) | Paper deadline: Apr 15, 2026 |

**UPDATE:** Dev set CAN be used for training (per repo README).

## Pipeline Architecture

```
[Image] → caption_images.py (VLM: Qwen3-VL) → [Spanish Caption]
                                                       ↓
                                              translate_mt.py (mBART50)
                                                       ↓
                                              [Indigenous Language Caption]
                                                       ↓
                                              score_outputs.py (ChrF++)
```

## Repository Structure

```
americasnlp2026_project/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── caption_images.py            # Step 1: VLM image captioning (→ Spanish)
├── translate_mt.py              # Step 2: MT translation (Spanish → target)
├── score_outputs.py             # Step 3: Evaluation (ChrF++, BLEU, chrF)
├── run_pipeline.py              # End-to-end pipeline orchestrator
├── prepare_submission.py        # Format output for submission
├── configs/
│   ├── guarani.yaml             # Language-specific config
│   ├── wixarika.yaml
│   ├── bribri.yaml
│   └── maya.yaml
├── experiments/
│   ├── exp_baseline.sh          # Baseline experiment
│   ├── exp_augmented.sh         # With synthetic data
│   └── exp_ablation.sh          # Ablation studies
└── scripts/
    ├── download_data.sh         # Clone challenge repo + data
    └── setup_hipergator.sh      # HiPerGator SLURM setup
```

## Working Execution Plan

### Phase 0: Setup— Week of Mar 24
- [ ] Clone challenge repo, download all data (pilot + dev for all 4 languages)
- [ ] Set up HiPerGator environment (conda env, dependencies)
- [ ] Verify L4 GPU compatibility with PyTorch + transformers
- [ ] Run baseline notebook end-to-end on Wixárika pilot data

### Phase 1: Stabilize Captioning (3 hrs) — Week of Mar 31
- [ ] Test Qwen3-VL-8B on L4 (may need 4-bit quantization for 24GB VRAM)
- [ ] Benchmark captioning quality on pilot Spanish captions vs generated
- [ ] Tune prompts per language (Guaraní, Bribri, Maya cultural context)
- [ ] Generate Spanish captions for ALL dev sets

### Phase 2: Translation Models (5 hrs) — Week of Apr 7
- [ ] **Guaraní:** Fine-tune mBART50 using your existing parallel data + synthetic augmentation
- [ ] **Wixárika:** Use Sheffield fairseq checkpoint (baseline), also try mBART50 fine-tune
- [ ] **Bribri + Maya:** Gather parallel data from AmericasNLP 2021-2025 repos, fine-tune mBART50
- [ ] Run experiments comparing base vs augmented vs ensemble

### Phase 3: Optimize for ChrF++ (3 hrs) — Week of Apr 14
- [ ] Beam search tuning (beams=4,5,6)
- [ ] Length penalty tuning
- [ ] Ensemble decoding if multiple models available
- [ ] Dev set evaluation across all languages
- [ ] Error analysis: what types of images/captions score lowest?

### Phase 4: Test + Submit (2 hrs) — Apr 20-May 1
- [ ] Download test sets when released (Apr 20)
- [ ] Run full pipeline on test images
- [ ] Format submission per challenge spec
- [ ] Submit by May 1

## Competitive Strategy for Top-5

1. **Better VLM Captioning:** Use language-specific prompts with cultural context (your Wixárika template is great — replicate for each language)
2. **Data Augmentation:** Synthetic parallel data via back-translation (proven in your arxiv paper)
3. **Multi-model Ensembles:** Average or MBR decoding across mBART50 + NLLB-200
4. **NLLB-200 as Additional Backbone:** Has better coverage of low-resource languages
5. **Post-editing with LLMs:** Use Claude/GPT to fix obvious Spanish caption errors before MT
6. **Dev Set for Training:** Rules now allow using dev data for training — use it

## Issues Found in Current Modularization

See detailed analysis in the code files below.


# AmericasNLP 2026 Baseline Reproduction

This repository contains our code for the AmericasNLP 2026 Shared Task on Cultural Image Captioning for Indigenous Languages.

## What this repo contains

This repo is intentionally **code-only**. It includes:
- preprocessing scripts
- caption generation scripts
- translation scripts
- evaluation / scoring scripts
- configs and helper scripts

It does **not** include:
- shared task datasets
- image folders
- MT checkpoints
- training runs / optimizer states
- generated outputs

These large artifacts are excluded from Git so collaborators can clone the repo quickly.

---

## Official task resources

### AmericasNLP 2026 shared task
Official repository:
- `AmericasNLP/americasnlp2026`

Official task data includes:
- `data/pilot/`
- `data/dev/`

The official README states that:
- pilot and development datasets are distributed as **JSONL files plus corresponding images**
- development data is available for **Bribri, Guaraní, Maya, and Wixárika**
- **Spanish captions appear only in the pilot set** and should not be relied on for system building

### Official 2026 baseline
The official baseline is a **generate-then-translate** pipeline:
1. Generate a culturally-informed **Spanish caption** from the image using **Qwen3-VL-8B-Instruct**
2. Translate Spanish into the Indigenous target language using **Sheffield’s winning AmericasNLP 2023 MT system**

The official baseline README says the provided Colab notebook:
- clones repositories
- installs dependencies
- downloads the MT checkpoint
- was run on an **A100 GPU**

### Sheffield MT baseline
Official repository:
- `edwardgowsmith/americasnlp-2023-sheffield`

The Sheffield README states that:
- it contains the code to reproduce their AmericasNLP 2023 submission
- it is also reused as a baseline in later shared tasks
- it provides links to download their **best-performing models**
- `process_data.sh` is used to download and process training/evaluation data
- “Submission 3” is the relevant best single-checkpoint setup

---

## What you need to download separately

### 1. AmericasNLP 2026 shared task data
You need the official shared task data from the AmericasNLP 2026 repository.

Expected components:
- pilot JSONL files
- pilot image folders
- development JSONL files
- development image folders

In the official data:
- pilot includes Spanish captions
- development/test do not

### 2. Sheffield MT resources
You need the Sheffield MT system resources separately.

Expected components:
- Sheffield codebase
- any required `fairseq` setup from that repo
- processed MT data if you plan to retrain
- the best-performing checkpoint / model files if you only want inference

### 3. Vision-language model access
The official baseline uses:
- `Qwen/Qwen3-VL-8B-Instruct`

You will need Hugging Face access and enough GPU memory to run it comfortably.

---

## Recommended local layout

Keep all large assets outside Git. A practical local layout is:

```text
project_root/
├── baseline/                         # this repo
├── americasnlp2026/                  # official shared task repo / data
├── americasnlp-2023-sheffield/       # Sheffield MT repo
├── data/                             # local processed data
├── runs/                             # training checkpoints
└── outputs/                          # predictions / scoring outputs
