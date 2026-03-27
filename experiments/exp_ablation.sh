#!/bin/bash
# Experiment 3: Ablation Studies
#
# Tests the effect of:
#   1. Beam search width (1, 4, 6, 8)
#   2. VLM prompt variants (short vs detailed cultural context)
#   3. MT backbone (mBART50 vs NLLB-200)
#
# Usage: bash experiments/exp_ablation.sh LANGUAGE
# Example: bash experiments/exp_ablation.sh guarani

set -euo pipefail

LANG="${1:?Usage: exp_ablation.sh LANGUAGE}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/ablation_${LANG}_${TIMESTAMP}"
INPUT_JSONL="data/americasnlp2026/data/dev/${LANG}/${LANG}.jsonl"
BASE_PATH="data/americasnlp2026/data/dev/${LANG}"

mkdir -p "$RESULTS_DIR"

echo "=== Ablation Experiment for ${LANG}: ${TIMESTAMP} ==="

# -----------------------------------------------------------------------
# Ablation 1: Beam search width
# -----------------------------------------------------------------------
echo ""
echo "--- Ablation 1: Beam Search Width ---"

# First, generate Spanish captions once (shared across beam ablations)
SHARED_CAPTIONS="${RESULTS_DIR}/spanish_captions.txt"
MT_MODEL="models/${LANG}_mbart50"

if [ ! -f "$SHARED_CAPTIONS" ]; then
    python caption_images.py \
        --input-jsonl "$INPUT_JSONL" \
        --base-path "$BASE_PATH" \
        --language "$LANG" \
        --model-name "Qwen/Qwen3-VL-8B-Instruct" \
        --output-jsonl "${RESULTS_DIR}/captions.jsonl" \
        --output-txt "$SHARED_CAPTIONS" \
        --quantize 4bit
fi

for BEAMS in 1 4 6 8; do
    echo "  Testing beams=${BEAMS}..."
    OUT_DIR="${RESULTS_DIR}/beams_${BEAMS}"

    python run_pipeline.py \
        --input-jsonl "$INPUT_JSONL" \
        --base-path "$BASE_PATH" \
        --language "$LANG" \
        --mt-model-path "$MT_MODEL" \
        --output-dir "$OUT_DIR" \
        --skip-captioning \
        --spanish-captions-file "$SHARED_CAPTIONS" \
        --num-beams "$BEAMS" \
        2>&1 | tee "${OUT_DIR}/run.log"
done

# -----------------------------------------------------------------------
# Ablation 2: MT backbone (if NLLB model exists)
# -----------------------------------------------------------------------
NLLB_MODEL="models/${LANG}_nllb200"
if [ -d "$NLLB_MODEL" ]; then
    echo ""
    echo "--- Ablation 2: NLLB-200 vs mBART50 ---"

    OUT_DIR="${RESULTS_DIR}/nllb200"
    python run_pipeline.py \
        --input-jsonl "$INPUT_JSONL" \
        --base-path "$BASE_PATH" \
        --language "$LANG" \
        --mt-model-path "$NLLB_MODEL" \
        --output-dir "$OUT_DIR" \
        --skip-captioning \
        --spanish-captions-file "$SHARED_CAPTIONS" \
        2>&1 | tee "${OUT_DIR}/run.log"
fi

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
echo ""
echo "=========================================="
echo "ABLATION RESULTS: ${LANG}"
echo "=========================================="
for d in "${RESULTS_DIR}"/*/; do
    NAME=$(basename "$d")
    METRICS="${d}/metrics.json"
    if [ -f "$METRICS" ]; then
        CHRFPP=$(python3 -c "import json; print(f'{json.load(open(\"${METRICS}\"))[\"chrfpp\"]:.2f}')")
        echo "  ${NAME}: ChrF++ = ${CHRFPP}"
    fi
done
echo "=========================================="