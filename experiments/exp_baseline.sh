#!/bin/bash
# Experiment 1: Baseline — Run pipeline with default settings on all dev sets.
#
# This establishes baseline ChrF++ scores for each language.
# Run AFTER training MT models for each language.
#
# Usage: bash experiments/exp_baseline.sh

set -euo pipefail

LANGUAGES=("guarani" "wixarika" "bribri" "maya")
VLM="Qwen/Qwen3-VL-8B-Instruct"
QUANTIZE="4bit"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/baseline_${TIMESTAMP}"

mkdir -p "$RESULTS_DIR"

echo "=== Baseline Experiment: ${TIMESTAMP} ==="
echo "Results will be saved to: ${RESULTS_DIR}"

for LANG in "${LANGUAGES[@]}"; do
    echo ""
    echo "--- Processing: ${LANG} ---"

    OUTPUT_DIR="${RESULTS_DIR}/${LANG}"
    MT_MODEL="models/${LANG}_mbart50"

    # Check if MT model exists
    if [ ! -d "$MT_MODEL" ]; then
        echo "WARNING: No MT model found at ${MT_MODEL}, skipping ${LANG}"
        continue
    fi

    python run_pipeline.py \
        --input-jsonl "data/americasnlp2026/data/dev/${LANG}/${LANG}.jsonl" \
        --base-path "data/americasnlp2026/data/dev/${LANG}" \
        --language "$LANG" \
        --vlm-model "$VLM" \
        --mt-model-path "$MT_MODEL" \
        --output-dir "$OUTPUT_DIR" \
        --quantize "$QUANTIZE" \
        2>&1 | tee "${OUTPUT_DIR}/run.log"
done

# Aggregate results
echo ""
echo "=========================================="
echo "BASELINE RESULTS SUMMARY"
echo "=========================================="
for LANG in "${LANGUAGES[@]}"; do
    METRICS_FILE="${RESULTS_DIR}/${LANG}/metrics.json"
    if [ -f "$METRICS_FILE" ]; then
        CHRFPP=$(python3 -c "import json; print(f'{json.load(open(\"${METRICS_FILE}\"))[\"chrfpp\"]:.2f}')")
        echo "  ${LANG}: ChrF++ = ${CHRFPP}"
    else
        echo "  ${LANG}: SKIPPED"
    fi
done
echo "=========================================="