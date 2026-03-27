#!/bin/bash
# Download AmericasNLP 2026 challenge data and set up directory structure.
#
# Usage:
#   bash scripts/download_data.sh [TARGET_DIR]

set -euo pipefail

TARGET_DIR="${1:-data}"

echo "=== Cloning AmericasNLP 2026 repository ==="
if [ ! -d "${TARGET_DIR}/americasnlp2026" ]; then
    git clone https://github.com/AmericasNLP/americasnlp2026.git "${TARGET_DIR}/americasnlp2026"
else
    echo "Repository already exists, pulling latest..."
    cd "${TARGET_DIR}/americasnlp2026" && git pull && cd -
fi

echo ""
echo "=== Data structure ==="
find "${TARGET_DIR}/americasnlp2026/data" -name "*.jsonl" -o -name "*.json" | sort
echo ""
echo "=== Image counts ==="
for lang_dir in "${TARGET_DIR}/americasnlp2026/data"/*/; do
    if [ -d "$lang_dir" ]; then
        for split_dir in "$lang_dir"*/; do
            if [ -d "$split_dir" ]; then
                img_count=$(find "$split_dir" -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" 2>/dev/null | wc -l)
                if [ "$img_count" -gt 0 ]; then
                    echo "  $(basename $(dirname $split_dir))/$(basename $split_dir): ${img_count} images"
                fi
            fi
        done
    fi
done

echo ""
echo "=== Cloning Sheffield Wixárika baseline (for fairseq checkpoint) ==="
if [ ! -d "${TARGET_DIR}/americasnlp-2023-sheffield" ]; then
    git clone https://github.com/davidguzmanr/americasnlp-2023-sheffield.git "${TARGET_DIR}/americasnlp-2023-sheffield"
else
    echo "Sheffield repo already exists."
fi

echo ""
echo "=== Downloading Wixárika translation checkpoint ==="
CKPT_PATH="${TARGET_DIR}/submission_3.pt"
if [ ! -f "$CKPT_PATH" ]; then
    echo "Downloading checkpoint (~1.5GB)..."
    if command -v aria2c &> /dev/null; then
        aria2c -x 16 -s 16 -k 1M \
            -d "${TARGET_DIR}" \
            https://datasets-and-checkpoints.s3.us-east-1.amazonaws.com/americasnlp-2026/submission_3.pt
    else
        wget -O "$CKPT_PATH" \
            https://datasets-and-checkpoints.s3.us-east-1.amazonaws.com/americasnlp-2026/submission_3.pt
    fi
else
    echo "Checkpoint already exists."
fi

echo ""
echo "=== Cloning your mBART50-extended repo (for Guaraní training data) ==="
if [ ! -d "${TARGET_DIR}/mBART50-extended" ]; then
    git clone https://github.com/dhawan98/mBART50-extended.git "${TARGET_DIR}/mBART50-extended"
else
    echo "mBART50-extended repo already exists."
fi

echo ""
echo "=== Setup complete ==="
echo "Data directory: ${TARGET_DIR}"
echo ""
echo "Next steps:"
echo "  1. Check data/americasnlp2026/data/dev/ for dev sets"
echo "  2. Check data/americasnlp2026/data/pilot/ for pilot data"
echo "  3. Prepare parallel data for MT training"