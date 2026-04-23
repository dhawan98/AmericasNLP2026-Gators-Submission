#!/usr/bin/env bash
# run_test_submissions.sh
# Run final test set predictions for all 5 languages using confirmed best configs.
# Run ONE language at a time. Check each output before running the next.
#
# Usage:
#   bash run_test_submissions.sh --language guarani
#   bash run_test_submissions.sh --language nahuatl
#   bash run_test_submissions.sh --language bribri
#   bash run_test_submissions.sh --language wixarika
#   bash run_test_submissions.sh --language maya
#
# Outputs go to: test_submissions/{language}/
# Each run also writes submission_metadata.json for reproducibility.

set -euo pipefail

BASE="/blue/daisyw/aashish.dhawan"
MBART="${BASE}/mbart/data"
BL="${BASE}/americasnlp2026/baseline"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ── Confirmed best configs (frozen after dev tuning) ──────────
# Language : config       : dev chrF++ : notes
# guarani  : r80 d49      : 41.48      : frozen since April 15
# nahuatl  : r40 d20      : 25.67      : plateau, within noise of 25.97
# bribri   : r80 d20      : 11.50      : corpus ceiling ~11.5
# wixarika : r40 d20      : 18.99      : new best, +0.33 over prev
# maya     : r0  d49      : ~26.2      : d20=26.29 d49=26.21, use d49 for test

declare -A NUM_RETRIEVAL=(
    [guarani]=80
    [nahuatl]=40
    [bribri]=80
    [wixarika]=40
    [maya]=0
)
declare -A NUM_DEV=(
    [guarani]=49
    [nahuatl]=20
    [bribri]=20
    [wixarika]=20
    [maya]=49
)
declare -A TRAIN_ES=(
    [guarani]="${BL}/data/parallel/guarani_with_dev/train.es"
    [nahuatl]="${MBART}/nahuatl-spanish/train.es"
    [bribri]="${MBART}/bribri-spanish/train.es"
    [wixarika]="${BL}/data/parallel/wixarika_with_dev/train.es"
    [maya]="${BL}/outputs/maya/dummy_train.es"
)
declare -A TRAIN_TGT=(
    [guarani]="${BL}/data/parallel/guarani_with_dev/train.gn"
    [nahuatl]="${MBART}/nahuatl-spanish/train.nah"
    [bribri]="${MBART}/bribri-spanish/train.bzd"
    [wixarika]="${BL}/data/parallel/wixarika_with_dev/train.hch"
    [maya]="${BL}/outputs/maya/dummy_train.maya"
)
# Dev captions: used as exemplar pool (leave-one-out)
declare -A DEV_CAP=(
    [guarani]="${BL}/outputs/guarani/72b_v3_clean_spanish.txt"
    [nahuatl]="${BL}/outputs/nahuatl/dev_caption_es_v1.txt"
    [bribri]="${BL}/outputs/bribri/dev_caption_es_v1.txt"
    [wixarika]="${BL}/outputs/wixarika/dev_caption_es_v1.txt"
    [maya]="${BL}/outputs/maya/dev_caption_es_v1.txt"
)
declare -A DEV_REF=(
    [guarani]="${BL}/outputs/guarani/dev_references.txt"
    [nahuatl]="${BL}/outputs/nahuatl/dev_references.txt"
    [bribri]="${BL}/outputs/bribri/dev_references.txt"
    [wixarika]="${BL}/outputs/wixarika/dev_references.txt"
    [maya]="${BL}/outputs/maya/dev_references.txt"
)
# Test captions (already generated)
declare -A TEST_CAP=(
    [guarani]="${BL}/outputs/guarani/test_clean_spanish.txt"
    [nahuatl]="${BL}/outputs/nahuatl/test_clean_spanish.txt"
    [bribri]="${BL}/outputs/bribri/test_clean_spanish.txt"
    [wixarika]="${BL}/outputs/wixarika/test_clean_spanish.txt"
    [maya]="${BL}/outputs/maya/test_clean_spanish.txt"
)

LANGUAGE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --language) LANGUAGE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done
[[ -z "$LANGUAGE" ]] && { echo "ERROR: --language required."; exit 1; }

LANG=$LANGUAGE
SUBMIT_DIR="${BL}/test_submissions/${LANG}"
mkdir -p "${SUBMIT_DIR}"

NR="${NUM_RETRIEVAL[$LANG]}"
ND="${NUM_DEV[$LANG]}"
TSRC="${TRAIN_ES[$LANG]}"
TTGT="${TRAIN_TGT[$LANG]}"
DCAP="${DEV_CAP[$LANG]}"
DREF="${DEV_REF[$LANG]}"
TCAP="${TEST_CAP[$LANG]}"

echo "═══════════════════════════════════════════════════════"
echo " FINAL TEST SUBMISSION — ${LANG^^}"
echo "═══════════════════════════════════════════════════════"
echo " Config  : r=${NR}  d=${ND}"
echo " Test cap: ${TCAP}"
echo " Output  : ${SUBMIT_DIR}/"
echo "═══════════════════════════════════════════════════════"

# Verify all input files exist
for f in "$TSRC" "$TTGT" "$DCAP" "$DREF" "$TCAP"; do
    [[ -f "$f" ]] || { echo "ERROR: missing: $f"; exit 1; }
done
echo "All input files verified."

# Count test examples
TEST_N=$(wc -l < "$TCAP")
echo "Test examples: ${TEST_N}"
echo

# ── Run translation ───────────────────────────────────────────
python "${BL}/translate_llm_manyshot_gemini.py" \
    --language    "${LANG}" \
    --train-src   "${TSRC}" \
    --train-tgt   "${TTGT}" \
    --input-spanish   "${TCAP}" \
    --output-preds    "${SUBMIT_DIR}/preds.txt" \
    --samples-tsv     "${SUBMIT_DIR}/samples.tsv" \
    --raw-jsonl       "${SUBMIT_DIR}/raw.jsonl" \
    --model gemini-2.5-flash \
    --num-retrieval   "${NR}" \
    --num-dev-examples "${ND}" \
    --dev-example-spanish "${DCAP}" \
    --dev-example-refs    "${DREF}" \
    --temperature 0.0 --sleep 0.15

# ── Verify output line count matches input ────────────────────
PRED_N=$(wc -l < "${SUBMIT_DIR}/preds.txt")
echo
echo "Input lines : ${TEST_N}"
echo "Output lines: ${PRED_N}"
if [[ "$PRED_N" -ne "$TEST_N" ]]; then
    echo "ERROR: line count mismatch! Check for failed API calls."
    exit 1
fi
echo "Line count OK."

# ── Write submission metadata ─────────────────────────────────
python3 - << PYEOF
import json, datetime
meta = {
    "language": "${LANG}",
    "model": "gemini-2.5-flash",
    "num_retrieval": ${NR},
    "num_dev_examples": ${ND},
    "use_rerank": False,
    "temperature": 0.0,
    "test_caption_file": "${TCAP}",
    "dev_caption_file": "${DCAP}",
    "dev_reference_file": "${DREF}",
    "train_src": "${TSRC}",
    "train_tgt": "${TTGT}",
    "dev_best_chrf": {
        "guarani": 41.48,
        "nahuatl": 25.67,
        "bribri":  11.50,
        "wixarika": 18.99,
        "maya":    26.29,
    }.get("${LANG}", None),
    "timestamp": "${TIMESTAMP}",
    "submitted_at": datetime.datetime.utcnow().isoformat() + "Z",
    "n_test_examples": ${TEST_N},
}
path = "${SUBMIT_DIR}/submission_metadata.json"
json.dump(meta, open(path, "w"), indent=2)
print(f"Metadata written: {path}")
PYEOF

echo
echo "═══════════════════════════════════════════════════════"
echo " DONE — ${LANG^^}"
echo " Predictions : ${SUBMIT_DIR}/preds.txt"
echo " Metadata    : ${SUBMIT_DIR}/submission_metadata.json"
echo " Raw JSONL   : ${SUBMIT_DIR}/raw.jsonl"
echo "═══════════════════════════════════════════════════════"
echo
echo "Next: run prepare_submission.py to format for workshop upload."
echo "  python ${BL}/prepare_submission.py \\"
echo "    --language ${LANG} \\"
echo "    --pred-file ${SUBMIT_DIR}/preds.txt \\"
echo "    --output-dir ${SUBMIT_DIR}"