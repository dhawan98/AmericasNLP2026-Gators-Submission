#!/usr/bin/env bash
# run_improvements.sh  (v2 — all paths hardcoded from discover_paths.sh output)
# Run from: /blue/daisyw/aashish.dhawan/americasnlp2026/baseline/
#
# Usage:
#   bash run_improvements.sh --language nahuatl --experiment prompt-only
#   bash run_improvements.sh --language bribri  --experiment prompt-only
#   bash run_improvements.sh --language wixarika --experiment prompt-only
#   bash run_improvements.sh --language maya    --experiment prompt-only
#   bash run_improvements.sh --language guarani --experiment caption-filter
#   bash run_improvements.sh --language nahuatl  # runs all experiments for that language

set -euo pipefail

BASE="/blue/daisyw/aashish.dhawan"
MBART="${BASE}/mbart/data"
BL="${BASE}/americasnlp2026/baseline"

# ── All paths confirmed from discover_paths.sh ────────────────
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
declare -A TEST_CAP=(
    [guarani]="${BL}/outputs/guarani/test_clean_spanish.txt"
    [nahuatl]="${BL}/outputs/nahuatl/test_clean_spanish.txt"
    [bribri]="${BL}/outputs/bribri/test_clean_spanish.txt"
    [wixarika]="${BL}/outputs/wixarika/test_clean_spanish.txt"
    [maya]="${BL}/outputs/maya/test_clean_spanish.txt"
)

LANGUAGE=""
EXPERIMENT="all"
MODEL="gemini-2.5-flash"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --language)   LANGUAGE="$2"; shift 2 ;;
        --experiment) EXPERIMENT="$2"; shift 2 ;;
        --model)      MODEL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

[[ -z "$LANGUAGE" ]] && { echo "ERROR: --language required."; exit 1; }

LANG=$LANGUAGE
EXP_DIR="${BL}/dev_experiments/improvements_${LANG}_${TIMESTAMP}"
mkdir -p "${EXP_DIR}/preds" "${EXP_DIR}/metrics"

TSRC="${TRAIN_ES[$LANG]}"
TTGT="${TRAIN_TGT[$LANG]}"
DCAP="${DEV_CAP[$LANG]}"
DREF="${DEV_REF[$LANG]}"

for f in "$TSRC" "$TTGT" "$DCAP" "$DREF"; do
    [[ -f "$f" ]] || { echo "ERROR: missing: $f"; exit 1; }
done

translate() {
    local tag=$1; shift
    echo "  [${tag}]"
    python "${BL}/translate_llm_manyshot_gemini.py" \
        --language "${LANG}" \
        --train-src "${TSRC}" --train-tgt "${TTGT}" \
        --input-spanish "${DCAP}" --reference-file "${DREF}" \
        --output-preds    "${EXP_DIR}/preds/${tag}_preds.txt" \
        --metrics-json    "${EXP_DIR}/metrics/${tag}_metrics.json" \
        --per-example-tsv "${EXP_DIR}/metrics/${tag}_per_example.tsv" \
        --samples-tsv     "${EXP_DIR}/metrics/${tag}_samples.tsv" \
        --model "${MODEL}" --temperature 0.0 --sleep 0.15 "$@"
}

analyze() {
    local tag=$1
    echo "  [error analysis: ${tag}]"
    python "${BL}/error_analysis_report.py" \
        --input-spanish "${DCAP}" \
        --predictions   "${EXP_DIR}/preds/${tag}_preds.txt" \
        --references    "${DREF}" \
        --per-example-tsv "${EXP_DIR}/metrics/${tag}_per_example.tsv" \
        --output-dir    "${EXP_DIR}/error_analysis_${tag}" \
        --language      "${LANG}"
}

header() { echo; echo "══ ${LANG^^} — $1 ══"; }

# ══════════════════════════════════════════════════════════════
# NAHUATL  (16K train pairs)
# Previous: 25.8 chrF++, wrong_language 34/50
# Fix: unified script now uses NAH: prompt label
# ══════════════════════════════════════════════════════════════
exp_nahuatl_prompt_only() {
    header "prompt-only — verify NAH: label fixes wrong_language"
    translate "nahuatl_r80_d20" \
        --num-retrieval 80 --num-dev-examples 20 \
        --dev-example-spanish "${DCAP}" --dev-example-refs "${DREF}"
    analyze "nahuatl_r80_d20"
    echo
    echo "  REPORT BACK:"
    echo "    - mean chrF++ (was 25.8)"
    echo "    - wrong_language count in error_analysis_nahuatl_r80_d20/error_report.md (was 34/50)"
    echo "    - score bands: very_low / low / mid / high counts (was 12/36/2/0)"
}

exp_nahuatl_retrieval_sweep() {
    header "retrieval-sweep — r x d grid"
    for R in 20 40 80; do
        for D in 0 10 20; do
            translate "nahuatl_r${R}_d${D}" \
                --num-retrieval ${R} --num-dev-examples ${D} \
                --dev-example-spanish "${DCAP}" --dev-example-refs "${DREF}"
        done
    done
}

# ══════════════════════════════════════════════════════════════
# BRIBRI  (7.5K train pairs — smallest corpus)
# Previous: 11.2 chrF++, wrong_language 45/50
# Fix: unified script uses BZD: label. Small corpus → lower r.
# ══════════════════════════════════════════════════════════════
exp_bribri_prompt_only() {
    header "prompt-only — verify BZD: label fixes wrong_language"
    translate "bribri_r40_d20" \
        --num-retrieval 40 --num-dev-examples 20 \
        --dev-example-spanish "${DCAP}" --dev-example-refs "${DREF}"
    analyze "bribri_r40_d20"
    echo
    echo "  REPORT BACK:"
    echo "    - mean chrF++ (was 11.2)"
    echo "    - wrong_language count (was 45/50)"
    echo "    - score bands (was: very_low 48, low 2)"
}

exp_bribri_retrieval_sweep() {
    header "retrieval-sweep — test low r for small corpus"
    for R in 10 20 40 80; do
        for D in 0 10 20; do
            translate "bribri_r${R}_d${D}" \
                --num-retrieval ${R} --num-dev-examples ${D} \
                --dev-example-spanish "${DCAP}" --dev-example-refs "${DREF}"
        done
    done
}

exp_bribri_max_tokens() {
    header "max-tokens — shorter output"
    for MT in 60 80 100; do
        python "${BL}/translate_llm_manyshot_gemini.py" \
            --language "${LANG}" \
            --train-src "${TSRC}" --train-tgt "${TTGT}" \
            --input-spanish "${DCAP}" --reference-file "${DREF}" \
            --output-preds    "${EXP_DIR}/preds/bribri_r40_d20_mt${MT}_preds.txt" \
            --metrics-json    "${EXP_DIR}/metrics/bribri_r40_d20_mt${MT}_metrics.json" \
            --per-example-tsv "${EXP_DIR}/metrics/bribri_r40_d20_mt${MT}_per_example.tsv" \
            --samples-tsv     "${EXP_DIR}/metrics/bribri_r40_d20_mt${MT}_samples.tsv" \
            --model "${MODEL}" --num-retrieval 40 --num-dev-examples 20 \
            --max-tokens ${MT} --temperature 0.0 --sleep 0.15 \
            --dev-example-spanish "${DCAP}" --dev-example-refs "${DREF}"
    done
}

# ══════════════════════════════════════════════════════════════
# WIXARIKA  (9K train pairs, wixarika_with_dev already prepped)
# Previous: ~14 chrF++, wrong_language ~80%, repetition on Ex 1/16
# Fix: unified script uses HCH: label + dedup post-process
# ══════════════════════════════════════════════════════════════
exp_wixarika_prompt_only() {
    header "prompt-only — verify HCH: label + check repetition"
    translate "wixarika_r80_d20" \
        --num-retrieval 80 --num-dev-examples 20 \
        --dev-example-spanish "${DCAP}" --dev-example-refs "${DREF}"
    analyze "wixarika_r80_d20"
    echo
    echo "  REPORT BACK:"
    echo "    - mean chrF++ (was ~14)"
    echo "    - wrong_language count (was ~40/50)"
    echo "    - repetition_degeneration count (was 2 on Ex 1, 16)"
}

exp_wixarika_retrieval_sweep() {
    header "retrieval-sweep"
    for R in 20 40 80; do
        for D in 0 10 20; do
            translate "wixarika_r${R}_d${D}" \
                --num-retrieval ${R} --num-dev-examples ${D} \
                --dev-example-spanish "${DCAP}" --dev-example-refs "${DREF}"
        done
    done
}

exp_wixarika_dedup() {
    header "dedup — truncate repeated token spans in predictions"
    local SRC="${EXP_DIR}/preds/wixarika_r80_d20_preds.txt"
    local DEDUP="${EXP_DIR}/preds/wixarika_r80_d20_deduped_preds.txt"
    [[ -f "$SRC" ]] || { echo "  Run prompt-only first to generate predictions."; return; }
    python3 - "$SRC" "$DEDUP" << 'PYEOF'
import sys

def dedup(text, span=3):
    tokens = text.split()
    out = []
    i = 0
    while i < len(tokens):
        if i + span*2 <= len(tokens) and tokens[i:i+span] == tokens[i+span:i+span*2]:
            out.extend(tokens[i:i+span])
            break
        out.append(tokens[i])
        i += 1
    return " ".join(out)

src, dst = sys.argv[1], sys.argv[2]
lines = open(src, encoding="utf-8").readlines()
fixed = sum(1 for l in lines if dedup(l.strip()) != l.strip())
with open(dst, "w", encoding="utf-8") as f:
    for l in lines:
        f.write(dedup(l.strip()) + "\n")
print(f"  Fixed {fixed}/{len(lines)} predictions with repetition")
PYEOF
    python "${BL}/score_outputs.py" \
        --pred-file "${DEDUP}" --ref-file "${DREF}" \
        --metrics-json    "${EXP_DIR}/metrics/wixarika_r80_d20_deduped_metrics.json" \
        --per-example-tsv "${EXP_DIR}/metrics/wixarika_r80_d20_deduped_per_example.tsv"
}

# ══════════════════════════════════════════════════════════════
# MAYA  (no real train corpus — only dummy_train.es / dummy_train.maya)
# Strategy: r=0, rely entirely on dev few-shot examples.
# ══════════════════════════════════════════════════════════════
exp_maya_prompt_only() {
    header "prompt-only — pure dev few-shot (no real train corpus)"
    # With dummy train, BM25 retrieval has nothing useful.
    # Use only dev examples as in-context shots.
    for D in 10 20 25; do
        translate "maya_r0_d${D}" \
            --num-retrieval 0 --num-dev-examples ${D} \
            --dev-example-spanish "${DCAP}" --dev-example-refs "${DREF}"
    done
    analyze "maya_r0_d25"
    echo
    echo "  REPORT BACK:"
    echo "    - mean chrF++ for r0_d10, r0_d20, r0_d25"
    echo "    - wrong_language count (was ~80%)"
    echo "    - over_generation count (was present on Ex 23, 9 in foldA)"
    echo
    echo "  NOTE: If still very low after label fix, Maya may need a separate"
    echo "  train corpus. Check if maya_with_dev exists elsewhere on the cluster:"
    echo "  find ${BASE} -name 'train.maya' 2>/dev/null"
}

exp_maya_max_tokens() {
    header "max-tokens — cap output to reduce over-generation"
    for MT in 60 80; do
        python "${BL}/translate_llm_manyshot_gemini.py" \
            --language "${LANG}" \
            --train-src "${TSRC}" --train-tgt "${TTGT}" \
            --input-spanish "${DCAP}" --reference-file "${DREF}" \
            --output-preds    "${EXP_DIR}/preds/maya_r0_d20_mt${MT}_preds.txt" \
            --metrics-json    "${EXP_DIR}/metrics/maya_r0_d20_mt${MT}_metrics.json" \
            --per-example-tsv "${EXP_DIR}/metrics/maya_r0_d20_mt${MT}_per_example.tsv" \
            --samples-tsv     "${EXP_DIR}/metrics/maya_r0_d20_mt${MT}_samples.tsv" \
            --model "${MODEL}" --num-retrieval 0 --num-dev-examples 20 \
            --max-tokens ${MT} --temperature 0.0 --sleep 0.15 \
            --dev-example-spanish "${DCAP}" --dev-example-refs "${DREF}"
    done
}

# ══════════════════════════════════════════════════════════════
# GUARANÍ  (53K train pairs, frozen config r=80 d=49)
# ══════════════════════════════════════════════════════════════
exp_guarani_caption_filter() {
    header "caption-filter — identify vague captions for re-generation"
    python3 - "${DCAP}" "${EXP_DIR}/vague_report.json" << 'PYEOF'
import sys, json
MARKERS = [
    "parece", "probablemente", "podría ser", "tal vez", "quizá",
    "posiblemente", "como si fuera", "evoca", "simboliza",
    "hermosa escena", "se observa", "en la imagen",
]
lines = open(sys.argv[1], encoding="utf-8").readlines()
vague = [(i, l.strip()) for i, l in enumerate(lines)
         if any(m in l.lower() for m in MARKERS)]
report = {"total": len(lines), "vague_count": len(vague),
          "vague_pct": round(100*len(vague)/len(lines), 1),
          "vague_examples": vague[:10]}
json.dump(report, open(sys.argv[2], "w"), indent=2, ensure_ascii=False)
print(f"  Vague captions: {len(vague)}/{len(lines)} ({report['vague_pct']}%)")
for i, l in vague[:5]:
    print(f"  [{i:3d}] {l[:90]}")
PYEOF
    echo
    echo "  REPORT BACK: vague count and percentage"
    echo "  Full report: ${EXP_DIR}/vague_report.json"
}

exp_guarani_retrieval_sweep() {
    header "retrieval-sweep — around frozen r=80 d=49"
    for R in 60 80 100 120; do
        translate "guarani_r${R}_d49" \
            --num-retrieval ${R} --num-dev-examples 49 \
            --dev-example-spanish "${DCAP}" --dev-example-refs "${DREF}"
    done
}

# ══════════════════════════════════════════════════════════════
# Dispatch
# ══════════════════════════════════════════════════════════════
echo; echo "Language: ${LANG}  |  Experiment: ${EXPERIMENT}  |  ${TIMESTAMP}"
echo "Output dir: ${EXP_DIR}"

case "${LANG}_${EXPERIMENT}" in
    nahuatl_prompt-only)     exp_nahuatl_prompt_only ;;
    nahuatl_retrieval-sweep) exp_nahuatl_retrieval_sweep ;;
    nahuatl_all) exp_nahuatl_prompt_only; exp_nahuatl_retrieval_sweep ;;

    bribri_prompt-only)     exp_bribri_prompt_only ;;
    bribri_retrieval-sweep) exp_bribri_retrieval_sweep ;;
    bribri_max-tokens)      exp_bribri_max_tokens ;;
    bribri_all) exp_bribri_prompt_only; exp_bribri_max_tokens; exp_bribri_retrieval_sweep ;;

    wixarika_prompt-only)     exp_wixarika_prompt_only ;;
    wixarika_retrieval-sweep) exp_wixarika_retrieval_sweep ;;
    wixarika_dedup)           exp_wixarika_dedup ;;
    wixarika_all) exp_wixarika_prompt_only; exp_wixarika_dedup; exp_wixarika_retrieval_sweep ;;

    maya_prompt-only) exp_maya_prompt_only ;;
    maya_max-tokens)  exp_maya_max_tokens ;;
    maya_all) exp_maya_prompt_only; exp_maya_max_tokens ;;

    guarani_caption-filter)  exp_guarani_caption_filter ;;
    guarani_retrieval-sweep) exp_guarani_retrieval_sweep ;;
    guarani_all) exp_guarani_caption_filter; exp_guarani_retrieval_sweep ;;

    *)
        echo "ERROR: Unknown: --language ${LANG} --experiment ${EXPERIMENT}"
        echo "Valid experiments:"
        echo "  nahuatl  : prompt-only  retrieval-sweep  all"
        echo "  bribri   : prompt-only  retrieval-sweep  max-tokens  all"
        echo "  wixarika : prompt-only  retrieval-sweep  dedup  all"
        echo "  maya     : prompt-only  max-tokens  all"
        echo "  guarani  : caption-filter  retrieval-sweep  all"
        exit 1 ;;
esac

# ── Results summary ────────────────────────────────────────────
echo
echo "══════════════════════════════════════════════════════"
echo " RESULTS — ${LANG^^}"
echo "══════════════════════════════════════════════════════"
python3 - "${EXP_DIR}" << 'PYEOF'
import json, glob, os, sys
d = sys.argv[1]
results = []
for f in sorted(glob.glob(f"{d}/metrics/*_metrics.json")):
    try:
        data = json.load(open(f))
        s = float(data.get("chrfpp", data.get("chrf++", data.get("chrF++", 0))))
        tag = os.path.basename(f).replace("_metrics.json","")
        results.append((s, tag))
    except: pass
results.sort(reverse=True)
if not results:
    print("  No metrics files yet.")
else:
    print(f"  {'Config':<45} {'chrF++':>8}")
    print("  " + "-"*55)
    for sc, tag in results:
        print(f"  {tag:<45} {sc:>8.2f}")
    print(f"\n  Best: {results[0][1]}  ({results[0][0]:.2f})")
print(f"\n  All outputs: {d}/")
PYEOF