#!/usr/bin/env bash
# run_wixarika_caption_experiment.sh
# Tests new caption prompts (v1, v2) against current best for Wixarika.
# The new prompts include a cultural glossary (hikuri, tsikuri, mara'akame, etc.)
# which should help the VLM ground culturally specific objects correctly.
#
# REQUIRES: GPU node with Qwen3-VL loaded
# Run from: /blue/daisyw/aashish.dhawan/americasnlp2026/baseline/
#
# Usage:
#   sbatch --partition=gpu --gpus=1 run_wixarika_caption_experiment.sh
#   # or on interactive GPU node:
#   bash run_wixarika_caption_experiment.sh

set -euo pipefail

BL="/blue/daisyw/aashish.dhawan/americasnlp2026/baseline"
MBART="/blue/daisyw/aashish.dhawan/mbart/data"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_DIR="${BL}/dev_experiments/wixarika_caption_exp_${TIMESTAMP}"
mkdir -p "${EXP_DIR}/captions" "${EXP_DIR}/preds" "${EXP_DIR}/metrics"

TASK_DEV="${BL}/americasnlp2026/data/dev/wixarika"
DREF="${BL}/outputs/wixarika/dev_references.txt"
TRAIN_ES="${BL}/data/parallel/wixarika_with_dev/train.es"
TRAIN_TGT="${BL}/data/parallel/wixarika_with_dev/train.hch"
CURRENT_BEST_CAP="${BL}/outputs/wixarika/dev_caption_es_v1.txt"

translate() {
    local tag=$1 caption_file=$2
    python "${BL}/translate_llm_manyshot_gemini.py" \
        --language wixarika \
        --train-src "${TRAIN_ES}" --train-tgt "${TRAIN_TGT}" \
        --input-spanish "${caption_file}" \
        --reference-file "${DREF}" \
        --output-preds    "${EXP_DIR}/preds/${tag}_preds.txt" \
        --metrics-json    "${EXP_DIR}/metrics/${tag}_metrics.json" \
        --per-example-tsv "${EXP_DIR}/metrics/${tag}_per_example.tsv" \
        --samples-tsv     "${EXP_DIR}/metrics/${tag}_samples.tsv" \
        --model gemini-2.5-flash \
        --num-retrieval 40 --num-dev-examples 20 \
        --dev-example-spanish "${caption_file}" \
        --dev-example-refs "${DREF}" \
        --temperature 0.0 --sleep 0.15
}

# ── Step 0: Baseline — current best captions + best config ────
echo "── Step 0: Baseline (current captions, r40 d20) ──"
translate "wixarika_v1cap_r40_d20" "${CURRENT_BEST_CAP}"

# ── Step 1: Re-caption with v2 prompt (compact, inline glossary) ──
# v2 is shorter — better for inference speed and less prompt leakage risk.
echo
echo "── Step 1: Re-caption with v2 prompt ──"
echo "NOTE: requires GPU. Skip this block if not on GPU node."

# Write the v2 prompt to a temp file
cat << 'PROMPTEOF' > /tmp/wixarika_caption_prompt_v2.txt
Escribe UNA sola leyenda breve en español para esta imagen (máximo 1 oración; usa 2 solo si es indispensable). Empieza con los sustantivos principales y evita frases como "En la imagen…" o "Se observa…". Describe solo lo claramente visible: personas, objetos, ropa, acción y entorno. Sé literal, concreta y específica; menciona colores, materiales y relaciones espaciales solo si ayudan a identificar el objeto.

Si aparece un elemento cultural wixárika claramente reconocible, usa el término correcto en español; puedes apoyarte en estas equivalencias para reconocerlo mejor: peyote/hikuri, ojo de Dios/tsikuri, nierika, xiriki, tuki, tukipa, mara'akame, rupurero, juayame, kamirra/kutuni. No menciones identidad wixárika, ritual, deidades, lugares sagrados ni significados simbólicos a menos que sean inequívocos en la imagen. Si no estás seguro del nombre exacto, descríbelo visualmente en español simple.
PROMPTEOF

if [[ -d "${TASK_DEV}" ]]; then
    python "${BL}/caption_images.py" \
        --input-jsonl "${TASK_DEV}/wixarika.jsonl" \
        --base-path   "${TASK_DEV}" \
        --language    wixarika \
        --model-name  Qwen/Qwen3-VL-8B-Instruct \
        --prompt-file /tmp/wixarika_caption_prompt_v2.txt \
        --output-jsonl "${EXP_DIR}/captions/wixarika_v2_captions.jsonl" \
        --output-txt   "${EXP_DIR}/captions/wixarika_v2_captions.txt" \
        --quantize 4bit
    echo "  Captions written: ${EXP_DIR}/captions/wixarika_v2_captions.txt"
    translate "wixarika_v2cap_r40_d20" "${EXP_DIR}/captions/wixarika_v2_captions.txt"
else
    echo "  Task dev dir not found at ${TASK_DEV} — skipping caption generation"
    echo "  Path to check: ls ${BL}/americasnlp2026/data/dev/"
fi

# ── Results ───────────────────────────────────────────────────
echo
echo "══════════════════════════════════════════════════════"
echo " WIXARIKA CAPTION EXPERIMENT RESULTS"
echo "══════════════════════════════════════════════════════"
python3 - "${EXP_DIR}" << 'PYEOF'
import json, glob, os, sys
d = sys.argv[1]
results = []
for f in sorted(glob.glob(f"{d}/metrics/*_metrics.json")):
    try:
        data = json.load(open(f))
        s = float(data.get("chrfpp", data.get("chrf++", 0)))
        tag = os.path.basename(f).replace("_metrics.json","")
        results.append((s, tag))
    except: pass
results.sort(reverse=True)
print(f"  {'Config':<45} {'chrF++':>8}")
print("  " + "-"*55)
for sc, tag in results:
    marker = " ← NEW BEST" if sc > 18.99 else ""
    print(f"  {tag:<45} {sc:>8.2f}{marker}")
if results:
    print(f"\n  Current best (r40 d20 v1 captions): 18.99")
    if results[0][0] > 18.99:
        print(f"  IMPROVEMENT: +{results[0][0]-18.99:.2f} from caption upgrade")
    else:
        print(f"  No improvement from caption upgrade")
print(f"\n  All outputs: {d}/")
PYEOF

echo
echo "REPORT BACK:"
echo "  - wixarika_v1cap_r40_d20 chrF++ (should match 18.99)"
echo "  - wixarika_v2cap_r40_d20 chrF++ (new captions)"
echo "  - If v2 captions improve by >0.5: use v2 captions for test submission"
echo "  - If not: proceed with test submission using current captions"