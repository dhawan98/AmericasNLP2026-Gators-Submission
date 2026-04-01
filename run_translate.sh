#!/bin/bash
#SBATCH --job-name=anlp
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=96:00:00
#SBATCH --output=anlp_%j.out
#SBATCH --error=anlp_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aashish.dhawan@ufl.edu

module load cuda
export PYTHONNOUSERSITE=1

cd /blue/daisyw/aashish.dhawan/americasnlp2026
source .venv/bin/activate
cd baseline

echo "===== ENV CHECK ====="
which python
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))
PY
nvidia-smi
echo "====================="

python translate_mt.py train \
    --language guarani \
    --model-name facebook/nllb-200-distilled-600M \
    --train-src data/parallel/guarani_augmented/train.es \
    --train-tgt data/parallel/guarani_augmented/train.gn \
    --val-src data/parallel/guarani_augmented/val.es \
    --val-tgt data/parallel/guarani_augmented/val.gn \
    --output-dir runs/nllb_es_gn_v1_augmented_nols \
    --fp16 \
    --train-batch-size 2 \
    --eval-batch-size 8 \
    --gradient-accumulation-steps 16 \
    --num-train-epochs 10 \
    --eval-steps 500 \
    --save-steps 500 \
    --logging-steps 100 \
    --warmup-steps 400 \
    --learning-rate 3e-5 \
    --weight-decay 0.01 \
    --label-smoothing-factor 0.0 \
    --metric-for-best-model chrfpp \
    --early-stopping-patience 5 \
    --generation-num-beams 1 \
    --generation-max-length 128 \
    --save-total-limit 2 \
    --seed 42