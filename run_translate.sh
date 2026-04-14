#!/bin/bash
#SBATCH --job-name=anlp
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=96:00:00
#SBATCH --output=anlp_%j.out
#SBATCH --error=anlp_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=driggersellis.cw@ufl.edu

module load mamba
mamba activate anlp-env

module load cuda
export PYTHONNOUSERSITE=1

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
	--model-name "facebook/mbart-large-50" \
	--train-src data/parallel/guarani/train.es \
	--train-tgt data/parallel/guarani/train.gn \
	--val-src data/parallel/guarani/val.es \
	--val-tgt data/parallel/guarani/val.gn \
	--output-dir runs/guarani_mbart50 \
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
	--save-total-limit 2  \
	--seed 42

mamba deactivate
