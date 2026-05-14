#!/bin/bash
#SBATCH --job-name=foraging-ppo-8pt
#SBATCH --partition=gpu-ffa
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=48G
#SBATCH --output=/home/kaiserd/mas_foraging/logs/foraging_ppo_%j.out
#SBATCH --error=/home/kaiserd/mas_foraging/logs/foraging_ppo_%j.err
#SBATCH --open-mode=append

set -euo pipefail

SCRATCH_BASE=/scratch/tmp/$USER
VENV_DIR="$SCRATCH_BASE/mas_foraging_venv"

HOME_BASE=/home/$USER/mas_foraging
LOG_DIR="$HOME_BASE/mas_foraging_logs"

cd $HOME_BASE

mkdir -p logs runs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
  source "$VENV_DIR/bin/activate"
  python -m pip install --upgrade pip
  python -m pip install torch --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
  python -m pip install matplotlib tqdm
else
  source "$VENV_DIR/bin/activate"
fi

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY

python -m ppo_assignment.train \
  --run-name ppo_8pt_5x5_2a \
  --runs-dir runs \
  --updates 1000 \
  --episodes-per-update 256 \
  --eval-every 25 --eval-episodes 20 \
  --hidden-dim 128 --ppo-epochs 2 --ppo-minibatch-size 512 --ppo-token-budget 8192 \
  --lr 0.00025 --gamma 0.99 --gae-lambda 0.95 \
  --clip-eps 0.15 --value-coef 0.5 --entropy-coef 0.015 \
  --max-plan-steps 10 \
  --device cuda \
  --workers 64 --worker-chunk-size 0 --worker-torch-threads 1 \
  --worker-transport plain \
  --keep-best-checkpoints 5 \
  --strict-device \
  --eval-seed 12345 \
  --min-height 5 --max-height-train 5 \
  --min-width 5 --max-width-train 5 \
  --min-agents 2 --max-agents-train 2 \
  --min-objects 3 --max-objects-train 23 \
  --min-occupancy 0.20 --max-occupancy 1.0 \
  --min-objects-per-agent 1.5 --max-objects-per-agent 11.0
