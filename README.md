# Foraging PPO Assignment

Standalone foraging environment plus an event-driven autoregressive PPO agent.

## Install

Create an environment with your preferred Python tool, then install dependencies.

CPU / generic:

```bash
pip install -r requirements.txt
```

CUDA example:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install matplotlib tqdm
```

On the school Slurm server, the provided `train_l4.sh` creates a pip venv in
`/scratch/tmp/$USER/mas_foraging_venv`, writes Slurm logs to
`/scratch/tmp/$USER/mas_foraging_logs`, and saves training runs under
`runs/<run-name>` in this repository.

## Train

```bash
python -m ppo_assignment.train \
  --run-name ppo_5x5_5a_server \
  --runs-dir runs \
  --width 5 \
  --height 5 \
  --objects 10 \
  --agents 5 \
  --updates 1000 \
  --episodes-per-update 32 \
  --eval-every 25 \
  --eval-episodes 20 \
  --hidden-dim 128 \
  --ppo-epochs 2 \
  --lr 0.00025 \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --clip-eps 0.15 \
  --value-coef 0.5 \
  --entropy-coef 0.015 \
  --max-plan-steps 20 \
  --device cuda \
  --workers 8 \
  --worker-chunk-size 0 \
  --keep-best-checkpoints 5 \
  --strict-device
```

Outputs are saved under:

```text
runs/<run-name>/
```

## Evaluate

```bash
python -m ppo_assignment.evaluate \
  --checkpoint runs/ppo_5x5_5a_server/checkpoints/latest.pt \
  --width 5 \
  --height 5 \
  --objects 10 \
  --agents 5 \
  --hidden-dim 128 \
  --episodes 100 \
  --device cuda \
  --strict-device
```

## Visualize

```bash
python -m ppo_assignment.visualize \
  --checkpoint runs/ppo_5x5_5a_server/checkpoints/latest.pt \
  --width 5 \
  --height 5 \
  --objects 10 \
  --agents 5 \
  --hidden-dim 128 \
  --device cuda \
  --strict-device \
  --candidates 10
```
