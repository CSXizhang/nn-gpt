#!/bin/bash
#SBATCH --job-name=baseline-brute-gen
#SBATCH --partition=small_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/s471802/nn-gpt
#SBATCH --output=/home/s471802/nn-gpt/slurm_logs/%x-%j.out
#SBATCH --error=/home/s471802/nn-gpt/slurm_logs/%x-%j.err

set -euo pipefail

mkdir -p /home/s471802/nn-gpt/slurm_logs
source /home/s471802/.venv/bin/activate
cd /home/s471802/nn-gpt

: "${NNGPT_BASELINE_RUN_ROOT:?set NNGPT_BASELINE_RUN_ROOT}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/s471802/nn-gpt:/home/s471802/nn-dataset:${PYTHONPATH:-}"

exec python scripts/baseline_experiment_runner.py brute-gen-only \
  --setting "${NNGPT_BASELINE_SETTING:-brute_constrained_random}" \
  --source-run "${NNGPT_BASELINE_SOURCE_RUN:-}" \
  --output-dir "${NNGPT_BASELINE_RUN_ROOT}/candidates/${NNGPT_BASELINE_SETTING:-brute_constrained_random}" \
  --budget "${NNGPT_BASELINE_BUDGET:-100}" \
  --seed "${NNGPT_BASELINE_SEED:-42}" \
  --pattern-set "${NNGPT_BASELINE_BRUTE_PATTERN_SET:-all}" \
  --seed-accuracy-baseline "${NNGPT_BASELINE_SEED_ACCURACY_BASELINE:-0.10}"
