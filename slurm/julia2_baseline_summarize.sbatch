#!/bin/bash
#SBATCH --job-name=baseline-summarize
#SBATCH --partition=small_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --chdir=/home/s471802/nn-gpt
#SBATCH --output=/home/s471802/nn-gpt/slurm_logs/%x-%j.out
#SBATCH --error=/home/s471802/nn-gpt/slurm_logs/%x-%j.err

set -euo pipefail

mkdir -p /home/s471802/nn-gpt/slurm_logs
source /home/s471802/.venv/bin/activate
cd /home/s471802/nn-gpt

: "${NNGPT_BASELINE_RUN_ROOT:?set NNGPT_BASELINE_RUN_ROOT}"
export PYTHONPATH="/home/s471802/nn-gpt:/home/s471802/nn-dataset:${PYTHONPATH:-}"
eval_dir="${NNGPT_BASELINE_EVAL_DIR:-${NNGPT_BASELINE_RUN_ROOT}/eval}"

exec python scripts/baseline_experiment_runner.py summarize \
  --input "eval=${eval_dir}/generation_samples.jsonl" \
  --output-dir "${eval_dir}" \
  --seed "${NNGPT_BASELINE_SEED:-42}"
