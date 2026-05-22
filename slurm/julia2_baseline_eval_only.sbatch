#!/bin/bash
#SBATCH --job-name=baseline-eval
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/s471802/nn-gpt
#SBATCH --output=/home/s471802/nn-gpt/slurm_logs/%x-%j.out
#SBATCH --error=/home/s471802/nn-gpt/slurm_logs/%x-%j.err

set -euo pipefail

mkdir -p /home/s471802/nn-gpt/slurm_logs
source /home/s471802/.venv/bin/activate
cd /home/s471802/nn-gpt

: "${NNGPT_BASELINE_RUN_ROOT:?set NNGPT_BASELINE_RUN_ROOT}"
if [ -z "${NNGPT_BASELINE_CANDIDATE_MANIFEST:-}" ] && [ -z "${NNGPT_BASELINE_CANDIDATE_FILES:-}" ]; then
  echo "set NNGPT_BASELINE_CANDIDATE_MANIFEST or NNGPT_BASELINE_CANDIDATE_FILES" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/home/s471802/nn-gpt:/home/s471802/nn-dataset:${PYTHONPATH:-}"
export NNGPT_RL_FORMAL_REWARD_EPOCHS="${NNGPT_RL_FORMAL_REWARD_EPOCHS:-1,5,10}"
export NNGPT_REWARD_WORKERS_PER_GPU="${NNGPT_REWARD_WORKERS_PER_GPU:-1}"

args=()
if [ -n "${NNGPT_BASELINE_CANDIDATE_MANIFEST:-}" ]; then
  while IFS= read -r path; do
    [ -n "${path}" ] || continue
    args+=(--candidate-file "${path}")
  done < "${NNGPT_BASELINE_CANDIDATE_MANIFEST}"
else
  IFS=',' read -r -a candidate_files <<< "${NNGPT_BASELINE_CANDIDATE_FILES}"
  for path in "${candidate_files[@]}"; do
    [ -n "${path}" ] || continue
    args+=(--candidate-file "${path}")
  done
fi

eval_dir="${NNGPT_BASELINE_EVAL_DIR:-${NNGPT_BASELINE_RUN_ROOT}/eval}"

exec python scripts/baseline_experiment_runner.py eval-only \
  "${args[@]}" \
  --output-dir "${eval_dir}" \
  --seed "${NNGPT_BASELINE_SEED:-42}" \
  --eval-concurrency "${NNGPT_BASELINE_EVAL_CONCURRENCY:-0}"
