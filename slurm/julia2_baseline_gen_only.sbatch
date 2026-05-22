#!/bin/bash
#SBATCH --job-name=baseline-gen
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --chdir=/home/s471802/nn-gpt
#SBATCH --output=/home/s471802/nn-gpt/slurm_logs/%x-%j.out
#SBATCH --error=/home/s471802/nn-gpt/slurm_logs/%x-%j.err

set -euo pipefail

mkdir -p /home/s471802/nn-gpt/slurm_logs
source /home/s471802/.venv/bin/activate
cd /home/s471802/nn-gpt

: "${NNGPT_BASELINE_RUN_ROOT:?set NNGPT_BASELINE_RUN_ROOT}"
: "${NNGPT_BASELINE_SETTING:?set NNGPT_BASELINE_SETTING}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NNGPT_RL_FORMAL_REWARD_EPOCHS="${NNGPT_RL_FORMAL_REWARD_EPOCHS:-1,5,10}"
export NNGPT_SFT_BASE_MODEL_ID="${NNGPT_SFT_BASE_MODEL_ID:-/home/s471802/nn-gpt/out/llm/deepseek-ai/deepseek-coder-6.7b-instruct}"
export NNGPT_SFT_TOKENIZER_ID="${NNGPT_SFT_TOKENIZER_ID:-deepseek-ai/deepseek-coder-6.7b-instruct}"

exec python scripts/baseline_experiment_runner.py gen-only \
  --setting "${NNGPT_BASELINE_SETTING}" \
  --source-run "${NNGPT_BASELINE_SOURCE_RUN:-}" \
  --output-dir "${NNGPT_BASELINE_RUN_ROOT}/candidates/${NNGPT_BASELINE_SETTING}" \
  --budget "${NNGPT_BASELINE_BUDGET:-100}" \
  --seed "${NNGPT_BASELINE_SEED:-42}" \
  --adapter-path "${NNGPT_BASELINE_ADAPTER_PATH:-}" \
  --adapter-mode "${NNGPT_BASELINE_ADAPTER_MODE:-trainable}" \
  --nn-prefixes "${NNGPT_SFT_RL_NN_PREFIXES:-rl-bb-test1}" \
  --prompt-mode "${NNGPT_SFT_RL_PROMPT_MODE:-sft_aligned}" \
  --feedback-char-budget "${NNGPT_SFT_FEEDBACK_CHAR_BUDGET:-0}" \
  --temperature "${NNGPT_SFT_TEMPERATURE:-0.8}" \
  --top-p "${NNGPT_SFT_TOP_P:-0.95}" \
  --top-k "${NNGPT_SFT_TOP_K:-50}" \
  --max-new-tokens "${NNGPT_SFT_MAX_COMPLETION_LENGTH:-1536}"
