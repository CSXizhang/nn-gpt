#!/bin/bash
#SBATCH --job-name=tunerl-reward-ablation
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --gres=gpu:4
#SBATCH --signal=TERM@300
#SBATCH --chdir=/home/s471802/nn-gpt
#SBATCH --output=/home/s471802/nn-gpt/slurm_logs/%x-%j.out
#SBATCH --error=/home/s471802/nn-gpt/slurm_logs/%x-%j.err

set -euo pipefail

if [ -z "${NNGPT_ABLATION_RUN_ID:-}" ]; then
  echo "NNGPT_ABLATION_RUN_ID is required" >&2
  exit 2
fi
if [ -z "${NNGPT_RL_REWARD_VARIANT:-}" ]; then
  echo "NNGPT_RL_REWARD_VARIANT is required" >&2
  exit 2
fi
if [ -z "${NNGPT_SFT_INIT_ADAPTER:-}" ]; then
  echo "NNGPT_SFT_INIT_ADAPTER is required" >&2
  exit 2
fi

case "${NNGPT_RL_REWARD_VARIANT}" in
  full|no_structural_novelty|strong_repeat_penalty) ;;
  *)
    echo "Unsupported NNGPT_RL_REWARD_VARIANT=${NNGPT_RL_REWARD_VARIANT}" >&2
    exit 2
    ;;
esac

mkdir -p /home/s471802/nn-gpt/slurm_logs
source /home/s471802/.venv/bin/activate
cd /home/s471802/nn-gpt

run_root="/home/s471802/nn-gpt/parallel_runs/${NNGPT_ABLATION_RUN_ID}"
mkdir -p "${run_root}/slurm"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/home/s471802/nn-gpt:/home/s471802/nn-dataset:${PYTHONPATH:-}"
if [ -z "${MASTER_PORT:-}" ]; then
  if [ -n "${SLURM_JOB_ID:-}" ]; then
    export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
  else
    export MASTER_PORT=$((20000 + $$ % 20000))
  fi
fi
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

export NNGPT_SFT_LOG_DIR="${run_root}/rl_output"
export NNGPT_SFT_MODEL_OUT="${run_root}/grpo_backbone_outputs"
export NNGPT_SFT_TRAINER_OUT="${run_root}/grpo_backbone_outputs_trainer"
export NNGPT_SFT_APPEND_LOGS=0

for dir in "${NNGPT_SFT_LOG_DIR}" "${NNGPT_SFT_MODEL_OUT}" "${NNGPT_SFT_TRAINER_OUT}"; do
  if [ -e "${dir}" ]; then
    echo "Output dir already exists: ${dir}" >&2
    exit 1
  fi
done

export NNGPT_RL_RESUME_STAGE="${NNGPT_RL_RESUME_STAGE:-stage2_formal_explore}"
export NNGPT_RL_FORMAL_REWARD_EPOCHS="${NNGPT_RL_FORMAL_REWARD_EPOCHS:-1}"
export NNGPT_SFT_LOAD_INITIAL_ADAPTER="${NNGPT_SFT_LOAD_INITIAL_ADAPTER:-1}"
export NNGPT_SFT_INITIAL_ADAPTER_MODE="${NNGPT_SFT_INITIAL_ADAPTER_MODE:-trainable}"
export NNGPT_SFT_RL_NN_PREFIXES="${NNGPT_SFT_RL_NN_PREFIXES:-rl-bb-struct1}"
export NNGPT_SFT_RL_PROMPT_MODE="${NNGPT_SFT_RL_PROMPT_MODE:-sft_aligned}"

export NNGPT_SFT_BASE_MODEL_ID="${NNGPT_SFT_BASE_MODEL_ID:-/home/s471802/nn-gpt/out/llm/deepseek-ai/deepseek-coder-6.7b-instruct}"
export NNGPT_SFT_TOKENIZER_ID="${NNGPT_SFT_TOKENIZER_ID:-deepseek-ai/deepseek-coder-6.7b-instruct}"
export NNGPT_SFT_NUM_GENERATIONS="${NNGPT_SFT_NUM_GENERATIONS:-4}"
export NNGPT_SFT_MAX_STEPS="${NNGPT_SFT_MAX_STEPS:-75}"
export NNGPT_SFT_MAX_PROMPT_LENGTH="${NNGPT_SFT_MAX_PROMPT_LENGTH:-3500}"
export NNGPT_SFT_MAX_COMPLETION_LENGTH="${NNGPT_SFT_MAX_COMPLETION_LENGTH:-1536}"
export NNGPT_SFT_DATASET_LIMIT="${NNGPT_SFT_DATASET_LIMIT:-500}"
export NNGPT_SFT_NUM_EPOCHS="${NNGPT_SFT_NUM_EPOCHS:-5}"
export NNGPT_SFT_SAVE_STEPS="${NNGPT_SFT_SAVE_STEPS:-25}"
export NNGPT_RL_STAGE1_EVAL_LIMIT_SECONDS="${NNGPT_RL_STAGE1_EVAL_LIMIT_SECONDS:-900}"

echo "Run id: ${NNGPT_ABLATION_RUN_ID}"
echo "Reward variant: ${NNGPT_RL_REWARD_VARIANT}"
echo "Run root: ${run_root}"
echo "Log dir: ${NNGPT_SFT_LOG_DIR}"
echo "Init adapter: ${NNGPT_SFT_INIT_ADAPTER}"
echo "NN prefixes: ${NNGPT_SFT_RL_NN_PREFIXES}"
echo "Formal reward epochs: ${NNGPT_RL_FORMAL_REWARD_EPOCHS}"
echo "Num generations: ${NNGPT_SFT_NUM_GENERATIONS}"
echo "Max steps: ${NNGPT_SFT_MAX_STEPS}"
echo "Master addr: ${MASTER_ADDR}"
echo "Master port: ${MASTER_PORT}"

exec python -m ab.gpt.TuneRLSft
