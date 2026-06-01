#!/bin/bash
#SBATCH --job-name=a3-direct-probe
#SBATCH --partition=gpu_computervision
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=/home/s471802/nn-gpt/slurm_logs/a3-direct-probe-%j.out
#SBATCH --error=/home/s471802/nn-gpt/slurm_logs/a3-direct-probe-%j.err

set -euo pipefail

cd /home/s471802/nn-gpt
source /home/s471802/.venv/bin/activate

mkdir -p /home/s471802/nn-gpt/tmp_probes /home/s471802/nn-gpt/slurm_logs

export PYTHONPATH="/home/s471802/nn-gpt:/home/s471802/nn-dataset:${PYTHONPATH:-}"
export NNGPT_RL_RESUME_STAGE="${NNGPT_RL_RESUME_STAGE:-stage1_structure_explore}"
export NNGPT_SFT_BASE_MODEL_ID="${NNGPT_SFT_BASE_MODEL_ID:-/home/s471802/nn-gpt/out/llm/deepseek-ai/deepseek-coder-6.7b-instruct}"
export NNGPT_SFT_TOKENIZER_ID="${NNGPT_SFT_TOKENIZER_ID:-/home/s471802/nn-gpt/parallel_runs/20260426_1905_main_resume_quality_diversity_std3/grpo_backbone_outputs_trainer/checkpoint-130}"
export NNGPT_SFT_INIT_ADAPTER="${NNGPT_SFT_INIT_ADAPTER:-/home/s471802/nn-gpt/out/nngpt/llm/epoch/A3/deepseek-ai/deepseek-coder-6.7b-instruct}"
export NNGPT_SFT_LOAD_INITIAL_ADAPTER="${NNGPT_SFT_LOAD_INITIAL_ADAPTER:-1}"
export NNGPT_SFT_INITIAL_ADAPTER_MODE="${NNGPT_SFT_INITIAL_ADAPTER_MODE:-trainable}"
export NNGPT_SFT_INITIAL_ADAPTER_DTYPE="${NNGPT_SFT_INITIAL_ADAPTER_DTYPE:-fp32}"
export NNGPT_SFT_RL_NN_PREFIXES="${NNGPT_SFT_RL_NN_PREFIXES:-rl-bb-struct1}"
export NNGPT_SFT_RL_PROMPT_MODE="${NNGPT_SFT_RL_PROMPT_MODE:-sft_aligned}"
export NNGPT_SFT_MAX_PROMPT_LENGTH="${NNGPT_SFT_MAX_PROMPT_LENGTH:-3500}"
export NNGPT_SFT_MAX_COMPLETION_LENGTH="${NNGPT_SFT_MAX_COMPLETION_LENGTH:-1536}"
export NNGPT_SFT_MODE="${NNGPT_SFT_MODE:-split}"
export NNGPT_SFT_REWARD_EXCLUDE_TRAIN_GPU="${NNGPT_SFT_REWARD_EXCLUDE_TRAIN_GPU:-1}"
export NNGPT_SFT_LOG_DIR="${NNGPT_SFT_LOG_DIR:-/home/s471802/nn-gpt/tmp_probes/a3_direct_probe_log_${SLURM_JOB_ID}}"

OUT="${NNGPT_PROBE_OUTPUT:-/home/s471802/nn-gpt/tmp_probes/a3_direct_probe_${SLURM_JOB_ID}.jsonl}"

exec python scripts/probe_struct1_a3_reward.py \
  --num-prompts "${NNGPT_PROBE_NUM_PROMPTS:-4}" \
  --samples-per-prompt "${NNGPT_PROBE_SAMPLES_PER_PROMPT:-4}" \
  --output "${OUT}"
