#!/bin/bash
#SBATCH --job-name=struct1-merge-ab
#SBATCH --partition=gpu_computervision
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=/home/s471802/nn-gpt/slurm_logs/struct1-merge-ab-%j.out
#SBATCH --error=/home/s471802/nn-gpt/slurm_logs/struct1-merge-ab-%j.err

set -euo pipefail

cd /home/s471802/nn-gpt
source /home/s471802/.venv/bin/activate

mkdir -p /home/s471802/nn-gpt/tmp_probes /home/s471802/nn-gpt/slurm_logs

export PYTHONPATH="/home/s471802/nn-gpt:/home/s471802/nn-dataset:${PYTHONPATH:-}"
export NNGPT_RL_RESUME_STAGE="${NNGPT_RL_RESUME_STAGE:-stage2_formal_explore}"
export NNGPT_RL_FORMAL_REWARD_EPOCHS="${NNGPT_RL_FORMAL_REWARD_EPOCHS:-1}"
export NNGPT_SFT_BASE_MODEL_ID="${NNGPT_SFT_BASE_MODEL_ID:-/home/s471802/nn-gpt/out/llm/deepseek-ai/deepseek-coder-6.7b-instruct}"
export NNGPT_SFT_TOKENIZER_ID="${NNGPT_SFT_TOKENIZER_ID:-/home/s471802/nn-gpt/parallel_runs/20260426_1905_main_resume_quality_diversity_std3/grpo_backbone_outputs_trainer/checkpoint-130}"
export NNGPT_SFT_INIT_ADAPTER="${NNGPT_SFT_INIT_ADAPTER:-/home/s471802/nn-gpt/out/nngpt/llm/epoch/A13/deepseek-ai/deepseek-coder-6.7b-instruct}"
export NNGPT_SFT_RL_NN_PREFIXES="${NNGPT_SFT_RL_NN_PREFIXES:-rl-bb-struct1}"
export NNGPT_SFT_RL_PROMPT_MODE="${NNGPT_SFT_RL_PROMPT_MODE:-sft_aligned}"
export NNGPT_SFT_MAX_PROMPT_LENGTH="${NNGPT_SFT_MAX_PROMPT_LENGTH:-3500}"
export NNGPT_SFT_MAX_COMPLETION_LENGTH="${NNGPT_SFT_MAX_COMPLETION_LENGTH:-1536}"
export NNGPT_SFT_MODE="${NNGPT_SFT_MODE:-split}"
export NNGPT_SFT_REWARD_EXCLUDE_TRAIN_GPU="${NNGPT_SFT_REWARD_EXCLUDE_TRAIN_GPU:-1}"
export NNGPT_SFT_REWARD_WORKERS="${NNGPT_SFT_REWARD_WORKERS:-1}"
export NNGPT_SFT_STOP_AFTER_FORWARD_XML="${NNGPT_SFT_STOP_AFTER_FORWARD_XML:-1}"

OUT_DIR="${NNGPT_PROBE_OUTPUT_DIR:-/home/s471802/nn-gpt/tmp_probes/struct1_merge_ab_${SLURM_JOB_ID}}"

exec python scripts/probe_struct1_merge_ab.py \
  --num-prompts "${NNGPT_PROBE_NUM_PROMPTS:-3}" \
  --samples-per-prompt "${NNGPT_PROBE_SAMPLES_PER_PROMPT:-2}" \
  --variants "${NNGPT_PROBE_VARIANTS:-unmerged_fp32,unmerged_bf16,merged_bf16}" \
  --output-dir "${OUT_DIR}"
