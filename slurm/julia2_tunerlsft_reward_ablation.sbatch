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
if [ -z "${NNGPT_ABLATION_SOURCE_COMMIT:-}" ]; then
  echo "NNGPT_ABLATION_SOURCE_COMMIT is required" >&2
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
repo_root="/home/s471802/nn-gpt"
nn_dataset_root="${NN_DATASET_ROOT:-/home/s471802/nn-dataset}"
cd "${repo_root}"

run_root="/home/s471802/nn-gpt/parallel_runs/${NNGPT_ABLATION_RUN_ID}"
mkdir -p "${run_root}/slurm"
source_commit_hash="$(git rev-parse "${NNGPT_ABLATION_SOURCE_COMMIT}")"
source_commit_short="$(git rev-parse --short "${source_commit_hash}")"
source_dir="${run_root}/source_${source_commit_short}"

if [ -e "${source_dir}" ] && [ ! -d "${source_dir}/.git" ]; then
  echo "Source dir exists but is not a git checkout: ${source_dir}" >&2
  exit 1
fi
if [ ! -d "${source_dir}" ]; then
  git clone --shared "${repo_root}" "${source_dir}"
fi
git -C "${source_dir}" fetch --all --tags >/dev/null 2>&1 || true
git -C "${source_dir}" checkout --detach "${source_commit_hash}" >/dev/null
actual_source_head="$(git -C "${source_dir}" rev-parse HEAD)"
if [ "${actual_source_head}" != "${source_commit_hash}" ]; then
  echo "Wrong source HEAD: expected ${source_commit_hash}, got ${actual_source_head}" >&2
  exit 2
fi

if [ ! -e "${source_dir}/db" ]; then
  for shared_db_dir in "${nn_dataset_root}/db" "${repo_root}/db"; do
    if [ -f "${shared_db_dir}/ab.nn.db" ]; then
      ln -s "${shared_db_dir}" "${source_dir}/db"
      echo "Shared nn-dataset DB: ${shared_db_dir}/ab.nn.db"
      break
    fi
  done
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export MALLOC_TRIM_THRESHOLD_="${MALLOC_TRIM_THRESHOLD_:-65536}"
export PYTHONPATH="${source_dir}:/home/s471802/nn-dataset:${PYTHONPATH:-}"
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
export NNGPT_SFT_EPOCH_ROOT="${run_root}/out/nngpt/llm/epoch_sft"
export NNGPT_SFT_APPEND_LOGS=0
export NNGPT_REWARD_WORKERS_PER_GPU="${NNGPT_REWARD_WORKERS_PER_GPU:-1}"

for dir in "${NNGPT_SFT_LOG_DIR}" "${NNGPT_SFT_MODEL_OUT}" "${NNGPT_SFT_TRAINER_OUT}"; do
  if [ -e "${dir}" ]; then
    echo "Output dir already exists: ${dir}" >&2
    exit 1
  fi
done

export NNGPT_RL_RESUME_STAGE="${NNGPT_RL_RESUME_STAGE:-stage2_formal_explore}"
export NNGPT_RL_FORMAL_REWARD_EPOCHS="${NNGPT_RL_FORMAL_REWARD_EPOCHS:-10}"
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
echo "Source dir: ${source_dir}"
echo "Source commit expected: ${source_commit_hash}"
echo "Source commit actual: ${actual_source_head}"
echo "Log dir: ${NNGPT_SFT_LOG_DIR}"
echo "Init adapter: ${NNGPT_SFT_INIT_ADAPTER}"
echo "Base model: ${NNGPT_SFT_BASE_MODEL_ID}"
echo "Tokenizer: ${NNGPT_SFT_TOKENIZER_ID}"
echo "NN prefixes: ${NNGPT_SFT_RL_NN_PREFIXES}"
echo "Prompt mode: ${NNGPT_SFT_RL_PROMPT_MODE}"
echo "Formal reward epochs: ${NNGPT_RL_FORMAL_REWARD_EPOCHS}"
echo "Num generations: ${NNGPT_SFT_NUM_GENERATIONS}"
echo "Generation batch size: ${NNGPT_SFT_GENERATION_BATCH_SIZE:-}"
echo "Max steps: ${NNGPT_SFT_MAX_STEPS}"
echo "Max completion length: ${NNGPT_SFT_MAX_COMPLETION_LENGTH}"
echo "Reward exclude train GPU: ${NNGPT_SFT_REWARD_EXCLUDE_TRAIN_GPU:-}"
echo "Generation kwargs JSON: ${NNGPT_SFT_GENERATION_KWARGS_JSON:-}"
echo "Master addr: ${MASTER_ADDR}"
echo "Master port: ${MASTER_PORT}"

cd "${source_dir}"
exec python -m ab.gpt.TuneRLSft
