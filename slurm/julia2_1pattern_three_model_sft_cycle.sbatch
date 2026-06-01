#!/bin/bash
#SBATCH --job-name=1pattern-sft
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --gres=gpu:4
#SBATCH --chdir=/home/s471802/nn-gpt
#SBATCH --output=/home/s471802/nn-gpt/slurm_logs/%x-%j.out
#SBATCH --error=/home/s471802/nn-gpt/slurm_logs/%x-%j.err

set -euo pipefail

source /home/s471802/.venv/bin/activate
cd /home/s471802/nn-gpt

: "${NNGPT_1PATTERN_SFT_RUN_ID:?set NNGPT_1PATTERN_SFT_RUN_ID}"
: "${NNGPT_1PATTERN_SFT_MODEL_LABEL:?set NNGPT_1PATTERN_SFT_MODEL_LABEL}"
: "${NNGPT_1PATTERN_SFT_LLM_CONF:?set NNGPT_1PATTERN_SFT_LLM_CONF}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/home/s471802/nn-gpt:/home/s471802/nn-dataset:${PYTHONPATH:-}"
export NNGPT_TUNEBACKBONE_EVAL_CUDA_VISIBLE_DEVICES="${NNGPT_TUNEBACKBONE_EVAL_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
export NNGPT_NNEVAL_USE_ALL_VISIBLE_GPUS=1
export NNGPT_NNEVAL_WORKERS_PER_GPU="${NNGPT_NNEVAL_WORKERS_PER_GPU:-1}"

run_root="/home/s471802/nn-gpt/parallel_runs/${NNGPT_1PATTERN_SFT_RUN_ID}"
epoch_root="${NNGPT_1PATTERN_SFT_EPOCH_ROOT:-/home/s471802/nn-gpt/out/nngpt/llm/${NNGPT_1PATTERN_SFT_RUN_ID}/epoch_sft}"
sft_nn_prefixes="${NNGPT_1PATTERN_SFT_NN_PREFIXES:-rl-bb-test1}"
gen_nn_prefix="${NNGPT_1PATTERN_SFT_GEN_NN_PREFIX:-rl-bb-test1-${NNGPT_1PATTERN_SFT_MODEL_LABEL}-sftcycle}"
num_cycles="${NNGPT_1PATTERN_SFT_NUM_CYCLES:-20}"
test_nn="${NNGPT_1PATTERN_SFT_TEST_NN:-30}"
nn_train_epochs="${NNGPT_1PATTERN_SFT_NN_TRAIN_EPOCHS:-1}"
num_train_epochs="${NNGPT_1PATTERN_SFT_NUM_TRAIN_EPOCHS:-1}"
sft_max_length="${NNGPT_1PATTERN_SFT_MAX_LENGTH:-6144}"
sft_batch_size="${NNGPT_1PATTERN_SFT_BATCH_SIZE:-2}"
sft_gradient_accumulation="${NNGPT_1PATTERN_SFT_GRADIENT_ACCUMULATION:-4}"

mkdir -p "${run_root}/slurm" "${run_root}/out" "$(dirname "${epoch_root}")" /home/s471802/nn-gpt/slurm_logs

code_commit="$(git -C /home/s471802/nn-gpt rev-parse HEAD)"
dataset_commit="$(git -C /home/s471802/nn-dataset rev-parse HEAD)"
cat > "${run_root}/run_config.json" <<EOF
{
  "phase": "1pattern_three_model_sft_cycle",
  "run_id": "${NNGPT_1PATTERN_SFT_RUN_ID}",
  "model_label": "${NNGPT_1PATTERN_SFT_MODEL_LABEL}",
  "code_commit": "${code_commit}",
  "nn_dataset_commit": "${dataset_commit}",
  "sft_nn_prefixes": ["${sft_nn_prefixes}"],
  "gen_nn_prefix": "${gen_nn_prefix}",
  "llm_conf": "${NNGPT_1PATTERN_SFT_LLM_CONF}",
  "base_model": "${NNGPT_1PATTERN_SFT_BASE_MODEL:-}",
  "num_cycles": ${num_cycles},
  "test_nn": ${test_nn},
  "nn_train_epochs": ${nn_train_epochs},
  "num_train_epochs": ${num_train_epochs},
  "sft_max_length": ${sft_max_length},
  "sft_batch_size": ${sft_batch_size},
  "sft_gradient_accumulation": ${sft_gradient_accumulation},
  "formal_eval_epochs": 1,
  "dataset": "cifar-10",
  "pattern": "1-pattern",
  "gpus": "${SLURM_GPUS_ON_NODE:-4}",
  "nneval_cuda_visible_devices": "${NNGPT_TUNEBACKBONE_EVAL_CUDA_VISIBLE_DEVICES}",
  "nneval_workers_per_gpu": ${NNGPT_NNEVAL_WORKERS_PER_GPU:-1},
  "epoch_root": "${epoch_root}"
}
EOF

python -m ab.gpt.TuneBackbone \
  --llm_conf "${NNGPT_1PATTERN_SFT_LLM_CONF}" \
  --sft_nn_prefixes "${sft_nn_prefixes}" \
  --gen_nn_prefix "${gen_nn_prefix}" \
  --num_cycles "${num_cycles}" \
  --test_nn "${test_nn}" \
  --nn_train_epochs "${nn_train_epochs}" \
  --num_train_epochs "${num_train_epochs}" \
  --sft_max_length "${sft_max_length}" \
  --sft_batch_size "${sft_batch_size}" \
  --sft_gradient_accumulation "${sft_gradient_accumulation}" \
  --epoch_root "${epoch_root}"
