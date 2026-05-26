#!/bin/bash
#SBATCH --job-name=struct1-v2-sft
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

: "${NNGPT_STRUCT1_V2_SFT_RUN_ID:?set NNGPT_STRUCT1_V2_SFT_RUN_ID}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/home/s471802/nn-gpt:/home/s471802/nn-dataset:${PYTHONPATH:-}"
export NNGPT_TUNEBACKBONE_EVAL_CUDA_VISIBLE_DEVICES="${NNGPT_TUNEBACKBONE_EVAL_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
export NNGPT_NNEVAL_USE_ALL_VISIBLE_GPUS=1
export NNGPT_NNEVAL_WORKERS_PER_GPU="${NNGPT_NNEVAL_WORKERS_PER_GPU:-1}"

run_root="/home/s471802/nn-gpt/parallel_runs/${NNGPT_STRUCT1_V2_SFT_RUN_ID}"
epoch_root="${NNGPT_STRUCT1_V2_SFT_EPOCH_ROOT:-${run_root}/out/nngpt/llm/epoch_sft}"
mkdir -p "${run_root}/slurm" "${run_root}/out/nngpt/llm" /home/s471802/nn-gpt/slurm_logs

code_commit="$(git -C /home/s471802/nn-gpt rev-parse HEAD)"
dataset_commit="$(git -C /home/s471802/nn-dataset rev-parse HEAD)"
cat > "${run_root}/run_config.json" <<EOF
{
  "phase": "struct1_v2_sft_cycle",
  "run_id": "${NNGPT_STRUCT1_V2_SFT_RUN_ID}",
  "code_commit": "${code_commit}",
  "nn_dataset_commit": "${dataset_commit}",
  "sft_nn_prefixes": ["rl-bb-struct1", "rl-bb-struct1-v2"],
  "gen_nn_prefix": "rl-bb-struct1-v2-sftcycle",
  "llm_conf": "backbone_sft_config.json",
  "num_cycles": 20,
  "test_nn": 30,
  "nn_train_epochs": 1,
  "num_train_epochs": 1,
  "sft_max_length": 6144,
  "sft_batch_size": 2,
  "sft_gradient_accumulation": 4,
  "gpus": 4,
  "nneval_cuda_visible_devices": "${NNGPT_TUNEBACKBONE_EVAL_CUDA_VISIBLE_DEVICES}",
  "nneval_workers_per_gpu": ${NNGPT_NNEVAL_WORKERS_PER_GPU:-1},
  "epoch_root": "${epoch_root}",
  "out_nngpt_archive": "${NNGPT_STRUCT1_V2_OUT_NNGPT_ARCHIVE:-}"
}
EOF

python -m ab.gpt.TuneBackbone \
  --llm_conf backbone_sft_config.json \
  --sft_nn_prefixes rl-bb-struct1,rl-bb-struct1-v2 \
  --gen_nn_prefix rl-bb-struct1-v2-sftcycle \
  --num_cycles 20 \
  --test_nn 30 \
  --nn_train_epochs 1 \
  --num_train_epochs 1 \
  --sft_max_length 6144 \
  --sft_batch_size 2 \
  --sft_gradient_accumulation 4 \
  --epoch_root "${epoch_root}"
