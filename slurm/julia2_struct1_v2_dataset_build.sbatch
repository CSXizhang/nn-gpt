#!/bin/bash
#SBATCH --job-name=struct1-v2-data
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

: "${NNGPT_STRUCT1_V2_RUN_ID:?set NNGPT_STRUCT1_V2_RUN_ID}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/home/s471802/nn-gpt:/home/s471802/nn-dataset:${PYTHONPATH:-}"
export NNGPT_NNEVAL_USE_ALL_VISIBLE_GPUS=1
export NNGPT_NNEVAL_WORKERS_PER_GPU="${NNGPT_NNEVAL_WORKERS_PER_GPU:-1}"

run_root="/home/s471802/nn-gpt/parallel_runs/${NNGPT_STRUCT1_V2_RUN_ID}"
synth_dir="/home/s471802/nn-gpt/out/nngpt/struct1_v2_dataset/${NNGPT_STRUCT1_V2_RUN_ID}/synth_nn"
mkdir -p "${run_root}/dataset" "${run_root}/slurm" /home/s471802/nn-gpt/slurm_logs

code_commit="$(git -C /home/s471802/nn-gpt rev-parse HEAD)"
dataset_commit="$(git -C /home/s471802/nn-dataset rev-parse HEAD)"
cat > "${run_root}/run_config.json" <<EOF
{
  "phase": "struct1_v2_dataset_build",
  "run_id": "${NNGPT_STRUCT1_V2_RUN_ID}",
  "code_commit": "${code_commit}",
  "nn_dataset_commit_before": "${dataset_commit}",
  "source_prefix": "rl-bb-struct1-v2",
  "old_prefix": "rl-bb-struct1",
  "raw_budget": ${NNGPT_STRUCT1_V2_RAW_BUDGET:-520},
  "target_new_samples": ${NNGPT_STRUCT1_V2_TARGET:-420},
  "per_pattern_target": ${NNGPT_STRUCT1_V2_PER_PATTERN_TARGET:-105},
  "formal_eval_epochs": 1,
  "transform": "norm_128_flip",
  "gpus": 4,
  "nneval_workers_per_gpu": ${NNGPT_NNEVAL_WORKERS_PER_GPU:-1},
  "cuda_visible_devices": "${CUDA_VISIBLE_DEVICES:-}",
  "synth_dir": "${synth_dir}"
}
EOF

python scripts/generate_struct1_v2_candidates.py \
  --output-synth-dir "${synth_dir}" \
  --budget "${NNGPT_STRUCT1_V2_RAW_BUDGET:-520}" \
  --seed "${NNGPT_STRUCT1_V2_SEED:-600}"

python -m ab.gpt.NNEval \
  --custom_synth_dir "${synth_dir}" \
  --nn_name_prefix rl-bb-struct1-v2 \
  --nn_alter_epochs 1 \
  --nn_train_epochs 1 \
  --transform norm_128_flip \
  --only_epoch 0 \
  --save_to_db \
  --cycle 0

python scripts/select_struct1_v2_sft_dataset.py \
  --source-root /home/s471802/nn-gpt \
  --repo-root /home/s471802/nn-dataset \
  --source-prefix rl-bb-struct1-v2 \
  --dest-prefix rl-bb-struct1-v2 \
  --target "${NNGPT_STRUCT1_V2_TARGET:-420}" \
  --per-pattern-target "${NNGPT_STRUCT1_V2_PER_PATTERN_TARGET:-105}" \
  --min-acc "${NNGPT_STRUCT1_V2_MIN_ACC:-0.70}" \
  --max-graph-repeats "${NNGPT_STRUCT1_V2_MAX_GRAPH_REPEATS:-2}" \
  --max-block-repeats "${NNGPT_STRUCT1_V2_MAX_BLOCK_REPEATS:-5}" \
  --max-backbone-repeats "${NNGPT_STRUCT1_V2_MAX_BACKBONE_REPEATS:-8}" \
  --required-transform norm_128_flip \
  --clean-dest-prefix \
  --report "${run_root}/dataset/selection_report.json"

python scripts/rebuild_nndataset_db.py \
  --repo-root /home/s471802/nn-dataset \
  --prefix rl-bb-struct1 \
  --prefix rl-bb-struct1-v2 \
  --report "${run_root}/dataset/db_rebuild_report.json"

echo "struct1-v2 dataset build complete: ${run_root}"
