#!/bin/bash
set -euo pipefail

run_id="${1:-struct1_v2_dataset_append_$(date +%Y%m%d_%H%M)}"
pattern_counts="${2:?pattern counts required, e.g. A_to_Fractal_plus_B:40,B_to_Fractal_plus_A:40}"

cd /home/s471802/nn-gpt
mkdir -p "/home/s471802/nn-gpt/parallel_runs/${run_id}/slurm"

job_id=$(sbatch --parsable -p h100 --gres=gpu:4 \
  --job-name "struct1-v2-append" \
  --output "/home/s471802/nn-gpt/parallel_runs/${run_id}/slurm/struct1-v2-append-%j.out" \
  --error "/home/s471802/nn-gpt/parallel_runs/${run_id}/slurm/struct1-v2-append-%j.err" \
  --export "ALL,NNGPT_STRUCT1_V2_APPEND_RUN_ID=${run_id},NNGPT_STRUCT1_V2_PATTERN_COUNTS=${pattern_counts},NNGPT_STRUCT1_V2_APPEND_RAW_BUDGET=80,NNGPT_STRUCT1_V2_TARGET=420,NNGPT_STRUCT1_V2_PER_PATTERN_TARGET=105,NNGPT_NNEVAL_WORKERS_PER_GPU=1" \
  slurm/julia2_struct1_v2_dataset_append.sbatch)
job_id="${job_id%%;*}"

cat <<EOF
run_id=${run_id}
job_id=${job_id}
run_root=/home/s471802/nn-gpt/parallel_runs/${run_id}
pattern_counts=${pattern_counts}
EOF
