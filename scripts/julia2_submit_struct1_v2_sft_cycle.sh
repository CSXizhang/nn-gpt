#!/bin/bash
set -euo pipefail

run_id="${1:-struct1_v2_sftcycle_$(date +%Y%m%d_%H%M)}"

cd /home/s471802/nn-gpt
mkdir -p "/home/s471802/nn-gpt/parallel_runs/${run_id}/slurm"

job_id=$(sbatch --parsable -p h100 --gres=gpu:4 \
  --job-name "struct1-v2-sft" \
  --output "/home/s471802/nn-gpt/parallel_runs/${run_id}/slurm/struct1-v2-sft-%j.out" \
  --error "/home/s471802/nn-gpt/parallel_runs/${run_id}/slurm/struct1-v2-sft-%j.err" \
  --export "ALL,NNGPT_STRUCT1_V2_SFT_RUN_ID=${run_id},NNGPT_NNEVAL_WORKERS_PER_GPU=1" \
  slurm/julia2_struct1_v2_sft_cycle.sbatch)
job_id="${job_id%%;*}"

cat <<EOF
run_id=${run_id}
job_id=${job_id}
run_root=/home/s471802/nn-gpt/parallel_runs/${run_id}
EOF
