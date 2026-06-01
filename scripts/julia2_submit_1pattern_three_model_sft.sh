#!/bin/bash
set -euo pipefail

run_group="${1:-$(date +%Y%m%d_%H%M)_1pattern_three_model_sft}"
partition="${NNGPT_1PATTERN_SFT_PARTITION:-h100}"
gpus="${NNGPT_1PATTERN_SFT_GPUS:-4}"

cd /home/s471802/nn-gpt

models=(
  "dscoder7b|backbone_sft_config.json|deepseek-ai/deepseek-coder-6.7b-instruct|1"
  "mistral7b|backbone_sft_mistral_7b_instruct_v03.json|mistralai/Mistral-7B-Instruct-v0.3|0"
  "qwen7b|backbone_sft_qwen2.5_coder_7b_instruct.json|Qwen/Qwen2.5-Coder-7B-Instruct|1"
)

printf 'run_group=%s\n' "${run_group}"

for spec in "${models[@]}"; do
  IFS='|' read -r label llm_conf base_model force_direct <<< "${spec}"
  run_id="${run_group}_${label}"
  run_root="/home/s471802/nn-gpt/parallel_runs/${run_id}"
  mkdir -p "${run_root}/slurm"

  job_id=$(sbatch --parsable -p "${partition}" --gres="gpu:${gpus}" \
    --job-name "1pat-${label}-sft" \
    --output "${run_root}/slurm/1pat-${label}-sft-%j.out" \
    --error "${run_root}/slurm/1pat-${label}-sft-%j.err" \
    --export "ALL,NNGPT_1PATTERN_SFT_RUN_ID=${run_id},NNGPT_1PATTERN_SFT_MODEL_LABEL=${label},NNGPT_1PATTERN_SFT_LLM_CONF=${llm_conf},NNGPT_1PATTERN_SFT_BASE_MODEL=${base_model},NNGPT_1PATTERN_SFT_GEN_NN_PREFIX=rl-bb-test1-${label}-sftcycle,NNGPT_1PATTERN_SFT_NN_PREFIXES=rl-bb-test1,NNGPT_1PATTERN_SFT_DATASET=cifar-10,NNGPT_1PATTERN_SFT_NUM_CYCLES=20,NNGPT_1PATTERN_SFT_TEST_NN=30,NNGPT_1PATTERN_SFT_NN_TRAIN_EPOCHS=1,NNGPT_1PATTERN_SFT_NUM_TRAIN_EPOCHS=1,NNGPT_FORCE_DIRECT_GENERATE=${force_direct},NNGPT_NNEVAL_WORKERS_PER_GPU=1" \
    slurm/julia2_1pattern_three_model_sft_cycle.sbatch)
  job_id="${job_id%%;*}"

  cat <<EOF
model=${label}
run_id=${run_id}
job_id=${job_id}
force_direct_generate=${force_direct}
run_root=${run_root}
epoch_root=/home/s471802/nn-gpt/out/nngpt/llm/${run_id}/epoch_sft
EOF
done
