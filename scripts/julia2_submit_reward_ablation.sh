#!/bin/bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Submit four-pattern reward ablation runs on Julia2.

Options:
  --run-prefix ID          default: YYYYmmdd_HHMM_reward_ablation_821f
  --variants CSV          default: full,no_structural_novelty,strong_repeat_penalty
  --init-adapter PATH     default: /home/s471802/nn-gpt/out/nngpt/llm/epoch/A3/deepseek-ai/deepseek-coder-6.7b-instruct
  --partition PART        default: h100
  --qos QOS               optional
  --gpus N                default: 4
  --mem MEM               default: 80G
  --cpus N                default: 16
  --max-steps N           default: 75
  --num-generations N     default: 4
  --formal-reward-epochs E
                           default: 10
  --no-run-archive        do not append metadata to run_archive_index.md
USAGE
}

run_prefix="$(date +%Y%m%d_%H%M)_reward_ablation_821f"
variants_csv="full,no_structural_novelty,strong_repeat_penalty"
init_adapter="/home/s471802/nn-gpt/out/nngpt/llm/epoch/A3/deepseek-ai/deepseek-coder-6.7b-instruct"
partition="h100"
qos=""
gpus="4"
mem="80G"
cpus="16"
max_steps="75"
num_generations="4"
formal_reward_epochs="10"
write_archive="1"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --run-prefix) run_prefix="$2"; shift 2 ;;
    --variants) variants_csv="$2"; shift 2 ;;
    --init-adapter) init_adapter="$2"; shift 2 ;;
    --partition) partition="$2"; shift 2 ;;
    --qos) qos="$2"; shift 2 ;;
    --gpus) gpus="$2"; shift 2 ;;
    --mem) mem="$2"; shift 2 ;;
    --cpus) cpus="$2"; shift 2 ;;
    --max-steps) max_steps="$2"; shift 2 ;;
    --num-generations) num_generations="$2"; shift 2 ;;
    --formal-reward-epochs) formal_reward_epochs="$2"; shift 2 ;;
    --no-run-archive) write_archive="0"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

cd /home/s471802/nn-gpt

commit_hash="$(git rev-parse HEAD)"
commit_short="$(git rev-parse --short HEAD)"
commit_subject="$(git log -1 --pretty=%s)"

IFS=',' read -r -a variants <<< "${variants_csv}"

for variant in "${variants[@]}"; do
  variant="$(echo "${variant}" | xargs)"
  case "${variant}" in
    full|no_structural_novelty|strong_repeat_penalty) ;;
    "")
      continue
      ;;
    *)
      echo "Unsupported variant: ${variant}" >&2
      exit 2
      ;;
  esac

  run_id="${run_prefix}_${variant}"
  run_root="/home/s471802/nn-gpt/parallel_runs/${run_id}"
  mkdir -p "${run_root}/slurm"

  sbatch_args=(
    --parsable
    -p "${partition}"
    --gres=gpu:"${gpus}"
    --cpus-per-task "${cpus}"
    --mem "${mem}"
    --job-name "abl-${variant}"
    --output "${run_root}/slurm/abl-${variant}-%j.out"
    --error "${run_root}/slurm/abl-${variant}-%j.err"
  )
  if [ -n "${qos}" ]; then
    sbatch_args+=(--qos "${qos}")
  fi

  export_vars="ALL"
  export_vars+=",NNGPT_ABLATION_RUN_ID=${run_id}"
  export_vars+=",NNGPT_RL_REWARD_VARIANT=${variant}"
  export_vars+=",NNGPT_SFT_INIT_ADAPTER=${init_adapter}"
  export_vars+=",NNGPT_RL_FORMAL_REWARD_EPOCHS=${formal_reward_epochs}"
  export_vars+=",NNGPT_SFT_NUM_GENERATIONS=${num_generations}"
  export_vars+=",NNGPT_SFT_MAX_STEPS=${max_steps}"
  export_vars+=",NNGPT_SFT_LOAD_INITIAL_ADAPTER=1"
  export_vars+=",NNGPT_SFT_INITIAL_ADAPTER_MODE=trainable"
  export_vars+=",NNGPT_SFT_RL_NN_PREFIXES=rl-bb-struct1"
  export_vars+=",NNGPT_RL_RESUME_STAGE=stage2_formal_explore"
  export_vars+=",NNGPT_REWARD_WORKERS_PER_GPU=1"
  export_vars+=",NNGPT_SFT_RESUME_TRAINER_CHECKPOINT="
  export_vars+=",NNGPT_SFT_RESUME_STAGE_CHECKPOINT="

  job_id="$(sbatch "${sbatch_args[@]}" --export "${export_vars}" slurm/julia2_tunerlsft_reward_ablation.sbatch)"
  job_id="${job_id%%;*}"

  if [ "${write_archive}" = "1" ]; then
    {
      echo
      echo "## ${run_id}"
      echo "- 提交时间: $(date +%Y-%m-%dT%H:%M:%S%z)"
      echo "- 状态: submitted"
      echo "- main job: ${job_id}"
      echo "- 分区/GPU: ${partition}, ${gpus} GPU"
      echo "- commit: ${commit_hash} (${commit_subject})"
      echo "- reward variant: ${variant}"
      echo "- init adapter: ${init_adapter}"
      echo "- prompt/prefix: sft_aligned, rl-bb-struct1"
      echo "- formal reward epochs: ${formal_reward_epochs}"
      echo "- samples target: $((num_generations * max_steps)) (${num_generations} generations x ${max_steps} steps)"
      echo "- run root: ${run_root}"
      echo "- stdout/stderr: ${run_root}/slurm/abl-${variant}-${job_id}.out / .err"
      echo "- 训练结果: TODO"
      echo "- 主要缺陷: TODO"
      echo "- 分析: TODO"
    } >> run_archive_index.md
  fi

  echo "variant=${variant} run_id=${run_id} job_id=${job_id} commit=${commit_short}"
done
