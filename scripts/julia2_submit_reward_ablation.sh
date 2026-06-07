#!/bin/bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Submit four-pattern reward ablation runs on Julia2.

Options:
  --run-prefix ID          default: YYYYmmdd_HHMM_reward_ablation_821f
  --variants CSV          default: full
  --seeds CSV             default: 42
  --init-adapter PATH     default: selected A18 struct1-v2 DeepSeek SFT adapter
  --nn-prefixes CSV       default: rl-bb-struct1,rl-bb-struct1-v2
  --formal-dataset NAME   default: cifar-10
  --partition PART        default: h100
  --qos QOS               optional
  --gpus N                default: 4
  --mem MEM               default: 160G
  --cpus N                default: 32
  --max-steps N           default: 125
  --num-generations N     default: 8
  --generation-batch-size N
                           optional NNGPT_SFT_GENERATION_BATCH_SIZE override
  --max-completion-length N
                           default: 1536
  --lr VALUE              optional NNGPT_RL_LR override
  --kl-coef VALUE         optional NNGPT_RL_KL_COEF override
  --run-root-base PATH    default: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs
  --generation-kwargs-json JSON
                           optional NNGPT_SFT_GENERATION_KWARGS_JSON override
  --formal-reward-epochs E
                           default: 1
  --no-run-archive        do not append metadata to run_archive_index.md
USAGE
}

run_prefix="$(date +%Y%m%d_%H%M)_struct1_v2_a18_current_reward"
variants_csv="full"
seeds_csv="42"
init_adapter="/home/s471802/nn-gpt/out/nngpt/llm/epoch_sft_20260527_1410_struct1_v2_sftcycle_h100x4_home/A18/deepseek-ai/deepseek-coder-6.7b-instruct"
nn_prefixes="rl-bb-struct1,rl-bb-struct1-v2"
formal_dataset="cifar-10"
partition="h100"
qos=""
gpus="4"
mem="160G"
cpus="32"
max_steps="125"
num_generations="8"
generation_batch_size=""
max_completion_length="1536"
lr=""
kl_coef=""
run_root_base="/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs"
generation_kwargs_json=""
formal_reward_epochs="1"
write_archive="1"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --run-prefix) run_prefix="$2"; shift 2 ;;
    --variants) variants_csv="$2"; shift 2 ;;
    --seeds) seeds_csv="$2"; shift 2 ;;
    --init-adapter) init_adapter="$2"; shift 2 ;;
    --nn-prefixes) nn_prefixes="$2"; shift 2 ;;
    --formal-dataset) formal_dataset="$2"; shift 2 ;;
    --partition) partition="$2"; shift 2 ;;
    --qos) qos="$2"; shift 2 ;;
    --gpus) gpus="$2"; shift 2 ;;
    --mem) mem="$2"; shift 2 ;;
    --cpus) cpus="$2"; shift 2 ;;
    --max-steps) max_steps="$2"; shift 2 ;;
    --num-generations) num_generations="$2"; shift 2 ;;
    --generation-batch-size) generation_batch_size="$2"; shift 2 ;;
    --max-completion-length) max_completion_length="$2"; shift 2 ;;
    --lr) lr="$2"; shift 2 ;;
    --kl-coef) kl_coef="$2"; shift 2 ;;
    --run-root-base) run_root_base="$2"; shift 2 ;;
    --generation-kwargs-json) generation_kwargs_json="$2"; shift 2 ;;
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
IFS=',' read -r -a seeds <<< "${seeds_csv}"

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

  for seed in "${seeds[@]}"; do
    seed="$(echo "${seed}" | xargs)"
    if [ -z "${seed}" ]; then
      continue
    fi

    run_id="${run_prefix}_${variant}_seed${seed}"
    run_root="${run_root_base%/}/${run_id}"
    mkdir -p "${run_root}/slurm"

    sbatch_args=(
      --parsable
      -p "${partition}"
      --gres=gpu:"${gpus}"
      --cpus-per-task "${cpus}"
      --mem "${mem}"
      --job-name "abl-${variant}-s${seed}"
      --output "${run_root}/slurm/abl-${variant}-s${seed}-%j.out"
      --error "${run_root}/slurm/abl-${variant}-s${seed}-%j.err"
    )
    if [ -n "${qos}" ]; then
      sbatch_args+=(--qos "${qos}")
    fi

    export NNGPT_ABLATION_RUN_ID="${run_id}"
    export NNGPT_RUN_ROOT="${run_root}"
    export NNGPT_RL_REWARD_VARIANT="${variant}"
    export NNGPT_RL_SEED="${seed}"
    export NNGPT_SFT_INIT_ADAPTER="${init_adapter}"
    export NNGPT_RL_FORMAL_DATASET="${formal_dataset}"
    export NNGPT_RL_FORMAL_REWARD_EPOCHS="${formal_reward_epochs}"
    export NNGPT_SFT_NUM_GENERATIONS="${num_generations}"
    export NNGPT_SFT_MAX_COMPLETION_LENGTH="${max_completion_length}"
    if [ -n "${generation_batch_size}" ]; then
      export NNGPT_SFT_GENERATION_BATCH_SIZE="${generation_batch_size}"
    else
      unset NNGPT_SFT_GENERATION_BATCH_SIZE
    fi
    if [ -n "${lr}" ]; then
      export NNGPT_RL_LR="${lr}"
    else
      unset NNGPT_RL_LR
    fi
    if [ -n "${kl_coef}" ]; then
      export NNGPT_RL_KL_COEF="${kl_coef}"
    else
      unset NNGPT_RL_KL_COEF
    fi
    if [ -n "${generation_kwargs_json}" ]; then
      export NNGPT_SFT_GENERATION_KWARGS_JSON="${generation_kwargs_json}"
    else
      unset NNGPT_SFT_GENERATION_KWARGS_JSON
    fi
    export NNGPT_SFT_MAX_STEPS="${max_steps}"
    export NNGPT_SFT_LOAD_INITIAL_ADAPTER=1
    export NNGPT_SFT_INITIAL_ADAPTER_MODE=trainable
    export NNGPT_SFT_RL_NN_PREFIXES="${nn_prefixes}"
    export NNGPT_RL_RESUME_STAGE=stage2_formal_explore
    export NNGPT_REWARD_WORKERS_PER_GPU=1
    export NNGPT_SFT_RESUME_TRAINER_CHECKPOINT=""
    export NNGPT_SFT_RESUME_STAGE_CHECKPOINT=""

    job_id="$(sbatch "${sbatch_args[@]}" --export ALL slurm/julia2_tunerlsft_reward_ablation.sbatch)"
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
        echo "- seed: ${seed}"
        echo "- formal dataset: ${formal_dataset}"
        echo "- init adapter: ${init_adapter}"
        echo "- prompt/prefix: sft_aligned, ${nn_prefixes}"
        echo "- formal reward epochs: ${formal_reward_epochs}"
        echo "- max completion length: ${max_completion_length}"
        if [ -n "${generation_batch_size}" ]; then
          echo "- generation batch size: ${generation_batch_size}"
        fi
        if [ -n "${lr}" ]; then
          echo "- lr: ${lr}"
        fi
        if [ -n "${kl_coef}" ]; then
          echo "- kl coef: ${kl_coef}"
        fi
        if [ -n "${generation_kwargs_json}" ]; then
          echo "- generation kwargs json: ${generation_kwargs_json}"
        fi
        echo "- samples target: $((num_generations * max_steps)) (${num_generations} generations x ${max_steps} steps)"
        echo "- run root: ${run_root}"
        echo "- stdout/stderr: ${run_root}/slurm/abl-${variant}-s${seed}-${job_id}.out / .err"
        echo "- 训练结果: TODO"
        echo "- 主要缺陷: TODO"
        echo "- 分析: TODO"
      } >> run_archive_index.md
    fi

    echo "variant=${variant} seed=${seed} run_id=${run_id} job_id=${job_id} commit=${commit_short}"
  done
done
