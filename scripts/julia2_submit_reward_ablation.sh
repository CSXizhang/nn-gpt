#!/bin/bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Submit four-pattern reward ablation runs on Julia2.

Options:
  --run-prefix ID          default: YYYYmmdd_HHMM_a3_reward_ablation_cleanhead
  --variants CSV          default: full,no_structural_novelty,strong_repeat_penalty
  --source-commit COMMIT  code commit to clone into every run; default: current HEAD
  --init-adapter PATH     default: archived A3 adapter from before struct1 v2 shared output
  --base-model-id PATH    default: /home/s471802/nn-gpt/out/llm/deepseek-ai/deepseek-coder-6.7b-instruct
  --tokenizer-id PATH     default: paper A3 tokenizer checkpoint-130
  --nn-prefixes CSV       default: rl-bb-struct1
  --prompt-mode MODE      default: sft_aligned
  --partition PART        default: h100
  --qos QOS               optional
  --gpus N                default: 4
  --mem MEM               default: 80G
  --cpus N                default: 16
  --max-steps N           default: 75
  --num-generations N     default: 4
  --generation-batch-size N
                           optional NNGPT_SFT_GENERATION_BATCH_SIZE override
  --max-completion-length N
                           default: 1536
  --generation-kwargs-json JSON
                           optional NNGPT_SFT_GENERATION_KWARGS_JSON override
  --exclude-train-gpu     set NNGPT_SFT_REWARD_EXCLUDE_TRAIN_GPU=1
  --formal-reward-epochs E
                           default: 10
  --no-run-archive        do not append metadata to run_archive_index.md
USAGE
}

run_prefix="$(date +%Y%m%d_%H%M)_a3_reward_ablation_cleanhead"
variants_csv="full,no_structural_novelty,strong_repeat_penalty"
source_commit=""
init_adapter="/home/s471802/nn-gpt/out/nngpt_archive_20260526_105845_before_struct1_v2_sftcycle_sharedout/llm/epoch/A3/deepseek-ai/deepseek-coder-6.7b-instruct"
base_model_id="/home/s471802/nn-gpt/out/llm/deepseek-ai/deepseek-coder-6.7b-instruct"
tokenizer_id="/home/s471802/nn-gpt/parallel_runs/20260426_1905_main_resume_quality_diversity_std3/grpo_backbone_outputs_trainer/checkpoint-130"
nn_prefixes="rl-bb-struct1"
prompt_mode="sft_aligned"
partition="h100"
qos=""
gpus="4"
mem="80G"
cpus="16"
max_steps="75"
num_generations="4"
generation_batch_size=""
max_completion_length="1536"
generation_kwargs_json=""
exclude_train_gpu="0"
formal_reward_epochs="10"
write_archive="1"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --run-prefix) run_prefix="$2"; shift 2 ;;
    --variants) variants_csv="$2"; shift 2 ;;
    --source-commit) source_commit="$2"; shift 2 ;;
    --init-adapter) init_adapter="$2"; shift 2 ;;
    --base-model-id) base_model_id="$2"; shift 2 ;;
    --tokenizer-id) tokenizer_id="$2"; shift 2 ;;
    --nn-prefixes) nn_prefixes="$2"; shift 2 ;;
    --prompt-mode) prompt_mode="$2"; shift 2 ;;
    --partition) partition="$2"; shift 2 ;;
    --qos) qos="$2"; shift 2 ;;
    --gpus) gpus="$2"; shift 2 ;;
    --mem) mem="$2"; shift 2 ;;
    --cpus) cpus="$2"; shift 2 ;;
    --max-steps) max_steps="$2"; shift 2 ;;
    --num-generations) num_generations="$2"; shift 2 ;;
    --generation-batch-size) generation_batch_size="$2"; shift 2 ;;
    --max-completion-length) max_completion_length="$2"; shift 2 ;;
    --generation-kwargs-json) generation_kwargs_json="$2"; shift 2 ;;
    --exclude-train-gpu) exclude_train_gpu="1"; shift ;;
    --formal-reward-epochs) formal_reward_epochs="$2"; shift 2 ;;
    --no-run-archive) write_archive="0"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

cd /home/s471802/nn-gpt

submit_commit_hash="$(git rev-parse HEAD)"
submit_commit_short="$(git rev-parse --short HEAD)"
submit_commit_subject="$(git log -1 --pretty=%s)"
if [ -z "${source_commit}" ]; then
  source_commit="${submit_commit_hash}"
fi
source_commit_hash="$(git rev-parse "${source_commit}")"
source_commit_short="$(git rev-parse --short "${source_commit_hash}")"
source_commit_subject="$(git log -1 --pretty=%s "${source_commit_hash}")"

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
  export_vars+=",NNGPT_ABLATION_SOURCE_COMMIT=${source_commit_hash}"
  export_vars+=",NNGPT_RL_REWARD_VARIANT=${variant}"
  export_vars+=",NNGPT_SFT_INIT_ADAPTER=${init_adapter}"
  export_vars+=",NNGPT_RL_FORMAL_REWARD_EPOCHS=${formal_reward_epochs}"
  export_vars+=",NNGPT_SFT_NUM_GENERATIONS=${num_generations}"
  if [ -n "${generation_batch_size}" ]; then
    export_vars+=",NNGPT_SFT_GENERATION_BATCH_SIZE=${generation_batch_size}"
  fi
  export_vars+=",NNGPT_SFT_MAX_COMPLETION_LENGTH=${max_completion_length}"
  if [ -n "${generation_kwargs_json}" ]; then
    export_vars+=",NNGPT_SFT_GENERATION_KWARGS_JSON=${generation_kwargs_json}"
  fi
  if [ "${exclude_train_gpu}" = "1" ]; then
    export_vars+=",NNGPT_SFT_REWARD_EXCLUDE_TRAIN_GPU=1"
  fi
  export_vars+=",NNGPT_SFT_MAX_STEPS=${max_steps}"
  export_vars+=",NNGPT_SFT_LOAD_INITIAL_ADAPTER=1"
  export_vars+=",NNGPT_SFT_INITIAL_ADAPTER_MODE=trainable"
  export_vars+=",NNGPT_SFT_RL_NN_PREFIXES=${nn_prefixes}"
  export_vars+=",NNGPT_SFT_RL_PROMPT_MODE=${prompt_mode}"
  export_vars+=",NNGPT_RL_RESUME_STAGE=stage2_formal_explore"
  export_vars+=",NNGPT_REWARD_WORKERS_PER_GPU=1"
  export_vars+=",NNGPT_SFT_RESUME_TRAINER_CHECKPOINT="
  export_vars+=",NNGPT_SFT_RESUME_STAGE_CHECKPOINT="
  export_vars+=",NNGPT_SFT_BASE_MODEL_ID=${base_model_id}"
  export_vars+=",NNGPT_SFT_TOKENIZER_ID=${tokenizer_id}"

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
      echo "- submitter commit: ${submit_commit_hash} (${submit_commit_subject})"
      echo "- source commit: ${source_commit_hash} (${source_commit_subject})"
      echo "- reward variant: ${variant}"
      echo "- init adapter: ${init_adapter}"
      echo "- base model: ${base_model_id}"
      echo "- tokenizer: ${tokenizer_id}"
      echo "- prompt/prefix: ${prompt_mode}, ${nn_prefixes}"
      echo "- formal reward epochs: ${formal_reward_epochs}"
      echo "- max completion length: ${max_completion_length}"
      if [ -n "${generation_batch_size}" ]; then
        echo "- generation batch size: ${generation_batch_size}"
      fi
      if [ "${exclude_train_gpu}" = "1" ]; then
        echo "- reward exclude train GPU: 1"
      fi
      if [ -n "${generation_kwargs_json}" ]; then
        echo "- generation kwargs json: ${generation_kwargs_json}"
      fi
      echo "- samples target: $((num_generations * max_steps)) (${num_generations} generations x ${max_steps} steps)"
      echo "- run root: ${run_root}"
      echo "- stdout/stderr: ${run_root}/slurm/abl-${variant}-${job_id}.out / .err"
      echo "- 训练结果: TODO"
      echo "- 主要缺陷: TODO"
      echo "- 分析: TODO"
    } >> run_archive_index.md
  fi

  echo "variant=${variant} run_id=${run_id} job_id=${job_id} source_commit=${source_commit_short} submit_commit=${submit_commit_short}"
done
