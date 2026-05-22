#!/bin/bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Submit the 100-sample baseline plan on Julia2.

Required:
  --onepattern-adapter PATH
  --fourpattern-adapter PATH

Options:
  --run-id ID              default: baseline_YYYYmmdd_HHMM
  --budget N              default: 100
  --partition PART        default: h100
  --eval-partition PART   default: h100
  --brute-partition PART  default: small_cpu
  --brute-gres GRES       optional, e.g. gpu:1 when CPU partitions are blocked
  --prompt-nn-prefixes P  default: rl-bb-test1
  --onepattern-nn-prefixes P
                           default: rl-bb-test1
  --fourpattern-nn-prefixes P
                           default: rl-bb-struct1

This submits three 1-GPU gen jobs, one CPU brute-gen job, then one 4-GPU
eval job depending on all generators. Results are written under
parallel_runs/<run_id>/baseline.
USAGE
}

onepattern_adapter=""
fourpattern_adapter=""
run_id="baseline_$(date +%Y%m%d_%H%M)"
budget="100"
partition="h100"
eval_partition="h100"
brute_partition="small_cpu"
brute_gres=""
prompt_nn_prefixes="rl-bb-test1"
onepattern_nn_prefixes="rl-bb-test1"
fourpattern_nn_prefixes="rl-bb-struct1"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --onepattern-adapter) onepattern_adapter="$2"; shift 2 ;;
    --fourpattern-adapter) fourpattern_adapter="$2"; shift 2 ;;
    --run-id) run_id="$2"; shift 2 ;;
    --budget) budget="$2"; shift 2 ;;
    --partition) partition="$2"; shift 2 ;;
    --eval-partition) eval_partition="$2"; shift 2 ;;
    --brute-partition) brute_partition="$2"; shift 2 ;;
    --brute-gres) brute_gres="$2"; shift 2 ;;
    --nn-prefixes) prompt_nn_prefixes="$2"; onepattern_nn_prefixes="$2"; fourpattern_nn_prefixes="$2"; shift 2 ;;
    --prompt-nn-prefixes) prompt_nn_prefixes="$2"; shift 2 ;;
    --onepattern-nn-prefixes) onepattern_nn_prefixes="$2"; shift 2 ;;
    --fourpattern-nn-prefixes) fourpattern_nn_prefixes="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -z "${onepattern_adapter}" ] || [ -z "${fourpattern_adapter}" ]; then
  usage >&2
  exit 2
fi

cd /home/s471802/nn-gpt
run_root="/home/s471802/nn-gpt/parallel_runs/${run_id}/baseline"
mkdir -p "${run_root}"

base_export="ALL,NNGPT_BASELINE_RUN_ROOT=${run_root},NNGPT_BASELINE_BUDGET=${budget},NNGPT_SFT_BASE_MODEL_ID=/home/s471802/nn-gpt/out/llm/deepseek-ai/deepseek-coder-6.7b-instruct,NNGPT_SFT_TOKENIZER_ID=deepseek-ai/deepseek-coder-6.7b-instruct,NNGPT_RL_FORMAL_REWARD_EPOCHS=1,5,10"

prompt_job=$(sbatch --parsable -p "${partition}" --gres=gpu:1 \
  --job-name baseline-gen-prompt \
  --export "${base_export},NNGPT_BASELINE_SETTING=prompt_only,NNGPT_BASELINE_SOURCE_RUN=prompt_only_base,NNGPT_SFT_RL_NN_PREFIXES=${prompt_nn_prefixes}" \
  slurm/julia2_baseline_gen_only.sbatch)
prompt_job="${prompt_job%%;*}"

one_job=$(sbatch --parsable -p "${partition}" --gres=gpu:1 \
  --job-name baseline-gen-sft1 \
  --export "${base_export},NNGPT_BASELINE_SETTING=sft_only_onepattern,NNGPT_BASELINE_SOURCE_RUN=onepattern_selected_sft,NNGPT_BASELINE_ADAPTER_PATH=${onepattern_adapter},NNGPT_BASELINE_ADAPTER_MODE=trainable,NNGPT_SFT_RL_NN_PREFIXES=${onepattern_nn_prefixes}" \
  slurm/julia2_baseline_gen_only.sbatch)
one_job="${one_job%%;*}"

four_job=$(sbatch --parsable -p "${partition}" --gres=gpu:1 \
  --job-name baseline-gen-sft4 \
  --export "${base_export},NNGPT_BASELINE_SETTING=sft_only_fourpattern,NNGPT_BASELINE_SOURCE_RUN=fourpattern_selected_sft,NNGPT_BASELINE_ADAPTER_PATH=${fourpattern_adapter},NNGPT_BASELINE_ADAPTER_MODE=trainable,NNGPT_SFT_RL_NN_PREFIXES=${fourpattern_nn_prefixes}" \
  slurm/julia2_baseline_gen_only.sbatch)
four_job="${four_job%%;*}"

brute_sbatch_args=(-p "${brute_partition}")
if [ -n "${brute_gres}" ]; then
  brute_sbatch_args+=(--gres="${brute_gres}")
fi

brute_job=$(sbatch --parsable "${brute_sbatch_args[@]}" \
  --job-name baseline-brute-gen \
  --export "${base_export},NNGPT_BASELINE_SETTING=brute_constrained_random,NNGPT_BASELINE_SOURCE_RUN=brute_heldout_random" \
  slurm/julia2_baseline_brute_gen_only.sbatch)
brute_job="${brute_job%%;*}"

candidate_files="${run_root}/candidates/prompt_only/candidates.jsonl,${run_root}/candidates/sft_only_onepattern/candidates.jsonl,${run_root}/candidates/sft_only_fourpattern/candidates.jsonl,${run_root}/candidates/brute_constrained_random/candidates.jsonl"

eval_job=$(sbatch --parsable -p "${eval_partition}" --gres=gpu:4 \
  --dependency "afterok:${prompt_job}:${one_job}:${four_job}:${brute_job}" \
  --job-name baseline-eval-400 \
  --export "${base_export},NNGPT_BASELINE_CANDIDATE_FILES=${candidate_files},NNGPT_REWARD_WORKERS_PER_GPU=1" \
  slurm/julia2_baseline_eval_only.sbatch)
eval_job="${eval_job%%;*}"

summary_job=$(sbatch --parsable -p small_cpu \
  --dependency "afterok:${eval_job}" \
  --job-name baseline-summary \
  --export "${base_export}" \
  slurm/julia2_baseline_summarize.sbatch)
summary_job="${summary_job%%;*}"

cat <<EOF
run_root=${run_root}
prompt_job=${prompt_job}
sft_onepattern_job=${one_job}
sft_fourpattern_job=${four_job}
brute_job=${brute_job}
eval_job=${eval_job}
summary_job=${summary_job}
EOF
