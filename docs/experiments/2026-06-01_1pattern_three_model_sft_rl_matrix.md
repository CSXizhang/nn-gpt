# 2026-06-01 1-Pattern Three-Model SFT+RL Matrix

## Scope

This experiment uses the same 1-pattern CIFAR-10 SFT dataset to train three independent backbone-generation adapters, then uses those adapters as the starting point for a later 3x3 RL matrix.

Current execution scope: start SFT only. Do not run smoke. Do not start RL until the SFT adapters are selected from SFT-cycle statistics.

## Models

| Label | LLM config | Base model |
| --- | --- | --- |
| `dscoder7b` | `backbone_sft_config.json` | `deepseek-ai/deepseek-coder-6.7b-instruct` |
| `mistral7b` | `backbone_sft_mistral_7b_instruct_v03.json` | `mistralai/Mistral-7B-Instruct-v0.3` |
| `qwen7b` | `backbone_sft_qwen2.5_coder_7b_instruct.json` | `Qwen/Qwen2.5-Coder-7B-Instruct` |

All three configs use the same backbone SFT behavior: `only_best_accuracy=true`, `context_length=4096`, `max_input_length=4096`, `max_new_tokens=2048`, `load_in_4bit=false`, `backbone=true`.

## Phase 1: SFT Cycles

Run three independent `TuneBackbone` jobs on Julia2, one per model.

Fixed SFT parameters:

```text
--sft_nn_prefixes rl-bb-test1
--sft_dataset cifar-10
--test_nn 30
--nn_train_epochs 1
--num_train_epochs 1
--num_cycles 20
--sft_max_length 6144
--sft_batch_size 2
--sft_gradient_accumulation 4
```

The SFT dataset is the existing 1-pattern CIFAR-10 backbone set behind prefix `rl-bb-test1`. Generated cycle prefixes are model-specific and recorded in each run's `run_config.json`.

Generation mode: DeepSeek/Qwen use direct generation for the existing local path; Mistral uses the transformers text-generation pipeline chat-message path (`NNGPT_FORCE_DIRECT_GENERATE=0`) to avoid the direct path's tokenizer length sentinel issue.

Output layout:

```text
/home/s471802/nn-gpt/parallel_runs/<run_id>/run_config.json
/home/s471802/nn-gpt/parallel_runs/<run_id>/slurm/
/home/s471802/nn-gpt/out/nngpt/llm/<run_id>/epoch_sft/A*/
```

The active `epoch_root` stays under `/home/s471802/nn-gpt/out/nngpt/...` because `NNEval` assumes synthesized model paths are relative to the active `nngpt_dir`.

## SFT Adapter Selection

Select one adapter per base model using SFT-cycle results only. Do not use RL hindsight to choose the SFT adapter.

Ranking rule:

1. Formal success count.
2. Mean accuracy.
3. Max accuracy and top-k mean accuracy.
4. Structural diversity, including backbone pair, pattern, block, and CNN signature concentration.

Adapter-view rule:

```text
A{k}/synth_nn evaluates the A{k-1} adapter.
A0/synth_nn evaluates the base model.
The final trained adapter has no natural generation/eval point unless an extra probe is run.
```

Therefore, if `A8/synth_nn` has the best SFT-cycle metrics, the selected adapter is `A7/<base_model_name>`.

## Phase 2: RL Matrix

Later, after selecting the three SFT adapters, run RL for every SFT adapter and dataset pair:

| SFT adapter source | RL datasets |
| --- | --- |
| `dscoder7b` selected SFT adapter | `cifar-10`, `cifar-100`, `imagenette` |
| `mistral7b` selected SFT adapter | `cifar-10`, `cifar-100`, `imagenette` |
| `qwen7b` selected SFT adapter | `cifar-10`, `cifar-100`, `imagenette` |

This produces 9 RL adapters.

Fixed RL requirements:

```text
fresh stage2
NNGPT_SFT_LOAD_INITIAL_ADAPTER=1
NNGPT_SFT_INITIAL_ADAPTER_MODE=trainable
NNGPT_RL_FORMAL_REWARD_EPOCHS=1
same RL parameters across the 9 runs except model adapter and dataset
```

Dataset output shapes:

```text
cifar-10: out_shape=(10,)
cifar-100: out_shape=(100,)
imagenette: out_shape=(10,)
```

Do not use `NNGPT_RL_FORMAL_REWARD_EPOCHS=1,5,10` for this matrix. The agreed evaluation/training budget for this experiment is 1 epoch.

## Phase 3: Generation

After the 9 RL adapters finish, generate 30 samples from each RL adapter:

```text
9 RL adapters x 30 samples = 270 samples
```

Keep generation parameters identical across all 9 adapters except the adapter path and dataset-specific metadata. Record each generated sample with its RL adapter source, base model label, RL dataset, generation index, and code commit.

## Phase 4: Cross-Dataset Eval

Evaluate all 270 generated samples on all three datasets:

```text
270 samples x 3 eval datasets = 810 eval records
```

The cross-eval step is for later horizontal comparison. It must preserve the model label, selected SFT adapter cycle, RL dataset, generated sample id, and eval dataset for each record.

## Current Submitter

Local files:

```text
/Users/zhangxi/code/RL/nn-gpt/scripts/julia2_submit_1pattern_three_model_sft.sh
/Users/zhangxi/code/RL/nn-gpt/slurm/julia2_1pattern_three_model_sft_cycle.sbatch
```

Julia2 command:

```bash
cd /home/s471802/nn-gpt
scripts/julia2_submit_1pattern_three_model_sft.sh
```

The submitter starts exactly three SFT jobs: `dscoder7b`, `mistral7b`, and `qwen7b`.

## 2026-06-05 RL Execution Notes

Selected SFT starts, using SFT-cycle statistics only:

| Model | Selected SFT adapter |
| --- | --- |
| DeepSeek | `/home/s471802/nn-gpt/out/nngpt/llm/20260601_1pattern_three_model_sft_v3_cifar10_dscoder7b/epoch_sft/A9/deepseek-ai/deepseek-coder-6.7b-instruct` |
| Mistral | `/home/s471802/nn-gpt/out/nngpt/llm/20260601_1pattern_three_model_sft_v3_cifar10_mistral7b/epoch_sft/A5/mistralai/Mistral-7B-Instruct-v0.3` |
| Qwen | `/home/s471802/nn-gpt/out/nngpt/llm/20260601_1pattern_three_model_sft_v3_cifar10_qwen7b/epoch_sft/A7/Qwen/Qwen2.5-Coder-7B-Instruct` |

Current fixed RL intent:

```text
fresh stage2_formal_explore
NNGPT_SFT_LOAD_INITIAL_ADAPTER=1
NNGPT_SFT_INITIAL_ADAPTER_MODE=trainable
NNGPT_RL_FORMAL_REWARD_EPOCHS=1
NNGPT_SFT_RL_NN_PREFIXES=rl-bb-test1
NNGPT_SFT_NUM_GENERATIONS=8
target: about 1000 training samples per RL adapter
```

The latest main matrix was submitted from commit `f3c397f005de18a8514e58d08b53eacfd3687a5e` with the warmup symmetry and split-cache fixes. Later Mistral KL experiments used commit `c9276599c8dd1df4def862b21dd3a766638bccd5`.

### Current Run Status

As of 2026-06-05 morning Julia2 time:

| Model | Dataset | Run ID / job | State | Notes |
| --- | --- | --- | --- | --- |
| Qwen | `cifar-10` | `20260604_1748_rl_qwen_cifar_10_std` / `2665741` | manually stopped after overrun | 1192 samples; stopped checkpoint is treated as the target 1000-sample adapter. |
| Qwen | `cifar-100` | `20260604_1748_rl_qwen_cifar_100_std` / `2665743` | manually stopped after overrun | 1600 samples; stopped checkpoint is treated as the target 1000-sample adapter. |
| Qwen | `imagenette` | `20260604_1748_rl_qwen_imagenette_std` / `2665745` | manually stopped after overrun | 1872 samples; stopped checkpoint is treated as the target 1000-sample adapter. |
| DeepSeek | `cifar-10` | `20260604_2057_rl_dscoder_cifar_10_std` / `2665839` | manually stopped after overrun | 1312 samples; stopped checkpoint is treated as the target 1000-sample adapter. |
| DeepSeek | `cifar-100` | `20260604_2057_rl_dscoder_cifar_100_std` / `2665841` | manually stopped after overrun | 1112 samples; stopped checkpoint is treated as the target 1000-sample adapter. |
| DeepSeek | `imagenette` | `20260605_0925_dscoder_imagenette_h100` / `2666209` | pending on `h100` | Pending reason `Resources`; Slurm estimated start `2026-06-05T11:40:19`. |
| Mistral | `cifar-10` | `20260605_0033_mistral_a7_cifar_10_kl08` / `2665950` | running | KL beta 0.08; at last check 744 samples and recovering after a weak early phase. |
| Mistral | `cifar-100` | `20260605_0033_mistral_a7_cifar_100_kl08` / `2665952` | `OUT_OF_MEMORY` | 1184 samples before OOM; also structurally collapsed. |
| Mistral | `imagenette` | `20260605_0033_mistral_a7_imagenette_kl08` / `2665954` | `OUT_OF_MEMORY` | 832 samples before OOM; late samples recovered, then host memory OOM killed the job. |

Important overrun decision: for the five stopped DeepSeek/Qwen jobs, do not reconstruct or roll back to the exact 1000th sample. The saved stopped adapter/checkpoint is the adapter used for the 1000-sample matrix result, and the extra samples are recorded as an overrun caused by a missing max-step/sample cap.

### DeepSeek and Qwen Health Check

DeepSeek and Qwen do not show the Mistral-style structural collapse. The main evidence is high execution validity:

| Run | Samples | first-1000 formal1 | first-1000 success | first-1000 forward/backward | Comment |
| --- | ---: | ---: | ---: | --- | --- |
| Qwen `cifar-10` | 1192 | 0.907 | 0.974 | 0.944 / 0.941 | Healthy through target budget; overrun tail degraded. |
| Qwen `cifar-100` | 1600 | 0.618 | 0.949 | 0.952 / 0.948 | Healthy; tail improved. |
| Qwen `imagenette` | 1872 | 0.991 | 0.951 | 0.944 / 0.941 | Healthy; tail improved. |
| DeepSeek `cifar-10` | 1312 | 0.904 | 0.980 | 0.980 / 0.980 | Healthy. |
| DeepSeek `cifar-100` | 1112 | 0.746 | 0.957 | 0.960 / 0.956 | Healthy. |

One caveat: several DeepSeek/Qwen overrun tails converge toward plain dual-backbone concat with low `actual_block_live`. This is not a dual-backbone failure and is not comparable to Mistral collapse, but it may limit structural diversity. Qwen `cifar-10` specifically showed overrun-tail degradation: last-100 reward -0.753, success 0.360, backward ok 0.320. Treat the stopped adapter as the agreed 1000-sample adapter, but inspect its later generation/cross-eval carefully.

### Mistral Diagnosis

Mistral failures split into three categories:

1. Bad early KL=0.04 submissions with wrong dataset names:
   - Jobs `2665917` and `2665919` failed in about 17 seconds because `NNGPT_RL_FORMAL_DATASET` was set to `cifar_10` / `cifar_100`; the code expects `cifar-10` / `cifar-100`.

2. KL=0.04 retry collapse:
   - `imagenette` retry: 64 samples, reward -1.901, success 0.315, forward/backward ok 0.266.
   - `cifar-10` retry: 48 samples, reward -1.129, success 0.378, forward/backward ok 0.354.
   - `cifar-100` retry: 48 samples, reward -0.345, success 0.783, but `actual_block_live` 0.152.
   - Interpretation: KL=0.04 allows the Mistral policy to drift too far from the SFT adapter, causing code-structure instability.

3. KL=0.08 mixed result:
   - `cifar-10` recovered after a weak early phase. Around 744 samples, last-100 reward was about 0.164, formal1 about 0.873, success about 0.889.
   - `cifar-100` remained structurally collapsed: 1184 samples, reward -2.754, success 0.034, forward ok 0.023, backward ok 0.021. It also ended with Slurm host-memory OOM.
   - `imagenette` recovered late: 832 samples total, last-100 reward 0.271, formal1 0.969, success 0.755, forward/backward ok 0.740. It ended with Slurm host-memory OOM, not reward collapse.

The Mistral pattern is consistent with the model not being code-specialized in the same way as DeepSeek-Coder or Qwen-Coder. When RL moves the policy distribution away from the SFT adapter, Mistral is more likely to fail on Python/model structure, tensor shapes, or forward/backward execution. KL=0.08 helps, but it does not rescue `cifar-100`, where the 100-class output head and shape constraints are stricter.

### OOM Notes

Mistral `2665952` and `2665954` were killed by Slurm host-memory OOM, not a CUDA OOM:

```text
2665952 batch MaxRSS about 83.9 GB, ReqMem 80G, state OUT_OF_MEMORY
2665954 batch MaxRSS about 83.9 GB, ReqMem 80G, state OUT_OF_MEMORY
Root cause: SIGKILL / oom_kill in the Slurm batch step
```

This is separate from reward collapse. For `cifar-100`, both collapse and host-memory OOM happened. For `imagenette`, late samples looked usable before the host-memory OOM.
