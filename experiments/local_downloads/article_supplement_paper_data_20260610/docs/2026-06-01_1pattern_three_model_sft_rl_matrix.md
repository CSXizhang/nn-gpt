# 2026-06-01 1-Pattern Three-Model SFT+RL Matrix

## Scope

This experiment uses the same 1-pattern CIFAR-10 SFT dataset to train three independent backbone-generation adapters, then uses those adapters as the starting point for a later 3x3 RL matrix.

Current execution scope: start SFT only. Do not start RL until the SFT adapters are selected from SFT-cycle statistics.

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

## 2026-06-05 Granite / OlympicCoder Follow-Up

Advisor suggested a quick Granite check and OlympicCoder as the most viable coder-specialized alternative if Granite works. We skipped raw instruct scoring because it is not comparable to the SFT-start matrix.

Added local/remote SFT configs:

| Label | Config | Base model |
| --- | --- | --- |
| Granite | `backbone_sft_granite_4_1_8b.json` | `ibm-granite/granite-4.1-8b` |
| OlympicCoder | `backbone_sft_olympic_coder_7b.json` | `open-r1/OlympicCoder-7B` |

Storage adjustment:

- New model cache directories are symlinked from `/home/s471802/nn-gpt/out/llm/{ibm-granite,open-r1}` to `/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-models/{ibm-granite,open-r1}`.
- New SFT run roots use `/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs`.
- SFT epoch logical paths remain under `/home/s471802/nn-gpt/out/nngpt/llm/<run_id>` but are symlinked to `/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/sft_epoch_roots/<run_id>` to preserve NNEval path assumptions while avoiding home quota.

Formal 10-cycle SFT:

| Model | Run ID / job | Dependency | Partition | Notes |
| --- | --- | --- | --- | --- |
| Granite | `20260605_granite_1pattern_sft10_pipe0_granite4_1_8b` / `2666708` | none | `gpu_computervision_long` | 10 cycles, `test_nn=30`, `max_length=6144`, `batch=2`, `grad_accum=4`, SFT dataset cap `1000`, `NNGPT_FORCE_DIRECT_GENERATE=0`. |
| OlympicCoder | `20260605_olympiccoder_1pattern_sft10_pipe0_no_think_template` / `2666897` | none | `gpu_computervision_long` | 10 cycles, `test_nn=30`, `max_length=4096`, `batch=1`, `grad_accum=8`, SFT dataset cap `1000`, config-level no-think chat template, `NNGPT_FORCE_DIRECT_GENERATE=0`. |

Initial H100 submissions `2666334` and `2666335` were cancelled before training started because H100 had no clear 4-GPU free slot; `gpu_computervision_long` had an idle L40S node and the 6144 smoke was already running successfully on L40S.

Dataset cap decision: cap Granite/OlympicCoder formal 10-cycle SFT at `1000` SFT rows via `NNGPT_1PATTERN_SFT_DATASET_LIMIT=1000`. The current prefix-backed pool has grown to about `1200` rows, so uncapped SFT would give these follow-up models a larger and later training set than the original three-model matrix. The cap is applied before prompt formatting and uses the database return order, so newly appended rows should not enter the fixed training budget.

Uncapped dependency submissions `2666337` and `2666338` were cancelled before training started. Earlier capped Granite/OlympicCoder submissions were cancelled before training started and replaced by the current formal jobs above.

Do not start RL for these models until the 10-cycle SFT results are reviewed and adapters are selected using SFT-cycle statistics only.

## 2026-06-05 RL Adapter Generation / Cross-Eval Follow-Up

The original `20260605_rl_adapter_gen30_snapshot` eval is no longer the comparison target for DeepSeek retry analysis. The usable follow-up generation is `20260605_rl_adapter_gen30_snapshot_min32`, where DeepSeek generation success was:

| Setting | Candidate-code success |
| --- | ---: |
| `dscoder_cifar10` | `26/30` |
| `dscoder_cifar100` | `29/30` |

The `rlaligned` retry was dropped because it still had low generation success (`16/30` for CIFAR10, `9/30` for CIFAR100).

Submitted Qwen `min32w` generation with the same generation parameters (`NNGPT_SFT_GENERATION_KWARGS_JSON={"min_new_tokens":32}`):

| Setting | Job |
| --- | --- |
| `qwen_cifar10` | `2666392` |
| `qwen_cifar100` | `2666393` |
| `qwen_imagenette` | `2666394` |

Submitted 3-dataset cross-eval over all `min32w` candidates after the Qwen generation jobs finish. Candidate files: `dscoder_cifar10`, `dscoder_cifar100`, `qwen_cifar10`, `qwen_cifar100`, `qwen_imagenette`.

| Eval dataset | Job | Dependency |
| --- | --- | --- |
| `cifar-10` | `2666395` | `afterok:2666392:2666393:2666394` |
| `cifar-100` | `2666396` | `afterok:2666392:2666393:2666394` |
| `imagenette` | `2666397` | `afterok:2666392:2666393:2666394` |

Initial Qwen `min32w` submissions `2666383`-`2666385` failed before candidate generation because the JSON env for `min_new_tokens` was shell-quoted incorrectly; dependent eval jobs `2666386`-`2666388` were cancelled. `baseline_experiment_runner.py` now records the actual `NNGPT_RL_FORMAL_DATASET` in eval-only `run_config.json` and supports explicit `--min-new-tokens`, so the final jobs above should have unambiguous metadata.

## 2026-06-05 Current Snapshot and Next RL Plan

This section supersedes the earlier pending/running notes in this file for the current 1-pattern matrix follow-up.

### Cluster status

As of the last check on 2026-06-05, `squeue -u s471802` is empty. All current generation and heldout-test eval jobs for the DeepSeek/Qwen cross-dataset comparison have finished.

DeepSeek Imagenette RL finished normally:

| Run | Job | State | Samples | Final checkpoint | Notes |
| --- | ---: | --- | ---: | --- | --- |
| `20260605_0925_dscoder_imagenette_h100` | `2666209` | `COMPLETED` | `1000` | `checkpoint-125` | H100 retry after L40 OOM; formally completed the intended 1000-sample budget. |

Late RL quality warning for this run: last-100 formal success and accuracy were high, but `actual_block_live=0/100`; the policy converged to runnable dual-backbone shortcut structures rather than live block usage.

### Heldout-test cross-eval result

Final merged cross-eval root:

```text
/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260605_rl_adapter_gen30_snapshot_min32_testset/cross_eval
```

This root now contains 180 rows per eval dataset: six settings times 30 generated candidates. Old bad `qwen_cifar10` rows were removed and replaced with the checkpoint-130 generation. `dscoder_imagenette` was generated from DeepSeek Imagenette `checkpoint-125` and then merged into the same comparison root.

DeepSeek:

| Setting | Source dataset | CIFAR10 success | CIFAR10 acc | CIFAR100 success | CIFAR100 acc | Imagenette success | Imagenette acc | All-3 success | Macro success | Zero-fail macro acc |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dscoder_cifar10` | CIFAR10 | 26/30 | 0.7960 | 26/30 | 0.6363 | 26/30 | 0.8627 | 26/30 | 0.8667 | 0.7650 |
| `dscoder_cifar100` | CIFAR100 | 29/30 | 0.8949 | 29/30 | 0.7281 | 29/30 | 0.9617 | 29/30 | 0.9667 | 0.8616 |
| `dscoder_imagenette` | Imagenette | 29/30 | 0.8732 | 29/30 | 0.6847 | 26/30 | 0.8554 | 26/30 | 0.9333 | 0.8044 |

Qwen:

| Setting | Source dataset | CIFAR10 success | CIFAR10 acc | CIFAR100 success | CIFAR100 acc | Imagenette success | Imagenette acc | All-3 success | Macro success | Zero-fail macro acc |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen_cifar10` | CIFAR10 | 30/30 | 0.9168 | 30/30 | 0.7242 | 30/30 | 0.9958 | 30/30 | 1.0000 | 0.8789 |
| `qwen_cifar100` | CIFAR100 | 26/30 | 0.8045 | 26/30 | 0.6493 | 26/30 | 0.8637 | 26/30 | 0.8667 | 0.7725 |
| `qwen_imagenette` | Imagenette | 21/30 | 0.6350 | 21/30 | 0.5044 | 21/30 | 0.6962 | 21/30 | 0.7000 | 0.6119 |

Interpretation:

- There is no strong dataset-specific specialization in the heldout-test cross-eval. A dataset-specialized adapter would be expected to improve mainly on its own eval dataset; instead, the stronger checkpoints tend to be stronger across all three eval datasets.
- The main observed factor is RL/checkpoint quality, not source-dataset bias. `qwen_cifar10` and `dscoder_cifar100` dominate broadly.
- All generated/eval settings report `Block live = 0.0000` in the comparison metrics. Therefore these results should be treated as runnable backbone-shortcut quality, not evidence of successful live-block structure learning.

### RL generation-distribution observation

The RL process did change the generated model distribution by dataset, but the exact final backbone pair is path-dependent.

Observed DeepSeek examples:

| Run | Dataset | Late dominant generated family / pair | Live-block status |
| --- | --- | --- | --- |
| `20260604_2057_rl_dscoder_cifar_10_std` | CIFAR10 | `OpenMotif_284dd2`; mostly `regnet_x_1_6gf + swin_t` variants | late `0/200` live |
| `20260604_2057_rl_dscoder_cifar_100_std` | CIFAR100 | `OpenMotif_b568d4`; `regnet_x_32gf + shufflenet_v2_x2_0` | late `0/200` live |
| `20260605_0925_dscoder_imagenette_h100` | Imagenette | `DualBackbone_Fuse_Wide_*`; mostly `resnet18 + resnet50` | late `0/200` live |
| `20260604_1829_rl_dscoder_cifar_100_std` | CIFAR100 | `DualBackbone_Fuse_Wide_*`; `regnet_y_800mf + shufflenet_v2_x0_5` | late `190/200` live |
| `20260604_1829_rl_dscoder_imagenette_std` | Imagenette | `DualBackbone_Fuse_Wide_*`; `regnet_y_800mf + shufflenet_v2_x1_0` | late `197/199` live |

The two `1829` DeepSeek runs are useful diagnostics: they were cancelled early at 208/248 samples, used the `ABrain/NNGPT-Backbone-deepseek-coder-6.7b-instruct` wrapper base, and still had high live-block rates. They are not full 1000-sample matrix runs, but they suggest that continuing RL under the current reward lets the policy later discover and exploit block-dead shortcuts.

### Granite / OlympicCoder status

The original 10-cycle Granite/OlympicCoder jobs `2666346` and `2666347` were cancelled because generation quality was unusable and would have polluted the experiment.

Current formal runs are `2666708` for Granite and `2666897` for OlympicCoder. Granite uses the non-direct generation path. OlympicCoder uses a config-level no-think chat template rather than a prompt change.

### Reward/RL problem summary

The current reward is insufficiently aligned with the intended `Parallel_Triple` structure. It allows high heldout-test accuracy with no live block contribution:

```text
generated code is runnable
dual TorchVision backbones exist or old target match passes
forward/backward works
but actual_block_live = 0
```

This causes late-stage RL to converge toward high-accuracy dual-backbone shortcut models. The problem is not mainly dataset bias. The main issue is that the reward gives enough credit to block-dead solutions.

### Recommended RL improvement plan

1. Make `actual_block_live` a hard structural gate for positive structural reward.

   Candidates with `actual_block_live=false` should not receive the same structure credit as valid `Parallel_Triple` candidates. Keep strong negative rewards for invalid Python/build/forward/backward failures, but separate "runnable shortcut" from "target structure achieved".

2. Split reward reporting into two explicit metrics.

   Track both:

   ```text
   task_reward = heldout/reward_eval accuracy component
   structure_reward = target pattern + live block + declared/actual consistency
   ```

   Do not let high task accuracy hide a zero structure score in dashboards or checkpoint selection.

3. Use a staged schedule instead of a single mixed objective.

   Recommended schedule:

   ```text
   first 200-300 samples: structure gate dominates; require live block and correct two-backbone topology
   next 400-500 samples: accuracy and diversity share weight with structure
   final 200-300 samples: checkpoint selection uses Pareto rule, not accuracy alone
   ```

   This directly targets the observed failure mode: early live-block samples exist, but later training drifts to shortcuts.

4. Add checkpoint selection by structure-aware validation.

   Do not use only the final checkpoint. Save/evaluate every 5 checkpoints and choose by:

   ```text
   live_block_rate >= threshold
   formal_success_rate >= threshold
   zero-fail macro accuracy
   diversity / duplicate penalty
   ```

   For the current evidence, the early `1829` checkpoints are exactly the type that would be preserved by this rule.

5. Add an anti-collapse penalty for repeated graph/backbone signatures.

   The RL traces show late concentration into one or two signatures. Penalize repeated `actual_structure_signature`, backbone pair, and block signature within a moving window. This should be a reward term, not a generation-time repair or filter.

6. Run a small ablation before rerunning the full matrix.

   Recommended ablation matrix:

   | Model | Dataset | Samples | Purpose |
   | --- | --- | ---: | --- |
   | DeepSeek | CIFAR100 | 300 | Check whether live-block gate prevents the known `OpenMotif` shortcut. |
   | DeepSeek | Imagenette | 300 | Check whether it preserves live-block behavior beyond the early good region. |
   | Qwen | CIFAR10 | 300 | Ensure the strongest current adapter does not lose basic generation validity. |

   Pass condition before formal rerun:

   ```text
   formal_success_rate >= 0.85
   actual_block_live_rate >= 0.50 in last 100 samples
   no single structure_signature > 0.50 in last 100 samples
   ```

7. Only after the ablation passes, rerun the formal matrix.

   Rerun DeepSeek and Qwen first. Do not spend more Mistral GPU time until the structure reward is fixed and a short Mistral probe shows stable forward/backward execution.
