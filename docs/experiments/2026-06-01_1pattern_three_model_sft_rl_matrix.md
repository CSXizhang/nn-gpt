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
--test_nn 30
--nn_train_epochs 1
--num_train_epochs 1
--num_cycles 20
--sft_max_length 6144
--sft_batch_size 2
--sft_gradient_accumulation 4
```

The SFT dataset is the existing 1-pattern CIFAR-10 backbone set behind prefix `rl-bb-test1`. Generated cycle prefixes are model-specific and recorded in each run's `run_config.json`.

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
