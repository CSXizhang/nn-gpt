# Consolidated experiment status

Status checked on 2026-08-02 against Julia2 Slurm and retained run outputs.

## 1. SFT-only counterfactual

The replacement smoke/runner validation cleared. All twelve matched generation jobs were submitted at the full protocol: one-pattern and four-pattern adapters, six seeds each, 800 candidates per adapter, followed by 5-epoch held-out CIFAR-10 evaluation.

Current state:

- Generation complete: 5/12 conditions; generation still running: 7/12.
- Fully merged evaluation complete: 4/12 conditions.
- The 192 single-GPU evaluation shards were submitted as dependencies.
- One 50-candidate evaluation shard for one-pattern seed123 failed with Slurm OOM on L40 after 33 minutes. The other 15 shards completed. A like-for-like H100/96 GB retry (`2991938`) also OOMed after candidate 6/50, confirming host-memory accumulation across candidates rather than GPU capacity as the cause.
- The failed shard is now split into 50 isolated single-candidate tasks in array `2992012`, with five H100 tasks running concurrently. Candidate index 6 also OOMed in isolation at 24 GB host memory and is being retried once as job `2992067` with 256 GB host memory. Shard merge `2992068` waits for the array and this replacement; final 800-row merge `2992069` follows. This changes only process isolation and host-memory allocation, not candidates, seeds, epochs, split, or metrics.
- No reduced experiment is needed because the full experiment is running. A defensible fallback would be three seeds per condition with the same 800-candidate budget, but it is not currently being used.
- Current ETA: approximately 12-18 hours for the remaining generation/evaluation chain if the running allocations continue normally; this is a cluster-dependent estimate.

Interim completed-condition metrics (not the final aggregate):

| Condition | Seed | Samples | Formal success | Mean 5-epoch test accuracy | Max accuracy |
|---|---:|---:|---:|---:|---:|
| one-pattern SFT-only | 114 | 800 | 741/800 (92.63%) | 88.11% | 92.90% |
| one-pattern SFT-only | 514 | 800 | 732/800 (91.50%) | 87.99% | 92.68% |
| four-pattern SFT-only | 114 | 800 | 797/800 (99.63%) | 86.95% | 93.20% |
| four-pattern SFT-only | 919 | 800 | 798/800 (99.75%) | 87.22% | 93.11% |

The fifth generation-complete condition, one-pattern seed123, is awaiting the replacement shard and merge. These interim results already show that the earlier one-epoch one-seed four-pattern value of 30.30% should not be generalized to the matched 5-epoch counterfactual. The title decision should wait for all twelve aggregates.

## 2. Four-pattern CIFAR-100

This experiment can run. The three seeds will be run in parallel on H100, and the final three-seed aggregate is expected by 2026-08-04.

## 3. Imagenette

This experiment can run. The planned matched conditions will be run after CIFAR-100, and the final results are expected by 2026-08-06.
