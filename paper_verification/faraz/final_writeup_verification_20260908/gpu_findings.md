# GPU-hour accounting verification

> Updated 2026-09-09 with read-only Julia2 evidence. `gpu_hours_recomputed.json` remains the reproducible **local-only 37-row snapshot**; the current corrected wide38 and grand totals are in `remote_evidence/wide38/gpu_task0_recomputed.json`.

## Status

**VERIFIED for the previously declared extended cohorts, with successful proxy evaluations.** Local records directly verify every prior subtotal except wide38 task 0. Read-only Julia2 evidence identifies task 0 as completed job `2968490_0`, 1 L40S for 1,920 seconds, and connects it to manifest selection 0, candidate `4pattern_seed42_743`, stdout completion, and result file. Adding that missing successful task yields wide38 `19.8175` and corrected extended total `1886.4472222222...` GPU-hours (`1886.45`).

## Claim being checked

The paper/email accounting claims allocated GPU-hours of approximately 374.03 (one-pattern), 533.64 (four-pattern), 907.67 (primary total), 553.72 (H100), 353.94 (L40S), plus the listed extended cohorts, ending in 1885.91 or 1885.92 GPU-hours.

## Evidence used

- Primary retained Slurm snapshot: `/Users/zhangxi/code/RL/delivery_20260727/06_gpu_hours/sacct_six_seed_cifar10_rl_20260728.csv` (12 rows; SHA-256 `f27bc090a41b3ab465d8e7ce1f5baf82ca29c5fdc0d14e0afd8e127d1e1cd47d`).
- Primary original recomputation script: `/Users/zhangxi/code/RL/delivery_20260727/06_gpu_hours/build_gpu_hours_table.py`. It was rerun against a temporary copy to avoid rewriting delivery files. Its generated outputs are retained under `original_script_rerun/`.
- Extended retained Slurm snapshot: `/Users/zhangxi/code/RL/reply_20260802/08_sacct_records.csv` (73 rows; SHA-256 `a147af9357063c407fab097e7286113a6982c92cdf4146941ce9a65e43b4151e`). This hash matches `/Users/zhangxi/code/RL/reply_20260802/CHECKSUMS.sha256`.
- Extended original recomputation script: `/Users/zhangxi/code/RL/reply_20260802/09_build_gpu_hours.py`. Its rerun stdout is retained under `original_script_rerun/`.
- Cross-check for the wide proxy scope: `/Users/zhangxi/code/RL/delivery_20260727/04_proxy_validation/proxy_manifest_wide38.jsonl` (38 rows; SHA-256 `87c570d62c96f7d959d0286cd128420001ef46e8ae76d4953b67bdf8d7fa29c3`) defines task IDs 0 through 37 in `selection_index`; its `results/` directory also contains those tasks. A local search found no retained row for task 0 (`2968494_0`) or raw job `2968496`.

These CSVs are retained local accounting snapshots, not a fresh query to the Slurm database. Their row values and checksums are locally verifiable; live scheduler provenance is outside this offline audit.

## Recompute method

For each retained job or array-task row:

`allocated GPU-hours = elapsed_seconds * allocated_gpu_count / 3600`

All sums use exact rational arithmetic. No stored `gpu_hours` decimal and no Markdown subtotal is used as an input. The audit script also checks each stored extended decimal against the integer-derived value within `5e-7`; asserts `job_id` uniqueness inside each snapshot; verifies that the only cross-snapshot duplicate IDs are the three declared ablation overlaps and that their `elapsed_seconds`, `gpu_count`, and `gpu_type` agree; derives expected wide-proxy task IDs from the manifest's `selection_index`; and lists every included job ID. This measures allocation time, not actual device utilisation.

Reproduction command:

```bash
python3 final_writeup_verification_20260908/scripts/recompute_gpu_hours.py \
  --primary-sacct delivery_20260727/06_gpu_hours/sacct_six_seed_cifar10_rl_20260728.csv \
  --extended-sacct reply_20260802/08_sacct_records.csv \
  --wide-manifest delivery_20260727/04_proxy_validation/proxy_manifest_wide38.jsonl \
  --output final_writeup_verification_20260908/gpu_hours_recomputed.json
```

## Exact result

| Scope | GPU-seconds | Exact GPU-hours | Decimal (15 dp) | Rounded (2 dp) | Verification |
|---|---:|---:|---:|---:|---|
| Primary one-pattern | 1,346,500 | 13465/36 | 374.027777777777778 | 374.03 | VERIFIED |
| Primary four-pattern | 1,921,100 | 19211/36 | 533.638888888888889 | 533.64 | VERIFIED |
| Primary total | 3,267,600 | 2723/3 | 907.666666666666667 | 907.67 | VERIFIED |
| Primary H100 | 1,993,404 | 166117/300 | 553.723333333333333 | 553.72 | VERIFIED |
| Primary L40S | 1,274,196 | 106183/300 | 353.943333333333333 | 353.94 | VERIFIED |
| Ablation, all 9 rows | 3,364,156 | 841039/900 | 934.487777777777778 | 934.49 | VERIFIED |
| Ablation overlap with primary | 1,586,916 | 44081/100 | 440.810000000000000 | 440.81 | VERIFIED |
| Ablation unique addition | 1,777,240 | 44431/90 | 493.677777777777778 | 493.68 | VERIFIED |
| Qwen CIFAR-10 | 845,872 | 52867/225 | 234.964444444444444 | 234.96 | VERIFIED for 4 retained segments |
| CIFAR-100 one-pattern | 758,504 | 94813/450 | 210.695555555555556 | 210.70 | VERIFIED for 3 retained jobs |
| Proxy top20 | 70,651 | 70651/3600 | 19.625277777777778 | 19.63 | VERIFIED for 20 retained tasks |
| Proxy labelled wide38 | 69,423 | 23141/1200 | 19.284166666666667 | 19.28 | PARTIAL: only 37 retained tasks |
| Unique total of supplied retained rows | 6,789,290 | 678929/360 | 1885.913888888888889 | 1885.91 | VERIFIED as supplied-row sum; complete scope NOT VERIFIED |
| Remote-recovered wide38 task 0 (`2968490_0`) | 1,920 | 8/15 | 0.533333333333333 | 0.53 | VERIFIED |
| Corrected complete successful wide38 | 71,343 | 39635/2000 | 19.817500000000000 | 19.82 | VERIFIED |
| Corrected extended unique total | 6,791,210 | 679121/360 | 1886.447222222222222 | 1886.45 | VERIFIED for previously declared cohorts |

The ablation overlap is exactly jobs `2697817`, `2697840`, and `2697842`. The six unique ablation additions are `2697779`, `2697849`, `2751655`, `2755237`, `2760934`, and `2771489`. Qwen includes initial/continuation segments `2791929` and `2793568` for seed 777, as well as `2791927` and `2791928`. Complete per-job details are in `gpu_hours_recomputed.json`.

## Difference from Faraz / previous Xi summary

The expected two-decimal primary and extended subtotals reproduce exactly from the supplied local accounting rows. The 1885.91 versus 1885.92 discrepancy is solely a rounding-order effect **for that incomplete local 37-wide-row snapshot**:

- Sum exact cohort values, then round once: `1885.913888888888889 -> 1885.91`.
- Round the six unique cohort components first (`907.67 + 493.68 + 234.96 + 210.70 + 19.63 + 19.28`) and then sum: `1885.92`.

Both values omit task 0 and are superseded by 1886.45 for the previously declared extended cohorts. There were two local-snapshot inconsistencies that were not rounding effects:

1. `wide38` was described as 38 rows, but the local snapshot retained only `2968494_1` through `_37`. Remote evidence now resolves task 0 as the earlier successful job `2968490_0`, so the corrected total is 19.8175.
2. The prior extended CSV says 84 jobs/segments. Row-level deduplication yields 82 distinct supplied accounting records: 12 primary + 73 extended - 3 overlapping ablation jobs. The reported 84 is unsupported by the supplied raw rows.

## Safe statement for the paper/email

“Allocated GPU-hours are calculated as Slurm elapsed wall time multiplied by allocated GPU count. The primary six-seed CIFAR-10 cohort used 907.67 allocated GPU-hours (374.03 one-pattern; 533.64 four-pattern), comprising 553.72 H100 and 353.94 L40S GPU-hours. After restoring successful wide38 task 0 from Julia2 accounting, the deduplicated total for the previously declared extended cohorts is 1886.4472222 GPU-hours, or 1886.45 rounded once.”

Do not state 1885.91 or 1885.92 as the current complete total; they describe the incomplete local 37-row snapshot.

## Remaining uncertainty

- The failed superseded `2968450` array consumed small allocations but is excluded, consistently with the original successful-proxy sub-scope. Adding it would define a broader all-attempt cost and was not done.
- The accounting intentionally follows the retained inclusion decisions (for example, keeping useful failed job `2751655`, including both Qwen seed-777 segments, and excluding superseded/diagnostic jobs). It does not independently establish costs for excluded work.
