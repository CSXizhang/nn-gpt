# GPU-hour accounting for the September 21 external cohorts

## Scope and method

The requested 15 retained runs are interpreted as: three Qwen CIFAR-10 full-reward runs (seeds 42/123/777), three CIFAR-100 1-pattern full-reward runs, three CIFAR-100 4-pattern full-reward runs, and six unique reward-ablation runs (no-diversity-bonus and no-repeat-penalty, each at seeds 42/123/777). The three CIFAR-10 full-reward ablation rows are excluded because they duplicate the primary cohort; those job allocations must not be added again.

Allocated GPU-hours are recomputed as `elapsed_seconds * allocated_gpu_count / 3600`. This is allocated wall-time, not measured GPU utilization. Qwen seed 777 has two allocation segments (timeout plus continuation), both counted.

## Run-level evidence and results

| Requested run | Seed | Slurm job(s) | GPU allocation | State in retained record | Elapsed seconds | Allocated GPU-hours |
|---|---:|---|---|---|---:|---:|
| Qwen CIFAR-10 | 42 | 2791927 | 4 × H100 | COMPLETED | 39,125 | 43.4722222222 |
| Qwen CIFAR-10 | 123 | 2791928 | 4 × H100 | COMPLETED | 82,075 | 91.1944444444 |
| Qwen CIFAR-10 | 777 | 2791929 + 2793568 | 4 × H100 each | TIMEOUT + COMPLETED | 86,429 + 3,839 | 100.2977777778 |
| CIFAR-100 1-pattern | 42 | 2866647 | 4 × H100 | COMPLETED | 64,384 | 71.5377777778 |
| CIFAR-100 1-pattern | 123 | 2943044 | 4 × H100 | COMPLETED | 43,774 | 48.6377777778 |
| CIFAR-100 1-pattern | 777 | 2944149 | 4 × H100 | COMPLETED | 81,468 | 90.5200000000 |
| CIFAR-100 4-pattern | 42 | 2960907 + 2968439 + 3043789 (requeues included) | 4 × H100 per allocation | Preemptions, failed startup, then retained 800 rows after manual TERM | 81,596 across 6 nonzero allocations | 90.6622222222 |
| CIFAR-100 4-pattern | 123 | 2960909 + 2968441 + 3043791 (requeues included) | 4 × H100 per allocation | Preemptions, then retained 800 rows after manual TERM | 70,435 across 5 nonzero allocations | 78.2611111111 |
| CIFAR-100 4-pattern | 777 | 2960911 + 2968443 + 3043793 (requeues included) | 4 × H100 per allocation | Preemptions, then retained 800 rows after manual TERM | 66,905 across 4 nonzero allocations | 74.3388888889 |
| Ablation: no-diversity-bonus | 42 | 2697779 | 4 × H100 | COMPLETED | 61,070 | 67.8555555556 |
| Ablation: no-repeat-penalty | 42 | 2697849 | 4 × L40S | COMPLETED | 113,466 | 126.0733333333 |
| Ablation: no-diversity-bonus | 123 | 2751655 | 4 × H100 | FAILED; retained as final useful condition job | 62,404 | 69.3377777778 |
| Ablation: no-repeat-penalty | 123 | 2755237 | 4 × H100 | COMPLETED | 58,502 | 65.0022222222 |
| Ablation: no-diversity-bonus | 777 | 2760934 | 4 × H100 | COMPLETED | 71,734 | 79.7044444444 |
| Ablation: no-repeat-penalty | 777 | 2771489 | 8 × H100 | COMPLETED | 38,567 | 85.7044444444 |

The verified subtotals from the retained raw Slurm snapshot are:

| Subset | Exact GPU-hours | Rounded to 2 decimals |
|---|---:|---:|
| Qwen CIFAR-10 (all three seeds, including both seed-777 segments) | 52867/225 = 234.9644444444 | 234.96 |
| CIFAR-100 1-pattern (three seeds) | 94813/450 = 210.6955555556 | 210.70 |
| Six unique ablation jobs | 44431/90 = 493.6777777778 | 493.68 |
| Subtotal for the other 12 requested runs | 211351/225 = 939.3377777778 | 939.34 |
| CIFAR-100 4-pattern, all allocations in these three run IDs | 54734/225 = 243.2622222222 | 243.26 |
| All 15 requested runs, retained run-ID allocation scope | 1182.6000000000 | 1182.60 |

All 15 rows were refreshed from Julia2 on 2026-09-23 using `sacct --duplicates`; the other twelve match the earlier local snapshot exactly. This option is essential for the CIFAR-100 4-pattern runs: without it, `sacct` returns only the final job state and hides earlier preempted allocations under the same job ID. Each parent job/allocation is counted once; `.batch` and `.extern` steps are excluded. Zero-second cancelled/requeued records contribute zero. The 15-run retained run-ID total is **1182.60 allocated GPU-hours**. It is not an all-attempt project ledger.

## Completeness limits and omitted attempts

- The local historical raw Slurm snapshots support the Qwen, CIFAR-100 1-pattern, and six unique ablation job times above. Julia2 `sacct --duplicates` supplied the previously missing CIFAR-100 4-pattern elapsed records. The local archive documents that all three were manually terminated after their first 800 formal samples were retained; Slurm therefore labels the final allocations FAILED despite usable formal outputs.
- This is not a complete all-attempt project cost ledger. The Qwen run archive mentions canceled resume attempts `2793551` and `2793555` before additional samples were written. The CIFAR-100 4-pattern preempted/requeued allocations under `2960907/09/11` and `2968439/41/43` **are included** above because they belong to the same retained run IDs, including seconds spent on failed startup. The Qwen seed-777 timed-out allocation is included because its 760/800 segment continued as job `2793568`.
- The ablation snapshot includes `2751655`, whose final Slurm state is FAILED but which was retained after manually aligned stopping at 800 usable rows. Omitting it would undercount the retained ablation condition. Conversely, the three `full_reward` ablation rows (`2697817`, `2697840`, `2697842`) overlap the six-seed CIFAR-10 primary cohort and are excluded from this distinct 15-run scope.
- The prior `gpu_reporting.md` table used an unrelated 15-run interpretation (12 CIFAR-10 plus three CIFAR-100 one-pattern). Its 1118.36 subtotal does not answer this September 21 scope and must not be reused.

## Local source records

- `reply_20260802/08_sacct_records.csv` — per-job sacct-derived records for Qwen, CIFAR-100 1-pattern, and ablations; SHA-256 `a147af9357063c407fab097e7286113a6982c92cdf4146941ce9a65e43b4151e`.
- `final_writeup_verification_20260908/gpu_hours_recomputed.json` and `final_writeup_verification_20260908/scripts/recompute_gpu_hours.py` — exact arithmetic audit for the above snapshot.
- `nn-gpt/run_archive_index.md`, entries for `20260704_qwen_cifar10_e5_seed*` and `20260726_4pattern_a18_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed*` — run identities, resume history, retained-results scope, and C100 four-pattern job IDs/states.

Fresh Julia2 evidence snapshots: `faraz_sep21_verification/remote_sacct_c100_fourpattern_20260923.txt` and `faraz_sep21_verification/remote_sacct_other12_20260923.txt`. The first includes all requeue/preemption records for `2960907/09/11`, `2968439/41/43`, and `3043789/91/93`; its parent-job rows were normalized to `remote_sacct_c100_fourpattern_parents.csv`. All nine job IDs were associated with the three retained run roots by the Julia2 `slurm/` filenames and the run archive. The second confirms one allocation per listed job for the other twelve runs. Further abandoned or diagnostic runs outside those retained run IDs would require a separate all-attempt ledger.
