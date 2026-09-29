# Faraz follow-up: external family-onset validation

Date: 2026-09-23

## Definition reproduced

For the directly comparable cohort-level results, I reran family onset on raw JSONL trajectories using the recovered threshold audit convention: eight **non-overlapping 100-attempt windows** (1–100, 101–200, …, 701–800). A row contributes only when `api_result.formal_success_candidate == true`; family identity is `family_hash`, with fallback to `family_id`; at least 20 formal-success rows must have a nonempty family identity in the window; and onset is the first window with family top-1 >=80%. The 20-row gate is explicit in the historical generators. It is a count of formal-success rows with family signatures, not a graph-validity gate. A first qualifying window is an onset only; it does not imply persistence.

The earlier `collapse_reward_component_analysis.py` also reports a sensitivity using overlapping windows (100 attempts, step 25). I keep that result separate below because it is not comparable to the primary cumulative-by-100/200/400 counts.

## Recomputed external cohorts

| Cohort | Seed | Attempts | Formal successes | First qualifying window (attempts, inclusive) | Formal successes with family in that window | Family top-1 |
|---|---:|---:|---:|---|---:|---:|
| Qwen CIFAR-10 | 42 | 800 | 781 | 1–100 | 93 | 95.70% |
| Qwen CIFAR-10 | 123 | 800 | 737 | 1–100 | 63 | 82.54% |
| Qwen CIFAR-10 | 777 | 800 | 723 | 1–100 | 62 | 88.71% |
| CIFAR-100 one-pattern | 42 | 800 | 762 | 1–100 | 85 | 98.82% |
| CIFAR-100 one-pattern | 123 rerun | 800 | 772 | 1–100 | 94 | 97.87% |
| CIFAR-100 one-pattern | 777 | 800 | 735 | 1–100 | 95 | 96.84% |
| CIFAR-100 four-pattern | 42 | 800 | 699 | No qualifying non-overlapping window | — | Maximum non-overlapping eligible-window share: 62.96% |
| CIFAR-100 four-pattern | 123 | 800 | 671 | No qualifying window | — | Maximum non-overlapping eligible-window share: 62.20% |
| CIFAR-100 four-pattern | 777 | 800 | 726 | No qualifying window | — | Maximum non-overlapping eligible-window share: 72.16% |

Under the non-overlapping definition, 6/9 external runs have an onset: Qwen CIFAR-10 3/3 and CIFAR-100 one-pattern 3/3; CIFAR-100 four-pattern 0/3. For CIFAR-100 four-pattern seed 42, the overlapping-window sensitivity finds an onset at attempts 51–150, but neither adjacent non-overlapping window reaches 80%. The window convention therefore changes this run's binary onset classification. The result is onset-in-an-eligible-window, not persistent collapse.

## Six non-full-reward ablation runs

| Ablation | Seed | Attempts | Formal successes | First qualifying window (attempts, inclusive) | Formal successes with family in that window | Family top-1 |
|---|---:|---:|---:|---|---:|---:|
| No diversity bonus | 42 | 800 | 766 | No qualifying window | — | Maximum non-overlapping eligible-window share: 58.59% |
| No diversity bonus | 123 | 800 | 778 | 101–200 | 97 | 85.57% |
| No diversity bonus | 777 | 800 | 766 | No qualifying window | — | Maximum non-overlapping eligible-window share: 70.10% |
| No repeat penalty | 42 | 800 | 715 | 401–500 | 88 | 89.77% |
| No repeat penalty | 123 | 800 | 791 | 101–200 | 100 | 91.00% |
| No repeat penalty | 777 | 800 | 765 | No qualifying window | — | Maximum non-overlapping eligible-window share: 75.56% |

Three of six ablation runs have a qualifying non-overlapping onset window. These six are the unique no-diversity-bonus and no-repeat-penalty runs for seeds 42/123/777; the full-reward controls are not counted here. The overlapping-window sensitivity leaves the same 3/6 onset count, with seed 123 first qualifying at attempts 76–175 rather than 101–200.

## Input provenance and coverage

The Qwen CIFAR-10 and six ablation trajectories were read directly from the local archive `faraz_delivery_20260726/article_raw_data_archive_20260718_resend_20260726.zip` (SHA-256 `afd723be7e0e3c47de145c4c2b764786047df05a0003a833a6d24b69939ac74f`), members under:

- `article_raw_data_archive_20260704/10_qwen_second_base_model_3seeds/seed{42,123,777}/generation_samples.jsonl`
- `article_raw_data_archive_20260704/02_reward_ablation/{no_diversity_bonus,no_repeat_penalty}_seed{42,123,777}/generation_samples.jsonl`

CIFAR-100 one-pattern trajectories were read from `faraz_delivery_20260726/cifar100_provenance/{seed42,seed123_rerun,seed777}/generation_samples.jsonl`; four-pattern trajectories were read from `faraz_followup_delivery_20260808/cifar100_four_pattern/raw/seed{42,123,777}/generation_samples.jsonl`.

The trajectory recomputation uses local archived raw files, all 15 of which are present. Julia2 was initially unreachable through the default resolver; a later direct-IP read-only session supplied the missing Slurm accounting records described below. The server was not used as an alternative source for the trajectory rows.

## Keep the graph metric separate

The earlier exact-forward-graph results use `graph_hash` and non-overlapping windows from a separately recovered historical generator. They are not the family-onset values above. For the CIFAR-100 four-pattern seed-42 trajectory, the exact-graph non-overlapping-window result remains “never” while the family sliding-window result is 51–150. Those answers differ because the signature field and window convention differ; neither should overwrite the other.

The primary 9/12 is verified as an ever-crossed onset count under non-overlapping windows. The 458/907 ratio is verified only as a modelled counterfactual allocation tail under the proportional-per-attempt assumption detailed below; it is not recorded realized savings.

## Primary CIFAR-10 in-sample onset and terminal-window check

I separately reran the 12-run, six-seed CIFAR-10 robustness cohort (seeds 42/123/777/114/514/919, one- and four-pattern) from the same raw archive. For the cumulative onset counts below, I used the recovered threshold audit's **eight non-overlapping 100-attempt windows** (1–100, 101–200, …, 701–800), with `formal_success_candidate`, `family_hash` (family-id fallback), at least 20 family-bearing formal successes, and family top-1 >=80%.

- First onset was present in 9/12 runs. Cumulative first-onset counts were 5/9 by attempt 100, 7/9 by 200, and 9/9 by 400. The three runs with no qualifying window were four-pattern seeds 123 and 777 (eligible windows but below threshold), and four-pattern seed 114 (no eligible family denominator).
- This is an **ever-crossed onset** result, not an “ending family-collapsed” result. On the last 100 attempted rows, only 8/12 runs had family top-1 >=80%: one-pattern seeds 42/123/114/514/919 and four-pattern seeds 42/514/919. One-pattern seed 777 did cross in the first window but ended at 55.56%, so it is a recovery/discordant case. Four-pattern seeds 123/777 ended below threshold; four-pattern seed 114 had no family-bearing formal-success rows in its final window. Thus the accurate statement is 9/12 ever crossed, 8/12 terminally collapsed, with one onset followed by a diverse final window; it is inaccurate to equate 9/12 onsets with 9/12 ending collapsed.

### External and ablation onset versus terminal state

For this cross-tab, “onset” uses the non-overlapping 100-attempt family procedure above; “terminally collapsed” means family top-1 >=80% among family-bearing formal successes in the final 100 attempted rows, requiring at least 20 such rows. The 15 runs are the nine external runs and six ablations reported above.

| | Terminally collapsed | Terminally diverse | Final window not evaluable | Total |
|---|---:|---:|---:|---:|
| At least one family-onset window | 8 | 1 | 0 | 9 |
| No family-onset window | 0 | 6 | 0 | 6 |
| Total | 8 | 7 | 0 | 15 |

Under the common non-overlapping rule, eight onset-positive runs are terminally collapsed, one onset-positive run is terminally diverse, and all six onset-negative runs are terminally diverse. The false positive is Qwen CIFAR-10 seed 777: it crosses at 1–100 (88.71%) but ends at 76.77%. The 80% rule therefore yields 8 true positives, 1 false positive, 6 true negatives, and 0 false negatives on these 15 archived runs. The overlapping-window sensitivity adds CIFAR-100 four-pattern seed 42 as another false positive (onset at 51–150, final top-1 30.77%). This is a descriptive cross-tab, not independent predictive validation.

| Run | First non-overlapping onset window end | Final-window family top-1 | Terminal label at 80% |
|---|---:|---:|---|
| Qwen CIFAR-10 seed 42 | 100 | 100.00% | Collapsed |
| Qwen CIFAR-10 seed 123 | 100 | 100.00% | Collapsed |
| Qwen CIFAR-10 seed 777 | 100 | 76.77% | Diverse |
| CIFAR-100 one-pattern seed 42 | 100 | 95.88% | Collapsed |
| CIFAR-100 one-pattern seed 123 rerun | 100 | 100.00% | Collapsed |
| CIFAR-100 one-pattern seed 777 | 100 | 91.49% | Collapsed |
| CIFAR-100 four-pattern seed 42 | None | 30.77% | Diverse |
| CIFAR-100 four-pattern seed 123 | None | 42.11% | Diverse |
| CIFAR-100 four-pattern seed 777 | None | 49.49% | Diverse |
| No-diversity-bonus seed 42 | None | 58.59% | Diverse |
| No-diversity-bonus seed 123 | 200 | 98.99% | Collapsed |
| No-diversity-bonus seed 777 | None | 40.82% | Diverse |
| No-repeat-penalty seed 42 | 500 | 100.00% | Collapsed |
| No-repeat-penalty seed 123 | 200 | 100.00% | Collapsed |
| No-repeat-penalty seed 777 | None | 39.58% | Diverse |

“None” means no eligible non-overlapping window reached 80%; all 15 final windows had at least 20 family-bearing formal successes.

### The 458/907 GPU-hour figure

The primary Slurm snapshot is `delivery_20260727/06_gpu_hours/sacct_six_seed_cifar10_rl_20260728.csv`; the audited total is 907.6666667 allocated GPU-hours. Using the first-onset ends in the non-overlapping windows above, I recomputed a **counterfactual tail-allocation estimate** for the nine onset-positive jobs:

`estimated hours avoidable after onset = job allocated GPU-hours × (800 − first-onset window end) / 800`.

The sum is exactly 3,301,139/7,200 = 458.4915278 GPU-hours, or 458.49 rounded; relative to 907.6666667, this is 50.513%. This reproduces “about 458 of 907 GPU-hours” only under a uniform allocation-per-attempt assumption. It is not observed savings: the jobs ran to completion, and the retained Slurm records contain whole-job elapsed times rather than per-attempt GPU allocation. Report it as an estimated avoidable tail, not as GPU-hours actually saved. The estimate also applies to the nine ever-onset runs even though one of them (one-pattern seed 777) later diversified, so it should not be described as stopping only terminally collapsed runs.


### External-cohort counterfactual tail

Using the same proportional-per-attempt formula for the nine non-overlapping-onset-positive external/ablation runs, and the retained per-job GPU hours in `gpu_reporting.md`, the modelled tail is 537.985 GPU-hours (537.99 rounded). Fresh Julia2 `sacct --duplicates` records now establish 1,182.60 allocated GPU-hours across all 15 retained run IDs, including the three CIFAR-100 four-pattern runs and their preempted/requeued allocations. The modelled tail is therefore 45.49% of that retained-run total. Those three added runs have no non-overlapping onset and add nothing to the numerator. This remains a counterfactual estimate under uniform GPU-hours per attempt, not realized savings or an all-attempt project accounting.
