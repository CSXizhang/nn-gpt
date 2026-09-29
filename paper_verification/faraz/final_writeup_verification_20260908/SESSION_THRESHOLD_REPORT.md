# Session-history threshold verification

## Result

`threshold_sensitivity_70_80_90` is **VERIFIED** from a pre-audit session's original generator tool call and an independent rerun over the delivered raw archive.

The concentration fields are exactly:

| Report label | Primary raw field |
|---|---|
| `family_top1` | `family_hash` |
| `descriptor_top1` | `descriptor_key` |
| `graph_top1` | `graph_hash` |

Therefore the statement that graph collapse was detected in at most four of twelve runs refers to **Exact Forward-Graph (`graph_hash`)**, specifically the 70% threshold result `4/12`. It does not refer to `actual_structure_signature` (Family+Block). Here “detected” means that at least one eligible, non-overlapping 100-attempt window reached the threshold; it does not mean the run stayed above it afterward.

## Historical source

Session `019f2703-a6dc-76b2-ada2-dccb85045e27`, timestamp `2026-07-04T10:35:07.102Z`, JSONL line 3274 / ordinal 3273 contains the corrected generator. Its paired output is line 3275 / ordinal 3274. See `session_evidence/historical_generator_excerpt.md`.

An immediately preceding first pass at line 3258 used `window_metrics.csv` and omitted the minimum-signature-denominator gate. The corrected generator replaced it by reading all raw JSONLs, selecting `formal_success_candidate`, and requiring at least 20 valid values for each metric in each non-overlapping 100-attempt window. The corrected output is the relevant final logic.

## Exact method

For each of 12 runs, the generator processes eight non-overlapping attempt windows: rows 1-100, 101-200, ..., 701-800. Within each window it:

1. selects rows whose `api_result.formal_success_candidate` is true;
2. extracts the signature for each metric;
3. drops missing signatures separately for that metric;
4. ignores the metric-window if its valid-signature denominator is below 20;
5. computes dominant count divided by that metric's valid-signature denominator;
6. records the earliest `window_end` reaching 70%, 80%, or 90%.

The gate is a **per-metric valid-signature denominator**, not formally the unqualified formal-success count. In the supplied archive all 8,030 formal-success records have `graph_hash`, so the two denominators are identical for `graph_top1` in every window.

## Rerun evidence

The delivered ZIP has SHA-256 `afd723be7e0e3c47de145c4c2b764786047df05a0003a833a6d24b69939ac74f`. The rerun processed 9,600 rows across 12 runs. Among 8,030 formal-success records:

- `graph_hash`: 8,030 uses;
- fallback to `signature`: 0 uses;
- fallback to `actual_structure_signature`: 0 uses.

The independently generated CSV matches every historical condition/seed/role/row-count/onset value. The only intentional difference is the abbreviated run label used by the audit script.

| Threshold | Exact Forward-Graph reached |
|---:|---:|
| 70% | 4/12 |
| 80% | 3/12 |
| 90% | 2/12 |

The historical denominator is all 12 attempted runs. Four-pattern seed 114 had only four formal successes in one window and zero eligible graph windows, so its `never` means **not evaluable at this minimum-denominator rule**, not evidence that it avoided collapse. Among the 11 runs with at least one eligible exact-graph window, the corresponding reached counts are 4/11, 3/11, and 2/11. This 11-run view is a caveat, not a replacement for the historical 12-run table.

## Terminology consequence

The safe paper terminology is:

- `actual_structure_signature`: **Family+Block Eff./Top-1**
- `graph_hash`: **Exact Forward-Graph Eff./Top-1**

Calling both fields “Graph” creates a real terminology conflict. The threshold statement can safely say: “Exact forward-graph concentration reached the 70%, 80%, and 90% top-1 thresholds in 4/12, 3/12, and 2/12 runs, respectively.”

## Remaining limitation

The historical generator was inline Python inside a Codex `exec_command`, not a standalone script. It wrote CSV/Markdown to `/Users/zhangxi/Desktop/SFT_RL/paper_used_data_check_20260704/01_processed_results/article_diagnostics_20260627/`. The inline source and paired output survive in the raw session JSONL, and the audit reconstruction reproduces the historical table exactly from the delivered raw archive.
