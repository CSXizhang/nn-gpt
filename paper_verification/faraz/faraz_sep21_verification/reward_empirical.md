# Empirical reward verification

## Scope and method

I scanned the twelve `generation_samples.jsonl` files under
`article_raw_data_archive_20260704/01_six_seed_robustness/` in
`/Users/zhangxi/code/RL/faraz_delivery_20260726/article_raw_data_archive_20260718_resend_20260726.zip`.
The archive contains 8,030 rows with `api_result.formal_success_candidate == true`
(all twelve files contributed). For each logged top-level `api_result` field whose
name starts with `r_`, the table below reports statistics over those 8,030 rows.
Values are the logged component fields; no missing or non-numeric values were
silently converted to zero. The archive SHA-256 is
`afd723be7e0e3c47de145c4c2b764786047df05a0003a833a6d24b69939ac74f`.

## Logged component distribution

`nonzero` counts values unequal to exactly zero. Every field below had 8,030
numeric observations.

| field | min | median | max | nonzero |
|---|---:|---:|---:|---:|
| `r_batch_elite` | 0 | 0 | 0.04 | 2,256 |
| `r_best_backbone_group` | -0.30 | 0 | 0.30 | 4,562 |
| `r_best_group` | -0.24 | 0.0017066865 | 0.1822016903 | 6,079 |
| `r_block_diversity` | -0.13 | -0.0200000000 | 0.14 | 5,800 |
| `r_cnn_diversity` | -0.35 | -0.1100000000 | 0.32 | 7,022 |
| `r_dense` | 0 | 0.0915164278 | 0.0977318167 | 7,239 |
| `r_descriptor_diversity` | -0.155 | -0.1050000000 | 0.19 | 7,904 |
| `r_formal_success_signal` | 0 | 0.28 | 0.28 | 7,239 |
| `r_generalization` | -0.20 | 0 | 0 | 3,288 |
| `r_goal_best` | 0 | 0 | 0.056 | 244 |
| `r_goal_match` | 0 | 0 | 0 | 0 |
| `r_history_context` | 0 | 0 | 0 | 0 |
| `r_length_compactness` | -0.0346666667 | 0 | 0 | 248 |
| `r_no_progress_penalty` | -0.03 | 0 | 0 | 2,839 |
| `r_plain_fuse_penalty` | -0.308 | -0.1100000000 | 0 | 5,637 |
| `r_prev_backbone_group` | -0.45 | 0 | 0.45 | 4,485 |
| `r_prev_group` | -0.36 | 0.0008717044 | 0.1866512420 | 6,079 |
| `r_repeat_family` | -0.055 | -0.0550000000 | 0 | 5,191 |
| `r_repeated_line_penalty` | -0.15 | 0 | 0 | 105 |
| `r_structure_archive` | 0 | 0 | 0.098 | 3,262 |
| `r_structure_group` | 0 | 0 | 0.196 | 3,262 |
| `r_target_structure_penalty` | -1.0 | 0 | 0 | 1,324 |
| `r_template_penalty` | -0.05 | 0 | 0 | 815 |
| `r_trainset_novelty` | 0 | 0.02 | 0.04 | 6,210 |

## Fixed-commit equation and Stage 2 constants

The source object `c91714dbe7dad1d02a9080243945bbf8e8ec9300` resolves as a Git
commit. The Stage 2 formal branch in `ab/gpt/TuneRL.py` computes, when a formal
epoch and reward target are available:

```text
r_dense = 0.50 * clip(0.03 + 0.28 * reward_target_value
                       + 0.04 * max(0, train_acc - 0.50), 0.02, 0.35)
r_formal_success_signal = 1[formal_success_candidate] *
    (0.08 + 0.20 * 1[target_structure_match is not False])
r_prev_group = 0.20 * clip(10*(target - (global_baseline + 0.003)), -1.8, 1.8)
    * 0.20 when a backbone baseline is present
r_best_group = 0.20 * clip(12*(target - (best_global + 0.0015)), -1.2, 1.2)
    * 0.20 when a backbone baseline is present
r_prev_backbone_group = 0.25 * clip(10*(target - (backbone_baseline + 0.003)), -1.8, 1.8)
r_best_backbone_group = 0.25 * clip(12*(target - (best_backbone + 0.0015)), -1.2, 1.2)
r_goal_best = 0.70 * 0.08 when the goal refresh condition is met
r_goal_match = 0.85 * 0.12 * goal_tag_hit_rate
r_no_progress_penalty = 0.50 * (-0.06) = -0.03 when the effective previous target is not beaten
r_generalization = clip(-2.0 * max(0, frozen_train_acc - frozen_test_acc - 0.02), -0.20, 0)
```

The remaining Stage 2/3 structural terms are added with these Stage 2 scales:
`structure_scale=1.40`, `repeat_family_scale=1.10`,
`plain_fuse_scale=1.10`; positive novelty is quality-gated at accuracy `0.90`.
The raw constants include descriptor `+0.03/+0.02/+0.08`, CNN
`+0.07/+0.05/+0.12`, block `+0.06/+0.08`, and repeated-block conditional cap
`2.0` before local competition. Stage 2 non-improvement and descriptor caps are
also `2.0`.

The base reward is the clip of the sum of the Stage 2 logged terms and
`r_goal_match`:

```text
r_primary = r_dense + r_formal_success_signal
  + r_prev_group + r_best_group + r_prev_backbone_group + r_best_backbone_group
  + r_goal_best + r_generalization + r_structure_group + r_structure_archive
  + r_descriptor_diversity + r_cnn_diversity + r_block_diversity + r_batch_elite
  + r_repeat_family + r_plain_fuse_penalty + r_target_structure_penalty
  + r_template_penalty + r_history_context + r_no_progress_penalty
base_reward = clip(r_primary + r_goal_match, -2, 2)
```

After this base clip, `TuneRL.py` applies non-improvement/repetition caps, the
repeated-block cap, local competition, executability and target-structure
clamps. `TuneRLSft.py` then adds extraction/contract deltas and compactness
terms and can apply format/hygiene and dual-backbone caps. Thus `[-2, 2]` is the
base TuneRL clip, not a universal range for the final logged `reward`.

## Zero-field checks and completeness

For all 8,030 selected rows, `prompt_goal_tags` is `None`, and
`goal_tag_total_count`, `goal_tag_hit_count`, and `goal_tag_hit_rate` are all
zero. Therefore `r_goal_match == 0` is explained by the recorded input and the
source equation, rather than a missing log field. The source also defines
`_history_context_reward(...)` as an unconditional `return 0.0` at this commit;
`r_history_context == 0` is implementation-inert, not an extraction gap.

All 24 logged `r_*` fields listed above are present and numeric on every selected
row. `r_primary` and `r_tiebreak` are not top-level fields in these JSONL rows;
the serialized `open_discovery` payload contains them for the corresponding
calculation. `r_trainset_novelty` is logged but is not included in the Stage 2
`r_primary` sum shown above. This is a source behavior, not an omitted value in
the empirical scan.

## Caveat

The six-seed archive rows record the implementation commit in run metadata, but
some run configurations also report a dirty working tree. This report verifies
the fixed commit's equation and the archived logged fields; it does not recover
any uncommitted diff that may have been present during a run.
