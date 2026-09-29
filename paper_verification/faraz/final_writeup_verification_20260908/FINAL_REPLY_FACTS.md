# Facts safe to reply to Faraz

## 1. SFT structural comparison

- SFT-only verified scope: one-pattern and four-pattern, seeds 42/123/777, 400 attempts each, 2,400 total; five-epoch CIFAR-10 held-out-test evaluation.
- Formal-success counts are one-pattern 367/367/353 and four-pattern 398/397/399.
- SFT-only family top-1 / effective number:
  - one-pattern 42: 96.730245% / 1.154881; 123: 96.457766% / 1.187794; 777: 96.317280% / 1.194185.
  - four-pattern 42: 30.150754% / 3.939967; 123: 30.730479% / 4.082227; 777: 29.824561% / 4.095900.
- SFT-only Family+Block effective number: one-pattern 15.397439/13.744157/16.750113; four-pattern 243.416870/243.716478/242.913701.
- Post-RL final-file-row-100 family top-1 / effective number:
  - one-pattern 42: 100% / 1.0; 123: 100% / 1.0; 777: 55.555556% / 2.188345.
  - four-pattern 42: 100% / 1.0; 123: 40.625% / 4.326783; 777: 59.183673% / 2.640359.
- Interpretation: one-pattern concentration is already present in the SFT-only samples and is maintained in two RL windows, while seed777 diversifies. Four-pattern begins less concentrated; dominant-family share rises after RL in all three seeds, but seed123's family effective number rises slightly, so uniform entropy collapse is false.
- Split caveat: signatures are code-derived, but SFT uses held-out-test and post-RL gating uses reward-eval. Formal-success conditioning makes the analyzed populations non-identical. This is descriptive, not a causal RL estimate.

## 2. 30.30%

- **`30.30% = STRUCTURAL CONCENTRATION`.** The original one-epoch independent audit's named field is CNN-signature top-1: 30/99 formal-success candidates = 30.3030303%, from 100 attempts. The archived summary path is `sft_only_fourpattern_current.cnn.top1_share`; its baseline table labels the column “CNN top-1 share.”
- It is not accuracy, Family+Block, or exact forward graph. In that artifact, Family+Block is 2/99 and exact graph is 3/99. `family_hash` happens to equal 30/99 there.
- New five-epoch four-pattern runs reproduce the same CNN top-1 statistic: 30.1508%, 30.7305%, and 29.8246%.
- Xi's August 8 “does not persist” wording is wrong because it compares structural concentration with an accuracy difference.

## 3. Reward table

- The current numerical manuscript table is incorrect as a whole at commit `c91714dbe7dad1d02a9080243945bbf8e8ec9300`.
- Correct source values/behavior: stage2 group prev/best `0.20/0.20`; backbone-group `0.25/0.25`; dense scale `0.50` and inner clip `[0.02,0.35]`; formal success `0.08` plus `0.20` when `target_structure_match is not False`; block archive novelty `0.08`; conditional pre-local-competition repeated-block upper cap `2.0`.
- The TuneRL base sum is clipped to `[-2,2]`; later TuneRLSft wrapper caps/deltas and compactness can make final logged reward lower than -2.
- Raw logs exactly verify `r_formal_success_signal=0.28`, `r_dense=0.09515141111111111`, `r_repeat_family=-0.05500000000000001`, and `r_plain_fuse_penalty=-0.11000000000000001` under their source formulas.
- Use separate Raw configuration and Effective/logged behavior columns; do not mix them in one scalar value column.
- The audited numerical table is the local `/Users/zhangxi/Desktop/example-cvpr/Paper_draft_en.tex` copy; whether it is the submission-final manuscript source is not verified.

## 4. GPU hours

- Definition: allocated GPU-hours = elapsed wall time × allocated GPU count; this is not utilization time.
- Primary cohort: one-pattern 374.02777778 (374.03), four-pattern 533.63888889 (533.64), total 907.66666667 (907.67). Hardware: H100 553.72333333 (553.72), L40S 353.94333333 (353.94).
- Extended components: unique ablation addition 493.67777778; Qwen 234.96444444; CIFAR-100 one-pattern 210.69555556; top20 proxy 19.62527778; corrected successful wide38 19.8175.
- Julia2 evidence identifies missing task 0 as `2968490_0`, COMPLETED, 1 L40S × 1,920 seconds = 0.5333333333 hours, matching manifest selection/result candidate `4pattern_seed42_743`.
- Corrected deduplicated total for the previously declared extended cohorts, with successful proxy evaluations, is 1886.44722222, rounded once to **1886.45**. The old 1885.91/1885.92 values are respectively exact-sum and rounded-component results for the incomplete local 37-row snapshot; both omit task 0.
- Failed superseded array `2968450` is excluded consistently with the successful-proxy sub-scope. Do not silently broaden this total into all attempted/failed GPU cost.

## 5. Threshold field

- `actual_structure_signature` means Family+Block; `graph_hash` means Exact Forward-Graph. Use those labels explicitly.
- Recovered session-history generator and independent raw rerun verify that `threshold_sensitivity_70_80_90` uses `graph_hash`. All 8,030 formal-success rows have `graph_hash`; fallback use is zero.
- Exact Forward-Graph top-1 first reaches 70%, 80%, and 90% in at least one eligible non-overlapping 100-attempt window in 4/12, 3/12, and 2/12 runs, respectively.
- This is onset, not continuous/persistent collapse. Four-pattern seed114 has no eligible window because it has only four formal successes; 11 runs have eligible graph windows, while the historical denominator remains all 12 runs.

## 6. SFT-only limitations

- Verified completed scope is three seeds per condition × two conditions × 400 attempts = 2,400, five epochs, CIFAR-10 `trainvaltest`, split seed 42, held-out-test role.
- Safe limitation: “The reduced experiment directly characterizes the pre-RL state for the three completed seeds, but provides less replication and statistical coverage than the planned six-seed design.”
- **DO NOT CLAIM YET:** the raw-evidence reason for reducing the design, or that reduction “does not change any claim.” The available reason is delivery/correspondence prose.

## 7. Imagenette

- Rule-constrained sampler Imagenette exists and is verified: 99 formal successes from 100 attempts, one epoch, held-out test.
- Historical learned one-pattern run `20260605_0925_dscoder_imagenette_h100` is directly verified remotely: completed Slurm job `2666209`, 1,000 raw rows, one-pattern Imagenette config, one formal epoch.
- Safe statement: “Historical learned one-pattern Imagenette runs are present on Julia2. No completed matched learned one-/four-pattern Imagenette conditions were found in the inspected Julia2 archive.”
- **DO NOT CLAIM YET:** that no additional run is planned before submission; this requires manual confirmation.

## 8. Code availability

- Canonical NNGPT repository: `https://github.com/ABrain-One/nn-gpt.git`. Fixed commit `c91714dbe7dad1d02a9080243945bbf8e8ec9300` exists and contains the audited full-reward source.
- All twelve cohort configs record five-epoch `full_reward` stage2 mode. Six name that commit but have `dirty=true`; the six previously blank-config runs have matching remote `run_state.json` records naming the same base commit. No source snapshot/diff proves clean byte identity. Safe wording: the cohort is archived against the base commit; exact clean-tree provenance is incomplete.
- NN Dataset repository identity is verified, but its historical experiment commit is **NOT VERIFIED**.

## 9. Direct consistency corrections

- SFT raw recomputation supports the displayed 88.07% one-pattern, 87.00% four-pattern, and -1.07 pp seed-mean accuracy difference, with seed-mean and pooled aggregation kept distinct.
- The completed four-pattern CIFAR-100 package has 2,400 attempts, 2,096 formal successes, and 2,060 positive rewards. Reusing `recompute_final_summaries.py:40–48` exactly (`api_result.test_acc` first) gives seed mean 69.57372212% and pooled 69.64405534%, matching `FINAL_RESULTS.md` at 69.57% and 69.64%. Every formal-success row has numeric `test_acc`; fallback use is zero. Some recorded horizon-5 values differ, so they must not silently replace the delivery field.
