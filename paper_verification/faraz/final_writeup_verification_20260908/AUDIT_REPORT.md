# Final write-up verification audit — 2026-09-08

> Updated 2026-09-09 with read-only Julia2 evidence. Local-only intermediate totals remain labelled as such; remote-recovered evidence controls the current GPU, provenance, and Imagenette conclusions.

| Item | Status | Result |
| ----------------------- | ------------------------------ | ------ |
| 1 SFT family comparison | VERIFIED | Six SFT-only and six post-RL raw logs recomputed; Faraz's per-seed family top-1 hypotheses match, with causal and split caveats. |
| 2 30.30% meaning | VERIFIED | Original raw audit proves 30.30% is CNN-signature structural top-1: 30/99 formal successes; new five-epoch runs reproduce it. |
| 3 Reward table | DISPROVED | The current numerical table mixes raw/effective values and contains multiple fixed-commit errors. |
| 4 GPU hours | VERIFIED | Recovered wide38 task 0 gives complete successful-proxy wide38 19.8175 and corrected declared-cohort total 1886.45. |
| 5 Threshold field | VERIFIED | Recovered historical inline generator and raw rerun prove `graph_hash`; threshold onsets are 4/12, 3/12, and 2/12. |
| 6 Limitations facts | PARTIAL | Actual 3×2×400, five-epoch scope is verified; the reason for reduction is supported only by delivery prose. |
| 7 Imagenette | PARTIAL | Sampler 99/100 and a historical completed learned one-pattern 1,000-row run are verified; no matched learned pair was found. |
| 8 Code availability | PARTIAL | Canonical NNGPT URL and base commit are verified for all 12 via configs/run states; clean-tree and NN Dataset commit remain unverified. |

Evidence priority throughout this report is raw JSONL/configuration/sacct records and fixed git objects. Manuscript, email, and Markdown statements are treated as claims to check.

All relative reproduction commands below are run with working directory `/Users/zhangxi/code/RL`.

## 1. SFT-only family-level structural comparison

### Claim being checked

Whether the six SFT-only runs comprise two conditions × seeds 42/123/777 × 400 attempts; whether one-pattern is already family-concentrated before RL; and whether the matched-seed final RL windows support Faraz's quoted family shares.

### Evidence used

Six raw files and configs under `faraz_followup_delivery_20260808/sft_only/raw/`; six post-RL JSONL/config pairs in `faraz_delivery_20260726/article_raw_data_archive_20260718_resend_20260726.zip`; the delivered formal filter `recompute_final_summaries.py` (`formal_success_candidate is True`); the delivered entropy implementation in `delivery_20260727/05_canonicalisation_granularity/analyze_canonicalisation_sweep.py`; and the local manuscript definition of the final 100 attempted samples. Input hashes are recorded in `sft_family_recompute.json`.

### Recompute method

The SFT window is all 400 file rows. The post-RL window is the final 100 file-order rows of each 800-row file, then formal-success filtering. For counts `n_i`, `p_i=n_i/N`, `H=-Σp_i ln(p_i)`, effective number `exp(H)`. The script cross-checks archived structural summaries and also computes a matched last-100 SFT sensitivity.

Reproduction:

```bash
python3 final_writeup_verification_20260908/scripts/recompute_sft_family.py
```

### Exact result

Shares use formal-success N as denominator.

| Condition | Seed | Stage | Attempted window | Formal-success N | Family top-1 | Family eff. | Family+Block top-1 | Family+Block eff. |
|---|---:|---|---|---:|---:|---:|---:|---:|
| one-pattern | 42 | SFT-only | all 400 | 367 | 355/367 = 96.73024523160763% | 1.1548812863780262 | 98/367 = 26.70299727520436% | 15.397438709758429 |
| one-pattern | 123 | SFT-only | all 400 | 367 | 354/367 = 96.45776566757494% | 1.187793833210298 | 104/367 = 28.337874659400547% | 13.744157363252462 |
| one-pattern | 777 | SFT-only | all 400 | 353 | 340/353 = 96.31728045325779% | 1.1941853171747545 | 90/353 = 25.4957507082153% | 16.75011305494363 |
| one-pattern | 42 | post-RL | file rows 701–800 | 91 | 91/91 = 100% | 1.0 | 8/91 = 8.791208791208792% | 52.47933267818359 |
| one-pattern | 123 | post-RL | file rows 701–800 | 100 | 100/100 = 100% | 1.0 | 32/100 = 32% | 4.362209961779034 |
| one-pattern | 777 | post-RL | file rows 701–800 | 99 | 55/99 = 55.55555555555556% | 2.188345144133069 | 13/99 = 13.131313131313133% | 30.824650727487168 |
| four-pattern | 42 | SFT-only | all 400 | 398 | 120/398 = 30.15075376884422% | 3.939966836842404 | 6/398 = 1.507537688442211% | 243.41686990881595 |
| four-pattern | 123 | SFT-only | all 400 | 397 | 122/397 = 30.730478589420656% | 4.082227400852246 | 9/397 = 2.2670025188916875% | 243.71647822509496 |
| four-pattern | 777 | SFT-only | all 400 | 399 | 119/399 = 29.82456140350877% | 4.095900379104709 | 5/399 = 1.2531328320802004% | 242.91370059655745 |
| four-pattern | 42 | post-RL | file rows 701–800 | 100 | 100/100 = 100% | 1.0 | 6/100 = 6% | 47.95020870750575 |
| four-pattern | 123 | post-RL | file rows 701–800 | 96 | 39/96 = 40.625% | 4.326782981703946 | 6/96 = 6.25% | 54.157351678628565 |
| four-pattern | 777 | post-RL | file rows 701–800 | 98 | 58/98 = 59.18367346938775% | 2.6403591353610456 | 10/98 = 10.204081632653061% | 24.509458456496066 |

One-pattern is already family-concentrated at SFT-only. Four-pattern SFT-only is substantially more diverse at family level. Post-RL top-1 concentration increases for four-pattern in all three seeds, but family effective number rises slightly for seed123. One-pattern seed777 diversifies post-RL. The full signature cross-check, entropy, unique counts, hashes, and matched-window sensitivity are in `structural_findings.md` and `sft_family_recompute.json`.

SFT-only formal evaluation uses `heldout_test` (10k); post-RL gating uses `reward_eval` (5k), with the same `trainvaltest` partition protocol and split seed. Structural signatures are code-derived, but the included population is conditioned on formal success, which depends on evaluation/execution completion.

### Difference from Faraz / previous Xi summary

Faraz's listed per-seed top-1 hypotheses are numerically supported. Any claim that one-pattern concentration arose wholly from RL is wrong. Any claim that all seeds show entropy collapse is also wrong: one-pattern seed777 diversifies and four-pattern seed123's family effective number increases slightly.

### Safe statement for the paper/email

“One-pattern family concentration is already present in the SFT-only samples and is maintained in two final RL windows, while seed 777 partially diversifies. Four-pattern SFT-only remains much less family-concentrated; final RL windows show higher dominant-family shares in all three seeds, most sharply for seeds 42 and 777. These observational comparisons do not isolate a causal RL effect. The signatures are split-independent, but the formal-success-conditioned candidate populations are not strictly identical.”

### Remaining uncertainty

Post-RL logs lack candidate IDs, so the file-order last-100 definition is reproduced but resume-boundary indices cannot be independently reconstructed. Unequal 400/100 windows constrain entropy comparison; matched-window sensitivity mitigates but does not create a causal counterfactual.

## 2. Meaning of 30.30%

### Claim being checked

Whether 30.30% is performance or structural concentration, its exact field/denominator, and whether the new four-pattern SFT-only data reproduce it.

### Evidence used

The ZIP `faraz_delivery_20260726/article_raw_data_archive_20260718_resend_20260726.zip`, member tree `03_independent_generation_audit/sft_only_heldout_test/` containing raw JSONL/config/baseline/structural summary; `FINAL_RESULTS.md`; the older manuscript tables; and the newly recomputed raw SFT-only logs.

### Recompute method

For the original and each four-pattern five-epoch SFT run, filter `formal_success_candidate is true` and calculate dominant shares separately for `family_hash`, `cnn_signature`, `actual_structure_signature`, and `graph_hash`.

### Exact result

The original one-epoch four-pattern audit has 100 attempts and 99 formal successes. Its delivered table labels the value CNN top-1; raw recomputation gives dominant `cnn_signature` count 30, exactly `30/99 = 30.303030303030304%`. `family_hash` has the same 30/99 distribution in this artifact. By contrast, `graph_hash` is 3/99 = 3.0303% and `actual_structure_signature` is 2/99 = 2.0202%. Therefore **`30.30% = STRUCTURAL CONCENTRATION`**, specifically CNN-signature top-1 among formal-success candidates, not accuracy.

The new data reproduce the same `cnn_signature` statistic: 120/398 = 30.15075376884422%, 122/397 = 30.730478589420656%, and 119/399 = 29.82456140350877%. Their Family+Block shares are only 1.5075%, 2.2670%, and 1.2531%.

### Difference from Faraz / previous Xi summary

`FINAL_RESULTS.md` connects an accuracy delta to “30.30% ... does not persist.” The raw original artifact proves this is a category error, and the new same-field values also show that the structural concentration does persist approximately.

### Safe statement for the paper/email

“The original one-epoch independent four-pattern SFT audit's 30.30% is the dominant CNN-signature share: 30 of 99 formal-success candidates. The five-epoch four-pattern SFT-only runs obtain 30.1508%, 30.7305%, and 29.8246% for the same statistic. This is structural concentration, not performance; the August 8 ‘does not persist’ wording is incorrect.”

### Remaining uncertainty

The original artifact is verified. Its `family_hash` happens to share the same 30/99 distribution, but the delivered table's exact named field is `cnn_signature`; do not generalize equality between the two keys to other runs.

## 3. Reward-components table at fixed commit

### Claim being checked

Whether `tab:reward_components` accurately represents the five-epoch `full_reward` implementation at `c91714dbe7dad1d02a9080243945bbf8e8ec9300`.

### Evidence used

`git cat-file -t`; only `git show <commit>:ab/gpt/TuneRL.py` and `TuneRLSft.py`; `/Users/zhangxi/Desktop/example-cvpr/Paper_draft_en.tex:151–173`; raw candidates for logged-value spot checks; and independent QA in `reward_qa.md`.

### Recompute method

The repeatable script parses constants from fixed-commit source, traces stage profiles and wrapper ordering, searches raw JSON records, and recomputes the dense value:

```bash
python3 final_writeup_verification_20260908/scripts/rebuild_reward_audit.py
```

### Exact result

| Component | Fixed-commit raw/source | Effective/logged behavior | Manuscript assessment |
|---|---|---|---|
| Final reward range | TuneRL base clip `[-2,2]` | TuneRLSft later caps/deltas and compactness can produce `< -2` | Incorrect as universal final range |
| Dense target | stage2/3 scale `0.50/0.70`; inner clip `[0.02,0.35]` | scale × clipped dense formula | `[0.02,0.22]` incorrect |
| Formal success | `0.08` plus conditional target bonus `0.20` | `0.28` when formal and `target_structure_match is not False` | `+0.02` incorrect |
| Group improvement | stage2 prev/best `0.20/0.20`; backbone `0.25/0.25`; stage3 `1.10/1.10`, `1.20/1.15` | data-dependent; global terms may receive baseline blend | `0.70/0.95` incorrect |
| No progress | raw `-0.06`; stage2/3 scales `0.50/1.15` | `-0.03/-0.069` | Correct only for stage2 |
| Generalization | tolerance `0.02`, scale `-2.0`, floor `-0.20` | clipped excess-gap formula | Correct |
| Descriptor diversity | `+0.03/+0.02/+0.08`; descriptor dominant `-0.03/-0.05` plus other repeat constants | data-dependent and quality-gated | bonuses partly correct; repeat pair unsupported |
| CNN diversity | `+0.07/+0.05/+0.12`; global dominant `-0.08/-0.12`; within-backbone `-0.04/-0.06` | branch-dependent and quality-gated | repeat pair needs branch label |
| Block diversity | batch `+0.06`, archive `+0.08`; repeat floors `-0.05/-0.08` | data-dependent and quality-gated | `+0.18`, `-0.12/-0.25` incorrect |
| Dominant descriptor/CNN | descriptor `-0.03/-0.05`; CNN values above | one applicable branch contributes | descriptor `-0.06/-0.10` incorrect |
| Repeated block cap | `STAGE23_REPEATED_BLOCK_REWARD_CAP=2.0` | conditional pre-local-competition upper cap; local competition then clips `[-2,2]` | `0.20` incorrect |
| Format/contract failure | wrapper caps core `≤-3.0`, hygiene `≤-2.0/-1.5`, dual backbone `≤-3.5` | later compactness can lower further | broadly correct, incompatible with universal `[-2,2]` |

Raw spot checks: `r_formal_success_signal=0.28` is exactly `0.08+0.20`; `r_dense=0.09515141111111111` is exactly `0.50*clip(0.03+0.28*0.507745+0.04*(0.9533555555555555-0.50),0.02,0.35)`; `r_repeat_family=-0.05500000000000001` is `-0.05×1.10`; and `r_plain_fuse_penalty=-0.11000000000000001` is `-0.10×1.10`. The complete source/value/component-field mapping is in `reward_components_rebuild.csv`.

### Difference from Faraz / previous Xi summary

All specifically proposed discrepancies are confirmed, with two precision qualifications: the `0.20` target addition uses `target_structure_match is not False`, and the repeated-block `2.0` cap occurs before local-competition clipping rather than being a separate final cap.

### Safe statement for the paper/email

Use separate **Raw configuration** and **Effective/logged behavior** columns. A single value column cannot consistently represent stage scaling, conditional bonuses, baseline blends, quality gates, and wrapper ordering.

### Remaining uncertainty

The mapping is verified for the fixed commit and the local manuscript copy. It does not prove every paper run used a clean tree at that commit, or that the local TeX is submission-final.

## 4. GPU-hour accounting

### Claim being checked

Whether primary and extended allocated GPU-hour totals reproduce from retained sacct rows, and why 1885.91 and 1885.92 differ.

### Evidence used

Local sacct CSVs and scripts, plus Julia2 `2968490_0` sacct, manifest selection 0, matching stdout/result, and `remote_evidence/wide38/gpu_task0_recomputed.json`. The remote evidence bundle passed SHA/run-state/raw-row validation.

### Recompute method

For every retained row: `elapsed_seconds × allocated_gpu_count / 3600`, summed with exact rational arithmetic. Original scripts were rerun against non-destructive copies. Reproduction:

```bash
python3 final_writeup_verification_20260908/scripts/recompute_gpu_hours.py \
  --primary-sacct delivery_20260727/06_gpu_hours/sacct_six_seed_cifar10_rl_20260728.csv \
  --extended-sacct reply_20260802/08_sacct_records.csv \
  --wide-manifest delivery_20260727/04_proxy_validation/proxy_manifest_wide38.jsonl \
  --output final_writeup_verification_20260908/gpu_hours_recomputed.json
```

That command regenerates the labelled local-only snapshot. The remote correction is separately reproducible with `python3 final_writeup_verification_20260908/scripts/remote_recompute_wide38_task0.py` and does not overwrite it.

### Exact result

| Scope | Exact GPU-hours | Rounded 2dp | Status |
|---|---:|---:|---|
| Primary one-pattern | 374.027777777777778 | 374.03 | VERIFIED |
| Primary four-pattern | 533.638888888888889 | 533.64 | VERIFIED |
| Primary total | 907.666666666666667 | 907.67 | VERIFIED |
| H100 | 553.723333333333333 | 553.72 | VERIFIED |
| L40S | 353.943333333333333 | 353.94 | VERIFIED |
| Ablation all rows | 934.487777777777778 | 934.49 | VERIFIED |
| Ablation overlap | 440.810000000000000 | 440.81 | VERIFIED |
| Unique ablation addition | 493.677777777777778 | 493.68 | VERIFIED |
| Qwen | 234.964444444444444 | 234.96 | VERIFIED for 4 retained segments |
| CIFAR-100 one-pattern | 210.695555555555556 | 210.70 | VERIFIED for 3 retained jobs |
| Proxy top20 | 19.625277777777778 | 19.63 | VERIFIED for 20 retained tasks |
| Local-only wide38 snapshot | 19.284166666666667 | 19.28 | 37 rows; incomplete |
| Local-only supplied-row total | 1885.913888888888889 | 1885.91 | incomplete; omits task 0 |
| Remote-recovered task 0 (`2968490_0`) | 0.533333333333333 | 0.53 | VERIFIED: 1 L40S × 1,920 s |
| Corrected successful wide38 | 19.817500000000000 | 19.82 | VERIFIED: 38 results |
| Corrected extended unique total | 1886.447222222222222 | 1886.45 | VERIFIED for previously declared cohorts |

For the incomplete local snapshot, summing exact cohorts then rounding gives 1885.91, while summing displayed components gives 1885.92. Both omit task 0. Adding its exact 0.5333333333 hours yields the current 1886.4472222222, rounded once to 1886.45. The quantity is allocated time, not measured GPU utilization.

### Difference from Faraz / previous Xi summary

The local retained CSV omitted task 0 and therefore undercounted wide38/grand total. Remote evidence shows task 0 completed in an earlier job rather than replacement array `2968494`. The original summary's 84-row bookkeeping remains unsupported by the local row inventory, but the total for its previously declared cohorts is now corrected.

### Safe statement for the paper/email

“The primary cohort used 907.67 allocated GPU-hours. After recovering successful wide38 task 0, the deduplicated total for the previously declared extended cohorts is 1886.4472222 GPU-hours (1886.45 rounded once).” Report primary and extended scopes separately.

### Remaining uncertainty

The failed superseded `2968450` array is excluded consistently with the original successful-result proxy scope. An all-attempt accounting would be a different scope and is not claimed here.

## 5. Threshold-sensitivity field

### Claim being checked

Which key generated `threshold_sensitivity_70_80_90` and whether “collapse in at most four of twelve runs” means exact forward graph or Family+Block.

### Evidence used

Recovered session `019f2703-a6dc-76b2-ada2-dccb85045e27` JSONL line 3274/ordinal 3273; preserved generator excerpt and paired output; delivered 12×800 raw archive; independent recomputation CSV and field-coverage JSON. Earlier local/Julia2 absence searches are retained only as search history.

### Recompute method

The historical inline generator maps `graph` to `graph_hash`, filters formal success, processes eight non-overlapping 100-attempt windows per 800-row run, requires at least 20 valid signatures per metric-window, and records the first window end reaching 0.70/0.80/0.90. The independent rerun applies that recovered logic to all 9,600 raw rows.

### Exact result

The key is `graph_hash` (**Exact Forward-Graph**). All 8,030 formal-success rows use it; fallback to `signature` or `actual_structure_signature` occurs zero times. First-threshold counts are 70%: 4/12, 80%: 3/12, 90%: 2/12. Four-pattern seed114 has four formal successes and no eligible window but remains in the historical 12-run denominator.

### Difference from Faraz / previous Xi summary

The earlier audit's `NOT VERIFIED` result is superseded because the source was recovered from session history. Calling the result Family+Block, continuous-window collapse, or persistent collapse is incorrect.

### Safe statement for the paper/email

“Exact forward-graph concentration reached the 70%, 80%, and 90% top-1 thresholds in at least one eligible non-overlapping 100-attempt window in 4/12, 3/12, and 2/12 runs, respectively.” Continue to label `actual_structure_signature` as Family+Block.

### Remaining uncertainty

The original standalone script file was not recovered because the source was an inline Python tool call. Counts describe first onset, not duration or persistence; only 11 runs had any eligible graph window.

## 6. SFT-only limitation facts

### Claim being checked

The actual seeds, attempts, conditions, epochs, split, and whether local evidence proves why the design was reduced.

### Evidence used

Six raw JSONL files; candidate IDs; generation/evaluation configs; and delivery README prose considered only as a narrative claim.

### Recompute method

Count rows and unique monotonic IDs, group directories/config seeds, and inspect evaluation protocol fields.

### Exact result

Verified actual scope: seeds 42/123/777 in each condition; 400 unique attempts per run (`0000`–`0399`); 3 seeds × 2 conditions × 400 = 2,400; CIFAR-10 five epochs; `trainvaltest`, split seed 42; 45k train, 5k reward-eval, 10k held-out test; SFT evaluation role `heldout_test`. Each packaged generation config describes a 100-candidate shard and each evaluation config a 50-candidate shard, so the consolidated raw file, not one shard config, proves the 400 total.

### Difference from Faraz / previous Xi summary

The completed scope is supported. The reduction rationale and any claim that reduction “does not change any claim” are not established by raw experiment/scheduler records.

### Safe statement for the paper/email

“The reduced experiment directly characterizes the pre-RL state for the three completed seeds, but provides less replication and statistical coverage than the planned six-seed design.”

### Remaining uncertainty

Why the planned scope was reduced is `NOT VERIFIED` beyond delivery/correspondence narrative.

## 7. Imagenette

### Claim being checked

Whether sampler, historical learned, and final matched learned Imagenette conditions exist in the supplied local archive.

### Evidence used

Rule-sampler raw/config package; delivered recomputation script; local archives/index; and remote Imagenette raw JSONL, run config/state, and sacct in the validated Julia2 evidence bundle.

### Recompute method

Count sampler raw rows/formal flags and search run metadata/packages by dataset, condition, completion state, protocol, seeds, and raw data availability.

### Exact result

Rule-constrained sampler Imagenette is directly verified: 100 attempts, 99 formal successes, one epoch, `trainvaltest`, held-out test. Remote evidence directly verifies historical learned one-pattern run `20260605_0925_dscoder_imagenette_h100`: Slurm job `2666209` completed, raw JSONL has 1,000 rows, config is one-pattern Imagenette with one formal epoch, and run state records base commit `c9276599...`. No completed protocol-matched learned four-pattern counterpart or matched one/four pair was found in the inspected local plus Julia2 scope.

### Difference from Faraz / previous Xi summary

“No new matched Imagenette result” is supported within scope. An unqualified “Imagenette was not run” would incorrectly erase the sampler and historical one-pattern records. The stated scheduling priority is prose, not raw evidence.

### Safe statement for the paper/email

“Historical learned one-pattern Imagenette runs are present on Julia2. No completed matched learned one-/four-pattern Imagenette conditions were found in the inspected Julia2 archive.” Keep the verified 99/100 sampler separate.

### Remaining uncertainty

This does not prove no run exists outside the inspected user-owned scope. Whether no further run is planned is `DO NOT CLAIM YET; confirm manually`.

## 8. Code availability

### Claim being checked

Canonical repositories, fixed commit availability, its relation to five-epoch full-reward runs, and NN Dataset provenance.

### Evidence used

Local git remotes/object/ancestry; all twelve raw six-seed configs; fixed-commit source; rule-sampler config; and six validated remote run states for seeds 114/514/919 in both conditions.

### Recompute method

```bash
python3 final_writeup_verification_20260908/scripts/check_provenance.py \
  --output final_writeup_verification_20260908/provenance_evidence.json
```

The script verifies remotes/objects and inventories raw config git fields rather than assuming that commit existence proves execution identity.

### Exact result

| Repository | Commit | Role | Evidence |
|---|---|---|---|
| NNGPT (`https://github.com/ABrain-One/nn-gpt.git`) | `c91714dbe7dad1d02a9080243945bbf8e8ec9300` base commit; exact trees `NOT VERIFIED` | Fixed reward/runtime source; archived base for six-seed CIFAR-10 | Six raw configs name commit with `dirty=true`; six matching remote run states name commit; no source snapshot/diff |
| NN Dataset (`https://github.com/ABrain-One/nn-dataset.git`) | `NOT VERIFIED` | Dataset loader/evaluator dependency | Remote verified; current HEAD cannot substitute for historical run provenance |

The fixed commit exists and contains the relevant reward/runtime source. It is not an ancestor of current local HEAD. The rule-sampler Imagenette config directly names it. Its existence does not make it the unique code commit for all paper experiments.

### Difference from Faraz / previous Xi summary

“All runs used clean commit c917...” remains unsupported. Remote run states fill the six blank base-commit declarations, but do not establish byte-identical clean trees.

### Safe statement for the paper/email

“The audited NNGPT source is available at `ABrain-One/nn-gpt`, commit `c91714d...`. The six-seed cohort is archived against that base commit, but exact clean-tree provenance is incomplete. The historical NN Dataset experiment commit is not verified.”

### Remaining uncertainty

Exact dirty/source diffs and the NN Dataset historical commit remain unresolved.

## 9. Secondary consistency sweep

The bounded sweep is documented in `secondary_consistency_sweep.md`. Directly relevant corrections are:

- structural statistics must state formal-success N separately from attempts;
- all-400 versus final-100 entropy comparisons need the matched-window sensitivity caveat;
- family and Family+Block changes are heterogeneous across seeds;
- one-epoch and five-epoch metrics cannot be interchanged;
- the original 30.30% raw artifact identifies CNN-signature top-1, so the August 8 performance comparison is wrong;
- `actual_structure_signature` and `graph_hash` require distinct names;
- reward raw constants and effective behavior cannot share one scalar column;
- 1885.91/1885.92 is rounding order for supplied rows, while missing wide task 0 is a separate completeness defect;
- the extended summary's 84 records is unsupported (82 distinct supplied rows);
- using the delivery function's documented `test_acc`-first precedence, the completed CIFAR-100 package recomputes to 69.57372212% seed mean (69.57%) and 69.64405534% pooled (69.64%); its 2,096 formal successes and 2,060 positive rewards also match `FINAL_RESULTS.md`. Horizon-5 values differ on some rows and must not silently replace `test_acc`;
- historical, sampler, and matched learned Imagenette categories must remain separate;
- derived provenance cannot override dirty/blank raw git fields.

The SFT accuracy and completed CIFAR-100 checks can be regenerated from raw JSONL with:

```bash
python3 final_writeup_verification_20260908/scripts/recompute_secondary_consistency.py
```

Its JSON output records each input SHA-256 and, per run, the formal-success N, numeric `test_acc` N, numeric horizon-5 N, equality N, and fallback N. It reuses the exact delivery precedence in `recompute_final_summaries.py:40–48`; all formal-success rows use numeric `test_acc`, and fallback N is zero.

No broader paper-wide audit was performed.
