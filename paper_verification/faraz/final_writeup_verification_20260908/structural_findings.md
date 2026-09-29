# TASK 1 and TASK 6 — independent structural recomputation

## Status

- **TASK 1: VERIFIED with an important split caveat.** The six SFT-only raw files and the six matched-seed post-RL raw trajectories were found and independently recomputed. Faraz's family-level hypotheses are numerically supported. The comparison is descriptive, not a clean causal estimate.
- **TASK 6: PARTIAL.** The completed local scope (3 seeds × 2 conditions × 400 attempts, 5 epochs) is verified from raw files and configs. The reason for reducing the planned design is stated in delivery prose but is not independently established by raw experimental records.

## Claim being checked

The audit tests whether one-pattern generation was already concentrated at family level before RL, whether four-pattern SFT-only retained family diversity, and how the same metrics change in the last-100-attempt post-RL windows for seeds 42, 123, and 777.

## Evidence used

Highest-priority inputs:

1. Six SFT-only JSONL files under `/Users/zhangxi/code/RL/faraz_followup_delivery_20260808/sft_only/raw/{1pattern,4pattern}_seed{42,123,777}/generation_samples.jsonl`.
2. Their `generation_run_config.json` and `evaluation_run_config.json` files.
3. Six post-RL JSONL members under `article_raw_data_archive_20260704/01_six_seed_robustness/` in `/Users/zhangxi/code/RL/faraz_delivery_20260726/article_raw_data_archive_20260718_resend_20260726.zip`, plus their `run_config.json` members.
4. Formal-success rule: `/Users/zhangxi/code/RL/faraz_followup_delivery_20260808/scripts/recompute_final_summaries.py`, lines 51–58, counts only `api_result.formal_success_candidate is True`.
5. Existing structural implementation used as a cross-check: `/Users/zhangxi/code/RL/delivery_20260727/05_canonicalisation_granularity/analyze_canonicalisation_sweep.py`, lines 32–38 (formal filter), 41–69 (entropy/effective number), and 89–95 (`rows[-100:]` before formal filtering).
6. Manuscript final-window definition: `/Users/zhangxi/Desktop/example-cvpr/Paper_draft_en.tex`, lines 205–207, defines the RL window as the last continuous 100 generated/attempted samples, followed by formal-success-only accuracy and structure statistics.

Every input JSONL SHA-256 is stored beside its run in `sft_family_recompute.json`. The post-RL container ZIP SHA-256 is also recorded there.

## Recompute method

For each attempted window, retain a row iff `api_result.formal_success_candidate is True`. For every requested signature, count non-missing values among those retained rows. If counts are \(n_i\), \(N=\sum_i n_i\), and \(p_i=n_i/N\):

\[
H=-\sum_i p_i\ln p_i,\qquad N_{\mathrm{eff}}=\exp(H).
\]

The implementation uses the natural logarithm, matching the delivered canonicalisation script. On all six SFT-only runs, independently recomputed backbone, CNN, backbone+CNN, and graph unique counts/top-1 counts/effective numbers match every corresponding archived `structural_diversity_summary.json` value (absolute effective-number difference at most `1e-12`). The stored files do not contain a `backbone_cnn_signature` field; the cross-check reconstructs the archived definition exactly as `backbone_signature + "::" + cnn_signature`.

Post-RL files contain 800 rows each but no `candidate_id`. Therefore, the audit can verify that the delivered files contain 800 distinct row fingerprints and can apply the established file-order `rows[-100:]` definition. It **cannot** independently prove candidate indices 0–799 or reconstruct/deduplicate a resume boundary from IDs that were not recorded. The safe attempted-window label is “file-order rows 701–800”.

## Exact result: pre-RL vs post-RL

All shares below use formal-success `N` as denominator, after selecting the attempted window. Full-precision machine-readable values are in `sft_family_pre_post.csv` and `sft_family_recompute.json`.

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

Faraz's cited family top-1 percentages are supported after ordinary rounding: one-pattern post-RL `100%, 100%, 55.6%`; four-pattern post-RL `100%, 40.6%, 59.2%`. The SFT-only hypotheses are likewise supported by the exact values above.

## SFT-only cross-check signatures

Each cell is `unique; top-1 count/N; top-1 share; effective number`. Shannon entropy and top-1 values themselves are preserved in the JSON output.

| Condition | Seed | Signature | Exact result |
|---|---:|---|---|
| one-pattern | 42 | backbone_signature | 40; 147/367; 0.40054495912806537; 9.372528584489976 |
| one-pattern | 42 | block_signature | 38; 98/367; 0.2670299727520436; 15.397438709758429 |
| one-pattern | 42 | cnn_signature | 2; 355/367; 0.9673024523160763; 1.1548812863780262 |
| one-pattern | 42 | backbone+cnn (derived) | 44; 140/367; 0.3814713896457766; 10.547495916158514 |
| one-pattern | 42 | graph_hash | 71; 88/367; 0.23978201634877383; 21.40924092062136 |
| one-pattern | 123 | backbone_signature | 38; 149/367; 0.40599455040871935; 8.747536083616616 |
| one-pattern | 123 | block_signature | 35; 104/367; 0.28337874659400547; 13.492098250862366 |
| one-pattern | 123 | cnn_signature | 3; 355/367; 0.9673024523160763; 1.1657636776638265 |
| one-pattern | 123 | backbone+cnn (derived) | 45; 147/367; 0.40054495912806537; 9.573756915851277 |
| one-pattern | 123 | graph_hash | 69; 94/367; 0.2561307901907357; 19.76688749820615 |
| one-pattern | 777 | backbone_signature | 33; 143/353; 0.40509915014164305; 8.148282617276205 |
| one-pattern | 777 | block_signature | 43; 90/353; 0.254957507082153; 16.422741735594503 |
| one-pattern | 777 | cnn_signature | 4; 340/353; 0.9631728045325779; 1.1941853171747545 |
| one-pattern | 777 | backbone+cnn (derived) | 40; 139/353; 0.3937677053824363; 9.209308849814803 |
| one-pattern | 777 | graph_hash | 68; 82/353; 0.23229461756373937; 19.1219780054366 |
| four-pattern | 42 | backbone_signature | 162; 14/398; 0.035175879396984924; 128.91827691703392 |
| four-pattern | 42 | block_signature | 167; 9/398; 0.022613065326633167; 126.50344450618233 |
| four-pattern | 42 | cnn_signature | 4; 120/398; 0.3015075376884422; 3.939966836842404 |
| four-pattern | 42 | backbone+cnn (derived) | 296; 6/398; 0.01507537688442211; 265.8547380396917 |
| four-pattern | 42 | graph_hash | 326; 6/398; 0.01507537688442211; 297.47576956412627 |
| four-pattern | 123 | backbone_signature | 173; 12/397; 0.030226700251889168; 138.6171552637986 |
| four-pattern | 123 | block_signature | 171; 17/397; 0.042821158690176324; 122.34247162822165 |
| four-pattern | 123 | cnn_signature | 6; 122/397; 0.30730478589420657; 4.082227400852246 |
| four-pattern | 123 | backbone+cnn (derived) | 294; 5/397; 0.012594458438287154; 264.7788304890211 |
| four-pattern | 123 | graph_hash | 339; 5/397; 0.012594458438287154; 316.0522952600716 |
| four-pattern | 777 | backbone_signature | 178; 18/399; 0.045112781954887216; 140.7314407238262 |
| four-pattern | 777 | block_signature | 163; 15/399; 0.03759398496240601; 123.6544109202024 |
| four-pattern | 777 | cnn_signature | 6; 119/399; 0.2982456140350877; 4.095900379104709 |
| four-pattern | 777 | backbone+cnn (derived) | 304; 5/399; 0.012531328320802004; 271.85451861273486 |
| four-pattern | 777 | graph_hash | 331; 5/399; 0.012531328320802004; 305.82271218157643 |

`family_hash` and `cnn_signature` are identical at the distribution level for seeds 42 and 777. Seed 123 one-pattern differs slightly: `family_hash` has 4 unique values, dominant count 354/367, and effective number 1.187793833210298; `cnn_signature` has 3 unique values, dominant count 355/367, and effective number 1.1657636776638265. They must not be assumed interchangeable as a general rule.

## Interpretation

1. **Was one-pattern already family-concentrated at SFT-only? YES.** Its dominant family accounts for 96.32–96.73% of formal-success candidates and family effective number is 1.1549–1.1942. Family-level concentration therefore cannot be attributed simply to RL. Post-RL maintains/strengthens this concentration for seeds 42 and 123, while seed 777 becomes less concentrated (top-1 55.56%, effective 2.1883). “RL amplified one-pattern family collapse” is not supported uniformly across all three seeds.
2. **Did four-pattern SFT-only preserve higher family diversity? YES, relative to one-pattern.** Dominant-family shares are 29.82–30.73% and family effective numbers are 3.9400–4.0959. Post-RL family concentration rises strongly for seeds 42 and 777. Seed 123's top-1 rises to 40.625%, but its entropy effective number also rises slightly from 4.0822 to 4.3268; calling all three seeds an unqualified entropy collapse would be inaccurate.
3. **Causal boundary.** These are observational comparisons between independent SFT-only samples and later RL trajectory windows. The safe wording is: “One-pattern family concentration is already present in the SFT-only samples and is maintained in two final RL windows, with seed 777 partially diversifying. Four-pattern SFT-only remains much less family-concentrated; the final RL windows show stronger dominant-family concentration in all three seeds, most sharply for seeds 42 and 777. The comparison does not by itself isolate a causal RL effect.”

At the Family+Block level, four-pattern SFT-only is extremely broad (effective ~243), and every post-RL window is numerically lower (24.51–54.16). However, entropy effective number is sample-size sensitive: the primary comparison uses 400 SFT attempts versus 100 post-RL attempts, and an effective number estimated from 100 attempts cannot exceed 100. It is therefore invalid to use the raw ~243→24–54 contrast alone as proof of an RL effect.

As a secondary sensitivity analysis only, applying the same last-100-file-row window to SFT-only gives:

| Condition | Seed | SFT last-100 formal N | SFT last-100 family top-1 / eff. | SFT last-100 Family+Block top-1 / eff. | Post-RL family top-1 / eff. | Post-RL Family+Block top-1 / eff. |
|---|---:|---:|---:|---:|---:|---:|
| one-pattern | 42 | 91 | 95.6044% / 1.1976 | 28.5714% / 13.8335 | 100% / 1.0000 | 8.7912% / 52.4793 |
| one-pattern | 123 | 95 | 95.7895% / 1.1907 | 31.5789% / 12.6312 | 100% / 1.0000 | 32.0000% / 4.3622 |
| one-pattern | 777 | 91 | 96.7033% / 1.1559 | 18.6813% / 15.2345 | 55.5556% / 2.1883 | 13.1313% / 30.8247 |
| four-pattern | 42 | 100 | 32.0000% / 3.9348 | 3.0000% / 84.2326 | 100% / 1.0000 | 6.0000% / 47.9502 |
| four-pattern | 123 | 100 | 34.0000% / 4.2286 | 2.0000% / 88.2703 | 40.6250% / 4.3268 | 6.2500% / 54.1574 |
| four-pattern | 777 | 100 | 32.0000% / 3.9348 | 2.0000% / 87.0551 | 59.1837% / 2.6404 | 10.2041% / 24.5095 |

This matched-window sensitivity preserves the four-pattern Family+Block decrease in all three seeds, though it remains observational. One-pattern Family+Block changes are heterogeneous: seed 42 and 777 increase in effective number, while seed 123 decreases. Family-level and Family+Block conclusions must remain separate. The primary requested table remains the 400-attempt SFT-only versus final-100 post-RL comparison.

## Split fairness

- All six SFT-only `evaluation_run_config.json` files specify `dataset=cifar-10`, `formal_reward_epochs="5"`, `split_protocol=trainvaltest`, `split_seed=42`, `eval_split_role=heldout_test`, train `45k`, reward-eval `5k`, and held-out test `10k`.
- All six post-RL `run_config.json` files specify CIFAR-10, formal epochs 5, the same `trainvaltest` partition sizes and split seed, but `eval_split_role=reward_eval`. Thus their formal gate/performance target is the 5k reward-eval split, not the SFT-only 10k held-out-test role.
- At fixed commit `c91714dbe7dad1d02a9080243945bbf8e8ec9300`, `ab/gpt/util/ArchDiscovery.py::extract_graph_info` (lines 632–700) parses `init_code` and `forward_code`, builds the AST-derived graph expression, and hashes `graph_expr`, `family_id|family_expr`, and `cnn_expr` at lines 671–685. `ab/gpt/TuneRL.py::build_actual_structure_signature` (lines 2644–2666) composes the family ID, descriptor, backbone-call count, family hash, CNN signature, and block signature. These functions consume architecture/code and structural metadata, not dataset examples or split names. This is the source basis for calling the signatures themselves split-independent.
- The downstream filter is not merely a convenience flag. In the same fixed commit, `ab/gpt/TuneRL.py::_stage1_trainability_ok` (lines 3386–3397) creates formal success from graph parse success plus evaluation-produced `built_ok`, `forward_shape_ok`, and at least one of `backward_ok`, `trained_step_ok`, or a completed formal epoch; `_is_trainable_candidate` forwards that gate at lines 3352–3353, and the result is written as `formal_success_candidate` at line 4773. For the SFT full-code runner, `scripts/baseline_experiment_runner.py::_augment_full_code_result` implements the equivalent gate at lines 669–690. Thus there is no accuracy-threshold test here, but the gate depends on completing construction/training/evaluation under the configured split. The reported distributions then condition on that result, so the observed candidate populations can differ through split-dependent evaluation/execution outcomes.

Safe statement: **“The structural signatures are split-independent, but the analyzed candidate population is conditioned on split-dependent formal success. This is a reasonable descriptive structural comparison, but the evaluation populations are not strictly identical.”**

## TASK 6 exact scope and limitations

Directly verified locally:

- Seeds: 42, 123, 777 for both conditions (directory names, candidate IDs, and generation config seed).
- Attempts: each raw JSONL has exactly 400 rows; each has 400 unique IDs running monotonically from `0000` to `0399`.
- Design: 3 seeds per condition × 2 conditions × 400 attempts = 2,400 attempted candidates.
- Evaluation: every evaluation config records held-out CIFAR-10, 5 epochs, `trainvaltest`, split seed 42, train 45k / reward-eval 5k / held-out test 10k.

Configuration caveat: each packaged `generation_run_config.json` records a single shard (`count: 100`), and each `evaluation_run_config.json` records a single evaluation shard (`candidate_count: 50`), while the corresponding consolidated raw JSONL has 400 unique candidates. The raw files prove the completed 400-attempt scope; the single archived shard config alone does not document all shard submissions.

The local delivery README (lines 17–25) says the completed SFT-only experiment was reduced and that GPU time prioritized the title-gating SFT-only and recoverable CIFAR-100 work over six new matched Imagenette runs. This is a hand-written delivery explanation, not a scheduler/sacct or original planning record proving why a planned 6×800 design became 3×400. Therefore:

- **Locally verified experimental scope:** 3 completed seeds per condition, 400 attempts per seed, 2,400 total, five-epoch held-out CIFAR-10.
- **Reason for reduction:** `NOT VERIFIED` from raw experiment evidence; available only as delivery/correspondence narrative unless a planning or scheduling record is supplied.

Safe limitation statement: **“The reduced experiment directly characterizes the pre-RL state for the three completed seeds, but provides less replication and statistical coverage than the planned six-seed design.”** The stronger sentence “the reduction does not change any claim” is not justified by these local records.

## Difference from Faraz / previous Xi summary

- Faraz's SFT-only and post-RL family top-1 hypotheses are verified at full precision.
- Any previous wording that attributes one-pattern family concentration wholly to RL requires correction: it is already present before RL.
- “Four-pattern becomes more family-concentrated after RL” is descriptively supported by top-1 share in all three seeds, but “family entropy collapses in every seed” is false for seed 123.
- Any wording implying identical evaluation populations requires the split caveat above.
- Any claim that the reduced scope cannot affect conclusions is too strong.

## Remaining uncertainty

1. Post-RL archived rows lack candidate IDs; exact file-order windowing is reproduced, but resume-boundary candidate-index provenance cannot be independently reconstructed from these files.
2. The reason for reducing the planned replication is not verified by raw local experiment evidence.
3. Observational pre/post distributions do not identify an RL causal effect without a stronger matched counterfactual design.
