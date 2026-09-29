# Fixed-commit reward-component rebuild

## Scope and evidence hierarchy

The implementation audit is fixed to NNGPT commit `c91714dbe7dad1d02a9080243945bbf8e8ec9300`. `git cat-file -t` resolves it as `commit`. Every implementation statement below was derived from `git show c91714dbe7dad1d02a9080243945bbf8e8ec9300:<path>`; the current working-tree versions were not used as substitutes.

The locally available numerical manuscript table is `/Users/zhangxi/Desktop/example-cvpr/Paper_draft_en.tex`, lines 151--173. It is the table labelled `tab:reward_components`. This audit does not establish that file as the submission-final source. The rerunnable script records its SHA-256 and modification timestamp in `reward_audit_evidence.json`. `/Users/zhangxi/Desktop/thesis-template-LS4CV/approach.tex`, lines 197 onward, contains a second table with the same label but mostly qualitative roles rather than the disputed numerical values.

## Manuscript table transcription

The following is the complete row-level extraction of the locally available numerical table at `Paper_draft_en.tex:158-169`. “Condition / description” and “manuscript-stated value” preserve the table's wording and are audit inputs, not evidence that the values are correct.

| Row name | Condition / description | Manuscript-stated value |
| --- | --- | --- |
| Final reward range | Applied after summing primary and tie-break terms | clipped to `[-2.0, 2.0]` |
| Dense target term | Uses `T(a)` and train accuracy | scale `0.50`; inner term clipped to `[0.02,0.22]` |
| Formal-success bonus | Candidate passes formal-success gate | `+0.02` |
| Group improvement | Target exceeds previous/best group baselines | group scale `0.70`; backbone-group scale `0.95` |
| No-progress penalty | Candidate does not beat the effective group target | `0.50 * (-0.06)` |
| Generalization penalty | frozen train/evaluation gap exceeds `0.02` | `-2.0 *` gap, capped at `-0.20` |
| Descriptor diversity | Batch/archive novelty or repetition | `+0.03` batch unique, `+0.02` archive novel, `+0.08` global archive novel; repeat penalties down to `-0.10 / -0.06` |
| CNN/backbone-CNN diversity | Batch/archive novelty or repetition | `+0.07` batch unique, `+0.05` archive novel, `+0.12` global archive novel; repeat penalties down to `-0.14 / -0.08` |
| Block diversity | Batch/archive novelty or repetition | `+0.06` batch unique, `+0.18` archive novel; repeat penalties down to `-0.12 / -0.25` |
| Dominant descriptor/CNN repeat | Reuse dominant signature with no quality refresh | descriptor repeat `-0.06` or `-0.10`; CNN repeat `-0.08` or `-0.12` |
| Repeated block cap | Repeated block with no quality refresh | reward capped at `0.20` |
| Format / contract failure | XML/signature, extra class/import, missing dual backbone, build/forward failure | core format cap `<= -3.0`; severe hygiene cap `<= -2.0`; missing dual backbone cap `<= -3.5`; forward/shape failures are capped below zero |

## Execution path

For stage 2 and stage 3, `TuneRL.py` selects the stage profile at lines 3448--3481, calculates formal reward components at lines 4213--4518, forms `r_primary` and the goal-match tie break at lines 4598--4621, and clips that base sum to `[-2,2]`. It then applies non-improvement/repetition upper caps, local competition, executability clamps, warmup replacement when applicable, and the target-structure final clamp (lines 4622--4698). Component fields and the resulting reward are stored at lines 4700--4758.

`TuneRLSft.py` is an additional wrapper. At lines 330--380 it adds extraction/contract deltas and applies a trainability clamp. At lines 382--401 it caps core-format violations at `-3.0`, severe hygiene violations at `-2.0`, minor hygiene violations at `-1.5`, and missing dual backbone at `-3.5`. At lines 418--432 it adds compactness penalties after those caps and applies the target gate. Therefore `[-2,2]` is the base component-sum clip, not the guaranteed final logged reward range.

## Corrected component table

The full row-by-row table, including raw constant, stage, scaling, cap, logged field, manuscript value, assessment, and recommendation, is in `reward_components_rebuild.csv`. The critical corrections are:

| Manuscript row | Fixed-commit result | Assessment |
| --- | --- | --- |
| Final reward range | Base sum is clipped to `[-2,2]`; wrapper caps/deltas can make final logged reward lower than `-2` | Incorrect if stated as the final logged range |
| Dense target | Stage-2 scale `0.50`, stage-3 scale `0.70`; inner clip `[0.02,0.35]` | Manuscript upper clip `0.22` is incorrect |
| Formal-success bonus | Raw `0.08`; `target_structure_match is not False` adds `0.20`; combined `0.28` | Manuscript `+0.02` is incorrect |
| Group improvement | Stage 2 prev/best scales `0.20/0.20`; backbone prev/best `0.25/0.25` | Manuscript `0.70/0.95` is incorrect |
| No progress | `-0.06 * 0.50 = -0.03` in stage 2; `-0.06 * 1.15 = -0.069` in stage 3 | Correct only as a stage-2 value |
| Generalization | Tolerance `0.02`, scale `-2.0`, floor `-0.20` | Correct |
| Descriptor novelty | `+0.03` batch, `+0.02` archive, `+0.08` global | Positive figures correct; repeat figures unsupported |
| CNN novelty | `+0.07` batch, `+0.05` archive, `+0.12` global | Positive figures correct; repeat pair mixes branches/levels |
| Block novelty | `+0.06` batch, `+0.08` archive | Manuscript `+0.18` is incorrect |
| Dominant repeats | descriptor `-0.03/-0.05`; global CNN `-0.08/-0.12`; within-backbone CNN `-0.04/-0.06` | Descriptor values incorrect; CNN values apply only to one branch |
| Repeated-block cap | Conditional pre-local-competition upper cap `2.0`; local competition runs afterward | Manuscript `0.20` is incorrect |
| Format/contract failure | Wrapper caps `-3.0`, `-2.0/-1.5`, and `-3.5`, followed by possible extra compactness penalties | Caps broadly correct; they contradict a universal final `[-2,2]` range |

## Raw-candidate verification

The candidate searches are performed by `scripts/rebuild_reward_audit.py`; it reads JSON objects rather than matching prose or summary files.

- `r_formal_success_signal = 0.28` occurs in raw candidates with `formal_success_candidate=true` and `target_structure_match=true`. The exact source predicate for adding `0.20` is `target_structure_match is not False`; a missing value also enters that branch. In the checked raw observations the value is explicitly `true`, and `0.28` is exactly explained by `0.08 + 0.20`.
- A candidate at archive member `article_raw_data_archive_20260704/01_six_seed_robustness/1pattern_seed42/generation_samples.jsonl`, line 800, has `reward_target_value=0.507745`, `frozen_train_acc=0.9533555555555555`, stage `stage2_formal_explore`, and `r_dense=0.09515141111111111`. Recalculation gives `0.50 * clip(0.03 + 0.28*0.507745 + 0.04*max(0,0.9533555555555555-0.50), 0.02, 0.35) = 0.09515141111111111`. The shortened `0.453356` in the proposed expression is the post-subtraction train-accuracy excess, not the raw log field.
- Raw candidates contain `r_repeat_family=-0.05500000000000001` in stage 2. This is exactly `REPEAT_FAMILY_PENALTY (-0.05) * STAGE2_REPEAT_FAMILY_SCALE (1.10)` at `TuneRL.py:4013-4015,4517`.
- The same raw path contains `r_plain_fuse_penalty=-0.11000000000000001` in stage 2. This is exactly `PLAIN_FUSE_PENALTY (-0.10) * STAGE2_PLAIN_FUSE_SCALE (1.10)` at `TuneRL.py:4023-4027,4518`. The alternative plain-dual-backbone constant is `-0.28`, so the `-0.11` observation identifies the ordinary plain-fuse branch.

Exact candidate paths and line numbers are written to `reward_audit_evidence.json` on every rerun.

## Reporting recommendation

Use separate **Raw configuration** and **Effective/logged behavior** columns. Raw constants alone conceal stage scaling and conditional composition; effective values alone are data-dependent and cannot be represented by one scalar for improvement and novelty components. A single mixed value column is what produced the current inconsistencies. For each row, report raw constants, stage-specific multiplier/formula, caps, and logged component field. Where the effective value depends on baselines, repetition counts, quality gates, or target match, give the formula or range rather than inventing one number.

## Remaining uncertainty

The code-to-value mapping is verified for the fixed commit. Whether every paper run executed this exact commit belongs to the provenance audit and cannot be inferred merely because the commit exists. The local numerical table version is explicitly identified above; confirmation that it is the exact final manuscript version remains outside this source audit.

The raw seed-42 six-seed run configuration in `article_raw_data_archive_20260718_resend_20260726.zip`, member `article_raw_data_archive_20260704/01_six_seed_robustness/1pattern_seed42/run_config.json`, directly records commit `c91714dbe7dad1d02a9080243945bbf8e8ec9300`, reward variant `full_reward`, formal epochs `5`, and resume stage `stage2_formal_explore`. It also records `dirty: true`, so this evidence establishes the recorded commit and configuration, while the exact uncommitted diff remains unavailable from that record.
