# Independent QA: fixed-commit reward-table audit

## Scope and verdict

This is an independent, read-only acceptance check of
`reward_components_rebuild.csv` and `reward_components_rebuild.md`.  It used
only `git show c91714dbe7dad1d02a9080243945bbf8e8ec9300:<path>` for code, the
raw member `01_six_seed_robustness/1pattern_seed42/run_config.json` in
`faraz_delivery_20260726/article_raw_data_archive_20260718_resend_20260726.zip`
for the relevant five-epoch run configuration, and raw JSONL members for the
four logged-component observations.

**PASS, with two required wording corrections before final use.**  The CSV
covers every numerical row in the locally available `Paper_draft_en.tex`
`tab:reward_components` (lines 158--169), and its raw constants, stage
profiles, clips, and component fields agree with the fixed commit.  It should
not say that the fixed commit is on the current `main`: this repository's
current `main` does not contain the commit.  The assertion needed here is
instead supported directly by the seed-42 five-epoch run configuration, whose
recorded commit is exactly `c91714dbe7dad1d02a9080243945bbf8e8ec9300` and whose
reward configuration is `full_reward`, formal epochs `5`, and resume stage
`stage2_formal_explore`.

The two required edits are:

1. Replace “target structure matches” in the Formal-success recommended value
   with “`target_structure_match is not False`”.  At fixed-commit
   `ab/gpt/TuneRL.py:4236-4239`, the additional `0.20` is taken on that precise
   condition; a missing value also takes the branch.  Raw checked candidates
   have the value `true`, so “matches” describes those observations but is not
   the exact source predicate.
2. Avoid calling `STAGE23_REPEATED_BLOCK_REWARD_CAP=2.0` a distinct final
   reward cap without its sequence.  It is applied at `TuneRL.py:4640-4659`
   before `_stage23_local_competition_reward` (4671-4684), which itself clips
   to `[-2,2]` at 2881.  The numerical correction from manuscript `0.20` to
   source `2.0` is correct; the safe wording is “a conditional pre-local-
   competition upper cap of 2.0.”

## Fixed-commit evidence

`git cat-file -t c91714dbe7dad1d02a9080243945bbf8e8ec9300` returns `commit`.
The code evidence below uses the line numbers of `git show` output, not the
working tree.

| Paper row | QA result | Fixed-commit path and lines | Notes |
| --- | --- | --- | --- |
| Final reward range | PASS correction | `TuneRL.py:4598-4621`; `TuneRLSft.py:376-432` | Base sum is clipped to `[-2,2]`. The wrapper applies deltas/caps, then adds compactness penalties after its caps, so it does not establish a universal final logged range of `[-2,2]`. |
| Dense target | PASS correction | `TuneRL.py:352,366`; `4231-4235` | Stage 2 / 3 scale is `0.50 / 0.70`; inner clip is `[0.02,0.35]`, not `0.22`. |
| Formal-success bonus | PASS, wording edit | `TuneRL.py:204,266,4236-4239` | `0.08`, plus `0.20` when `target_structure_match is not False`; no stage multiplier. |
| Group improvement | PASS correction | `TuneRL.py:353-357,367-371`; `3448-3481`; `4244-4283` | Stage-2 prev/best is `0.20/0.20`; backbone prev/best is `0.25/0.25`. Global values are subsequently multiplied by `global_baseline_blend` only when a backbone baseline exists. Thus no single data-independent “effective” number exists. |
| No-progress penalty | PASS | `TuneRL.py:205,363,377`; `4300-4305` | Effective stage 2 is `-0.06*0.50=-0.03`; stage 3 is `-0.06*1.15=-0.069`. |
| Generalization penalty | PASS | `TuneRL.py:210-212`; `4306-4312` | Exactly `clip(-2*max(0,frozen_train-frozen_test-0.02), -0.20, 0)`. |
| Descriptor diversity | PASS correction | `TuneRL.py:299-312`; `4327-4381` | Listed positive constants are correct. The manuscript's aggregate repeat pair does not map to one descriptor constant; the audit correctly reports individual terms. |
| CNN/backbone-CNN diversity | PASS correction | `TuneRL.py:313-321,343-349`; `4383-4454` | Positive constants are correct. `-0.08/-0.12` is the global CNN branch; the within-backbone branch is `-0.04/-0.06`, so the paper label needs branch specificity. |
| Block diversity | PASS correction | `TuneRL.py:322-328`; `4465-4492` | Batch `+0.06`, archive `+0.08`; manuscript `+0.18` has no matching constant. |
| Dominant descriptor/CNN repeat | PASS correction | `TuneRL.py:309-312,343-349`; `4360-4381,4416-4454` | Descriptor is `-0.03/-0.05`, not `-0.06/-0.10`; CNN needs branch qualification. |
| Repeated block cap | PASS, wording edit | `TuneRL.py:328`; `4640-4684`; `2834-2881` | Source is `2.0`, not `0.20`; see the sequence caveat above. |
| Format / contract failure | PASS with scope | `TuneRLSft.py:330-432` | Core-format `<=-3.0`, severe/minor hygiene `<=-2.0/-1.5`, and missing dual backbone `<=-3.5` are present. The CSV correctly states later compactness additions. |

The CSV is complete for the numerical manuscript table. It also correctly
exposes the hidden steps that the manuscript table had mixed: conditional
gates, stage profile values, global-baseline blend, quality gating of positive
novelty, base clipping, stage-23 local competition, executability/target
clamps, and the `TuneRLSft` wrapper.

## Raw-log spot checks

All four requested values are present in raw JSONL, and the target candidate
for the dense calculation is from a run whose raw `run_config.json` records
the fixed commit and `full_reward` five-epoch configuration.

| Field | Raw source | Result |
| --- | --- | --- |
| `r_formal_success_signal` | `faraz_delivery_20260726/cifar100_provenance/seed777/generation_samples.jsonl:1` | `0.28`, with `formal_success_candidate=true`, `target_structure_match=true`; exactly `0.08+0.20`. |
| `r_dense` | zip member `.../01_six_seed_robustness/1pattern_seed42/generation_samples.jsonl:800` | `0.09515141111111111`; raw `reward_target_value=train target=0.507745`, `train_acc=frozen_train_acc=0.9533555555555555`, stage 2. Exact calculation: `0.50*clip(0.03+0.28*0.507745+0.04*(0.9533555555555555-0.50),0.02,0.35)`. |
| `r_repeat_family` | `faraz_delivery_20260726/cifar100_provenance/seed777/generation_samples.jsonl:161` | `-0.05500000000000001`, exactly `-0.05*1.10`. |
| `r_plain_fuse_penalty` | same raw line | `-0.11000000000000001`, exactly `-0.10*1.10`; this identifies the ordinary plain-fuse branch rather than the `-0.28` dual-backbone-fuse branch. |

## Recommended representation

Use separate **Raw configuration** and **Effective/logged behavior** columns.
This is required for accuracy: group terms are data dependent and can acquire a
conditional baseline blend; novelty terms are conditionally quality-gated; and
the final reward passes through base, local-competition, trainability,
target-structure, and wrapper operations. A single mixed “Value / weight”
column cannot represent the implementation faithfully.

## Limit of this QA

The source-to-value mapping and the seed-42 five-epoch `full_reward`
provenance are directly checked. This QA does not prove that every experiment
executed that commit; that requires the per-run provenance audit.
