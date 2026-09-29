# Provenance findings for items 2, 7, and 8

## Item 2 — What 30.30% means

Status: **VERIFIED; prior wording is incorrect and conflates structural concentration with performance.**

### Claim being checked

Whether `30.30%` is performance or a structural top-1 concentration, and whether the new five-epoch SFT-only four-pattern experiment reproduces it under the same field.

### Evidence used

- The ZIP `/Users/zhangxi/code/RL/faraz_delivery_20260726/article_raw_data_archive_20260718_resend_20260726.zip` contains `article_raw_data_archive_20260704/03_independent_generation_audit/sft_only_heldout_test/`, including the raw JSONL, run config, baseline table, and structural summary.
- The raw run config records one-epoch held-out CIFAR-10 evaluation, split `trainvaltest`, and 200 candidates split across the one-pattern and four-pattern settings.
- The older manuscript at `/Users/zhangxi/Desktop/example-cvpr/Paper_draft_en.tex` does not contain 30.30%. Its one-epoch four-pattern SFT row reports 23/100 correctness (lines 247, 256–266), while its structural table reports module top-1 39.13% and exact graph top-1 4.35% among 23 formal successes (lines 347–357). These are different figures.
- The new five-epoch SFT-only raw files were independently recomputed using `api_result.formal_success_candidate == true`; the formal-success definition matches `summarize_rows()` in `/Users/zhangxi/code/RL/faraz_followup_delivery_20260808/scripts/recompute_final_summaries.py`, lines 47–64.

### Exact recomputation

Original one-epoch independent audit: four-pattern has 100 attempted candidates and 99 formal successes. The exact table artifact is ZIP member `article_raw_data_archive_20260704/03_independent_generation_audit/sft_only_heldout_test/baseline_table.md`; line 1 names the column **CNN top-1 share**, and line 3 is row `sft_only_fourpattern_current` with value `0.3030`. The unrounded companion field is ZIP member `article_raw_data_archive_20260704/03_independent_generation_audit/sft_only_heldout_test/structural_diversity_summary.json`, JSON path `sft_only_fourpattern_current.cnn.top1_share`, value `0.30303030303030304`.

The raw input is ZIP member `article_raw_data_archive_20260704/03_independent_generation_audit/sft_only_heldout_test/generation_samples.jsonl`, SHA-256 `73d03261218132a94e4eb3bb1bdd9b79d3b77d706767d5d29b7bdab0eeb49209`. Recompute selection is exactly `setting == "sft_only_fourpattern_current"`, followed by `api_result.formal_success_candidate is True`. Among those 99 rows, `api_result.cnn_signature` has dominant count 30, so top-1 share is exactly `30 / 99 = 0.30303030303030304 = 30.303030303030304%`. `family_hash` has the same 30/99 distribution in this archive. The exact forward graph (`graph_hash`) is 3/99 = 3.0303%, and `actual_structure_signature` is 2/99 = 2.0202%. Thus the original 30.30% is not accuracy and is not graph/Family+Block concentration.

| Condition | Seed | Attempted | Formal-success denominator | `family_hash` top-1 | `cnn_signature` top-1 | `actual_structure_signature` top-1 |
|---|---:|---:|---:|---:|---:|---:|
| four-pattern SFT-only, 5 epoch | 42 | 400 | 398 | 120/398 = 30.15075376884422% | 120/398 = 30.15075376884422% | 6/398 = 1.507537688442211% |
| four-pattern SFT-only, 5 epoch | 123 | 400 | 397 | 122/397 = 30.730478589420655% | 122/397 = 30.730478589420655% | 9/397 = 2.2670025188916875% |
| four-pattern SFT-only, 5 epoch | 777 | 400 | 399 | 119/399 = 29.82456140350877% | 119/399 = 29.82456140350877% | 5/399 = 1.2531328320802004% |

Thus the new data directly establish a roughly 30% **structural concentration** in `family_hash`/`cnn_signature`; they do not establish a 30% accuracy or performance value. They also show that this number cannot mean `actual_structure_signature` in the new dataset.

### Difference from previous Xi summary

`FINAL_RESULTS.md` says the four-minus-one accuracy difference is -1.07 percentage points and immediately says “the earlier one-epoch 30.30% four-pattern audit does not persist.” The original raw artifact now proves that 30.30% was CNN-signature top-1 concentration, not an accuracy result. The sentence compares unlike metrics and is wrong. The new five-epoch data also reproduce approximately 30% CNN/family top-1 concentration in all three seeds.

### Safe statement

The safe local-evidence statement is:

> The original one-epoch independent four-pattern SFT audit's 30.30% is the dominant CNN-signature share: 30 of 99 formal-success candidates. The five-epoch four-pattern SFT-only runs obtain 30.1508%, 30.7305%, and 29.8246% for the same CNN-signature top-1 statistic. It is a structural-concentration metric, not performance. Xi's August 8 “does not persist” wording is incorrect because it compares this structural share with an accuracy difference.

Auditable classification: **`30.30% = STRUCTURAL CONCENTRATION`** (`cnn_signature` top-1; 30/99 formal-success candidates).

## Item 7 — Imagenette

Status: **PARTIAL**.

### Directly verified sampler result

The supplied archive contains a complete rule-constrained sampler Imagenette evaluation:

- Raw: `/Users/zhangxi/code/RL/delivery_20260727/01_rule_sampler/raw/imagenette/generation_samples.jsonl`
- Config: same directory, `run_config.json`
- Attempted candidates: 100
- Formal successes: 99
- Formal-success rate: 99%
- Protocol: one epoch, `trainvaltest`, held-out Imagenette test set; the config identifies `imagenette-train[7500]`, `imagenette-train[1969]`, and `imagenette-test[3925]`.

The delivered recomputation implementation is `/Users/zhangxi/code/RL/delivery_20260727/01_rule_sampler/audit_rule_sampler_3dataset.py`; it filters on `formal_success_candidate` and rejects any input other than 100 attempts with an accuracy for every formal success.

### Learned Imagenette evidence

The local June archive index reports historical **one-pattern learned** Imagenette RL runs. For example, the index reports a completed 1000-sample DeepSeek run (`20260605_0925_dscoder_imagenette_h100`) and Qwen/Mistral runs with different completion/failure states. This is index-level evidence, not a locally supplied raw completion package for that run. These records belong to the old three-model one-pattern matrix and are not the requested matched one-pattern-versus-four-pattern final experiment.

No completed learned four-pattern Imagenette package, and no completed matched learned one-pattern/four-pattern Imagenette pair with the final CIFAR-10-style five-epoch design, matched seeds, and corresponding raw run packages, was found in:

- `/Users/zhangxi/code/RL/delivery_20260727`
- `/Users/zhangxi/code/RL/faraz_delivery_20260726`
- `/Users/zhangxi/code/RL/faraz_followup_delivery_20260808`
- `/Users/zhangxi/code/RL/reply_20260802`
- `/Users/zhangxi/code/RL/nn-gpt/downloads/article_supplement_paper_data_20260610`
- `/Users/zhangxi/code/RL/nn-gpt/run_archive_index.md`

Safe conclusion: **No completed matched learned Imagenette conditions were found in the supplied experiment archive.** This does not negate the existing sampler result or the historical one-pattern learned runs, and it does not prove no such run exists outside the supplied local archive.

Whether no additional Imagenette run is planned before submission is a planning decision. No explicit current planning record supporting that statement was found: **DO NOT CLAIM YET; confirm manually.**

## Item 8 — Code availability

See `code_availability_check.md`. Repository URL and fixed commit are verified. The commit assignment for the six-seed cohort is archive-supported but not raw-config-complete; the historical NN Dataset commit is **NOT VERIFIED**.
