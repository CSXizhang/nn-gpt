# Reward audit findings for integration

### Claim being checked

Does the numerical `tab:reward_components` agree with the main five-epoch reward implementation at commit `c91714dbe7dad1d02a9080243945bbf8e8ec9300`, including stage scaling, caps, wrapper behavior, and representative logged values?

### Evidence used

- Fixed git object verified with `git cat-file -t`.
- `git show <commit>:ab/gpt/TuneRL.py` and `git show <commit>:ab/gpt/TuneRLSft.py` only.
- Local numbered manuscript table at `/Users/zhangxi/Desktop/example-cvpr/Paper_draft_en.tex:151-173`; its status as the submission-final source is not verified.
- Raw candidate JSONL from the delivery directories and the original archive ZIP; summary Markdown was not used to prove logged values.

### Recompute method

Run `python3 final_writeup_verification_20260908/scripts/rebuild_reward_audit.py`. It verifies the commit object, retrieves both source files through `git show`, parses constants, recursively scans raw JSON records for the four requested logged values, and independently recalculates the dense term from raw candidate fields.

### Exact result

Status: **DISPROVED** for the current numerical table as a whole. Corrected row-level results are in `reward_components_rebuild.csv`. The fixed-commit results include stage-2 group scales `0.20/0.20`, stage-2 backbone scales `0.25/0.25`, formal success `0.08` plus `0.20` when `target_structure_match is not False`, dense clip `[0.02,0.35]`, block archive novelty `0.08`, and a conditional pre-local-competition repeated-block upper cap `2.0`. The final wrapper means the logged reward is not globally restricted to `[-2,2]`.

The raw candidate at the six-seed archive member `1pattern_seed42/generation_samples.jsonl:800` exactly reproduces `r_dense=0.09515141111111111` from `reward_target_value=0.507745` and `frozen_train_acc=0.9533555555555555`. Other raw candidates verify `r_formal_success_signal=0.28`, `r_repeat_family=-0.05500000000000001`, and `r_plain_fuse_penalty=-0.11000000000000001`.

### Difference from Faraz / previous Xi summary

The hypothesized corrections in the verification request are supported for all specifically enumerated issues: `0.70`, `0.95`, `0.20` repeated-block cap, `+0.02` formal-success bonus, dense upper clip `0.22`, and block archive novelty `+0.18` do not describe the fixed-commit stage-2 implementation. The statement that the final reward is simply clipped to `[-2,2]` omits post-clip wrapper logic.

### Safe statement for the paper/email

“At the archived implementation commit, the base TuneRL component sum was clipped to `[-2,2]`, after which the TuneRLSft wrapper applied contract, hygiene, dual-backbone, and compactness adjustments; final logged rewards could therefore fall below `-2`. The stage-2 dense scale was `0.50` with inner clip `[0.02,0.35]`. Formal success contributed `0.08`, with a separate `0.20` added when `target_structure_match is not False`. Stage-2 previous/best group scales were `0.20/0.20`, backbone-group scales were `0.25/0.25`, block archive novelty was `0.08`, and repeated blocks without quality refresh triggered a conditional pre-local-competition upper cap of `2.0`. The paper table should separate raw constants from stage-specific effective formulas.”

### Remaining uncertainty

The raw seed-42 six-seed `run_config.json` records this exact commit, `full_reward`, formal epochs `5`, and `stage2_formal_explore`; it also records `dirty: true`. Whether all reported runs used the same committed content, or what uncommitted difference was present, remains a provenance question. The local table was found in `Paper_draft_en.tex`; it has not been established as the final submitted manuscript source.
