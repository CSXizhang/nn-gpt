# Evidence-based run selection (2026-09-29)

The selection was rebuilt after reading Faraz's correspondence, especially the messages of 2026-07-27, 2026-09-21, 2026-09-26, and Xi's 2026-09-27 reply (Gmail message IDs `19fa5b754d491b1e`, `1a0c50c0dbebf782`, `1a0de2ce30fa3a79`, `1a0e13e9c083fd7c`). The September 27 reply corrects the earlier proposal that every retained run must have a `COMPLETED` Slurm state: a retained 800-attempt analysis window can include a continuation or end after manual termination. SFT-only controls use 400 attempts per seed.

`nn-gpt-julia2-selected-runs-20260929.tsv` is the authoritative directory list for this archive. Its 45 entries cover the twelve primary DeepSeek CIFAR-10 runs, fifteen external-cohort logical runs (six ablations, three Qwen, six CIFAR-100; Qwen seed 777 also has a pre-resume backup directory), six SFT-only controls, top-20 and wide-38 proxy evaluations, June 7/8 dual-backbone provenance, one-vs-ten-epoch reevaluation, rule-constrained sampler evaluations, the one-epoch independent-generation audit, and one historical 1,000-row learned Imagenette run directly cited in the September Faraz reply. The Imagenette run is historical evidence, not a matched five-epoch primary result.

Do not equate scheduler state with usefulness. The no-diversity seed-123 result has a `FAILED` final state yet supplies a retained 800-row analysis window. Qwen seed 777 has a timed-out segment followed by a completed continuation. CIFAR-100 four-pattern runs include preemption/requeue allocations and usable 800-row results after manual termination. Those records remain selected. Zero-sample cancellations, smoke tests, unsuccessful diagnostics, and superseded partial attempts are omitted from detailed raw transfer; their accounting and source paths remain in `slurm/raw`, `slurm/parsed`, `julia2_parallel_files.tsv`, and `nn-gpt-julia2-job-run-map-20260929.tsv`.

The Faraz packages under `paper_verification/faraz/` and `paper_verification/SFT_RL 2.zip` are retained as distinct delivered snapshots. Later email corrections take precedence over the older 2026-07-03 `DATA_STATUS` inside that ZIP: Qwen three-seed RL and wide-range proxy evaluations subsequently completed. The September 27 `faraz_final_verification_package.zip` is targeted verification, not a complete artifact release.

The filtered Julia2 run archives exclude model weights and checkpoints because
this publication is for paper data analysis rather than training continuation.
The original 44 selected directories contain about 105.94 GiB of checkpoint
files. Their source paths and sizes remain in `julia2_parallel_files.tsv`;
`EXCLUDED_WEIGHTS.tsv` records the separately identified SFT and stage adapters.
The original cluster files were not changed.
