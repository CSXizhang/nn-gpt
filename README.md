# nn-gpt thesis evidence archive (2026-09-29)

This `archive` branch indexes a filtered set of thesis evidence from the local
machine, Julia2, and workstation. The large raw artifacts are attached to the
repository release `thesis-archive-2026-09-29`; `ASSETS.tsv` gives each asset's
source path, SHA-256, and size. Download an asset from the release and compare
it with `ASSETS.tsv` before use. The original source locations were not changed.
`MANIFEST.tsv` also indexes files inside the selected Julia2 tar packages;
`CHECKSUMS.sha256` checks the files tracked directly on this branch. The
release tag is the original publication snapshot; this branch is the current
data-only index and takes precedence over that tag's older README.

The run selection follows Faraz's July–September correspondence. See
`inventory/SELECTION_RULES.md` and
`inventory/nn-gpt-julia2-selected-runs-20260929.tsv`. There were 337 Julia2
parallel-run directories; 45 were selected for detailed archival. Failed or
requeued jobs were retained when their generated data belongs to a usable
analysis window. Diagnostic, empty, and superseded attempts remain represented
in the scheduler inventory and are not copied as raw run directories.

## Key evidence

- `inventory/primary_8030_raw_check.tsv`: twelve main DeepSeek CIFAR-10 runs,
  9,600 raw candidate records and 8,030 formal-success records.
- `inventory/archived_run_coverage.tsv`: per-run raw/evaluation/config/adapter
  and job mapping. Blank provenance is unresolved, not inferred.
- `inventory/config_provenance.tsv`: exact archived JSON configuration or
  manifest path for 43 selected runs. The two proxy runs use their archived
  preparation/evaluation scripts and candidate manifests instead; see
  `inventory/CONFIG_COVERAGE.md`.
- `datasets/derived/sft_cycle_507/`: 507 recovered four-pattern SFT cycle
  generated code files paired one-to-one with their CIFAR-10 epoch-1 evaluation
  records; the source-path and SHA-256 manifest is beside the tar package.
- `slurm/raw/julia2-sacct-duplicates-20260929.psv`: scheduler output including
  duplicate allocation records. The fifteen-run external cohort recalculates
  to 1,182.60 allocated GPU-hours in
  `inventory/external_15_gpu_hour_recheck.tsv`.
- `paper_verification/faraz/`: the latest retained verification calculations
  and source reports. Private email messages and drafts are not published here.
- The paper's main code reference is
  `c91714dbe7dad1d02a9080243945bbf8e8ec9300`, which is already present in
  this fork's `experiment/four-pattern-reward-ablation-821f` ref. Six `nn-gpt`
  Git bundles from the local machine and both clusters are release assets.

## Limits

This archive is for paper data verification, not training continuation.
Model weights, including the A9/A18/Qwen A7 SFT adapters and selected RL stage
adapters, are deliberately outside the data-only release. Their source paths,
sizes, and checksums are recorded in `inventory/EXCLUDED_WEIGHTS.tsv`; the
original cluster files were not changed. The three derived `ab.nn.db` SQLite
snapshots were also removed from the release because the paper-used raw results
and configurations are preserved separately; their former source paths and
checksums remain in `MANIFEST.tsv` with `excluded_derived_cache` status.
Public base-model caches and other
re-downloadable dependencies are also excluded. The exact dirty-source patch
for some runs and one old Qwen table value's source were not recovered. The
507 cycle records were recovered after the first publication; a training-time
dataset manifest proving exactly which rows the trainer consumed remains
unavailable. See `inventory/CONFIG_COVERAGE.md`
for the practical effect on paper verification.

The `nn-gpt` source and fixed commit are in the fork's normal branches.
`nn-dataset` history bundles triggered a credential-like pattern during
screening and were not published; the public `nn-dataset` fork and the scanned
worktree snapshot preserve the usable code. Private correspondence is also
not published. See `inventory/PUBLIC_EXCLUSIONS.md` for these boundaries.
This branch is an evidence index and does not replace the original training
directories.
