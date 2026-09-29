# nn-gpt thesis evidence archive (2026-09-29)

This `archive` branch indexes a filtered set of thesis evidence from the local
machine, Julia2, and workstation. The large raw artifacts are attached to the
repository release `thesis-archive-2026-09-29`; `ASSETS.tsv` gives each asset's
source path, SHA-256, and size. Download an asset from the release and compare
it with `ASSETS.tsv` before use. The original source locations were not changed.

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
- `slurm/raw/julia2-sacct-duplicates-20260929.psv`: scheduler output including
  duplicate allocation records. The fifteen-run external cohort recalculates
  to 1,182.60 allocated GPU-hours in
  `inventory/external_15_gpu_hour_recheck.tsv`.
- `paper_verification/faraz/`: the latest retained verification calculations
  and source reports. Private email messages and drafts are not published here.
- The paper's main code reference is
  `c91714dbe7dad1d02a9080243945bbf8e8ec9300`, which is already present in
  this fork's `experiment/four-pattern-reward-ablation-821f` ref.

## Limits

The release does not contain complete optimizer/trainer checkpoints (over
105 GiB), public base-model caches, or every historical run. It includes the
paper-used A9/A18 and Qwen A7 SFT adapters and only stage adapters that were
fully transferred and validated. The exact dirty-source patch for some runs,
the 507-row historical SFT cycle manifest, and one old Qwen table value's
source were not recovered. See `inventory/julia2_selected_checkpoint_exclusions.tsv`
and `ASSETS.tsv` for the precise retained set.

The `nn-gpt` source and fixed commit are in the fork's normal branches. Git
bundles from other repositories and private correspondence remain in the
local research archive, not in this public fork. This branch is an evidence
index and does not replace the original training directories.
