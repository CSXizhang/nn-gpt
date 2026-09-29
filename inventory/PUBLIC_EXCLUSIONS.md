# Public archive exclusions

- `nn-dataset` Git bundles from the local machine, Julia2, and workstation
  were withheld because an archive-history scan matched a credential-like
  pattern. No matched value is included here. The separate public
  `CSXizhang/nn-dataset` fork and the screened worktree snapshot remain
  available for source review.
- Private messages, email drafts, and older handover ZIP snapshots were not
  published. The latest verification calculations and the final Faraz package
  are included.
- Public base-model caches, official dataset downloads, package caches, and
  virtual environments can be retrieved again and are not release assets.
- The three Julia2/workstation `ab.nn.db` SQLite snapshots were removed from
  the release as derived caches outside the paper-data-only scope. Their
  former source paths and checksums remain in `MANIFEST.tsv` with
  `excluded_derived_cache` status. Original server databases were not changed.
- All model weights and checkpoints are outside this paper-data-only archive,
  including the 29 selected stage adapters and three A9/A18/Qwen A7 SFT
  adapters previously attached to the release. `EXCLUDED_WEIGHTS.tsv` records
  their original archive mapping, size, and checksum; the cluster source files
  were not altered. Trainer/optimizer and repeated checkpoint paths are in
  `julia2_selected_checkpoint_exclusions.tsv`.
- Empty, diagnostic, and superseded run directories were filtered according
  to `SELECTION_RULES.md`; the complete Slurm accounting export retains their
  job history where the scheduler still held it.
