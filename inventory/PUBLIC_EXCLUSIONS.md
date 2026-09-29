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
- Complete trainer/optimizer checkpoints and repeated snapshots exceed
  105 GiB. The 29 selected stage adapters and three paper-used SFT adapters
  are release assets; individual omitted checkpoint paths are in
  `julia2_selected_checkpoint_exclusions.tsv`.
- Empty, diagnostic, and superseded run directories were filtered according
  to `SELECTION_RULES.md`; the complete Slurm accounting export retains their
  job history where the scheduler still held it.
