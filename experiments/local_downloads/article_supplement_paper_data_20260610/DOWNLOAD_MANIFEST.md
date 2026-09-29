# Article Supplement Paper Data Download

Downloaded on: 2026-06-10

This directory is a self-contained local copy of the completed experiment data needed for article writing. It intentionally excludes model checkpoints and adapter weights.

## Layout

- `full_runs/`: small run directories copied completely.
  - `20260608_1345_article_A_epoch1_epoch10`
  - `20260609_1020_article_A_epoch1_epoch10_remaining88`
  - `20260608_1345_article_B_single_dual_imagenette`
  - `20260607_1455_gen30min32_formal100_dsqwenolympic`
- `selected_rl_runs/`: large RL run directories copied with only paper-relevant files:
  - `rl_output/generation_samples.jsonl`
  - `rl_output/run_config.json`
  - `rl_output/group_progress.jsonl`
  - `rl_output/group_feedback_samples.jsonl`
  - `rl_output/best_group_feedback.json`
  - `slurm/*.out`
  - `slurm/*.err`
- `docs/`: experiment plans and archive snapshots.
- `CHECKSUMS.sha256`: SHA-256 checksums for all downloaded files.

## Notes

- D seed 114 is kept as a probe run; the aligned multi-seed aggregate is seed 42/123/777.
- Public article text should avoid the label `DeepSeek A9`; use a precise adapter/model description instead.
