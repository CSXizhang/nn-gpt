# Faraz consolidated open-items package (2026-08-02)

This directory contains only files intended for delivery. The email draft is intentionally not stored here.

## Contents

- `01_experiment_status.md`: SFT-only, CIFAR-100 four-pattern, and Imagenette status.
- `02_reward_configuration_audit.md`: source-verified reward values and manuscript discrepancies.
- `03_reward_source_TuneRL_c91714db.txt`: exact source excerpts from the main-run commit.
- `04_reward_source_TuneRLSft_c91714db.txt`: exact SFT wrapper/cap excerpts from the same commit.
- `05_split_and_signature_confirmation.md`: split provenance and the two distinct structure metrics.
- `06_extended_gpu_hours.csv`: consolidated compute-cost summary.
- `07_extended_gpu_hours.md`: accounting scope and interpretation.
- `08_sacct_records.csv`: retained Slurm records used for the new cohorts.
- `09_build_gpu_hours.py`: recomputation script for `08_sacct_records.csv`.
- `CHECKSUMS.sha256`: SHA-256 checksums.

## Accounting convention

GPU-hours are allocated GPU-hours: elapsed Slurm wall time multiplied by allocated GPU count. They are not device-utilisation hours. Failed diagnostic retries are excluded. A final useful job manually stopped after the aligned 800-row target is retained and explicitly marked.

## Important correction

The manuscript reward-component table should not be accepted as currently written. Several weights/caps differ from commit `c91714dbe7dad1d02a9080243945bbf8e8ec9300`, and post-processing caps can make the final raw reward lower than -2.
