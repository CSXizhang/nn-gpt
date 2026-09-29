# Slurm accounting

`raw/julia2-sacct-duplicates-20260929.psv` is the unmodified, headerless Julia2 output of:

```text
sacct -u s471802 --duplicates -S 2025-01-01 -E 2026-09-30 --format=JobID,JobName%100,State,ExitCode,Submit,Start,End,ElapsedRaw,AllocTRES%150,ReqTRES%150,NodeList%80,Partition,Restarts -P -n
```

The thirteen columns are `JobID|JobName|State|ExitCode|Submit|Start|End|ElapsedRaw|AllocTRES|ReqTRES|NodeList|Partition|Restarts`. There are 8,280 rows. `--duplicates` matters for requeues and preemptions. `parsed/julia2_all_jobs.csv` adds a header; `parsed/julia2_project_jobs.csv` joins job IDs inferred from run Slurm filenames. The inferred mapping covers the twelve primary and all 22 job IDs named in Faraz's fifteen-run external accounting, but should not be treated as a complete mapping for every historical attempt. The raw export covers all returned jobs for this account/date range, not a proof that the scheduler retained every job ever submitted.

The workstation has no accessible `sacct`/`squeue` CLI. Its work was launched through Kubernetes manifests and its historical logs are under `experiments/workstation-prototype/` and `configs/workstation-root/`.
