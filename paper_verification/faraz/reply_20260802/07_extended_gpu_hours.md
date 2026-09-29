# Extended allocated GPU-hours

All costs below are retained in Slurm accounting. GPU-hours are elapsed allocation time multiplied by allocated GPU count.

| Scope | GPU-hours | Notes |
|---|---:|---|
| six-seed CIFAR-10 RL | 907.67 | 12 jobs; already independently reproduced |
| reward ablation: full reward | 440.81 | three jobs already included in the six-seed four-pattern cohort |
| reward ablation: no diversity bonus | 216.90 | three useful final jobs |
| reward ablation: no repeat penalty | 276.78 | three useful final jobs |
| reward ablation: all nine rows | 934.49 | includes the 440.81 overlap |
| reward ablation: incremental unique cost | 493.68 | six non-overlapping ablation jobs |
| Qwen CIFAR-10, three seeds | 234.96 | includes seed777 initial timeout segment plus successful continuation |
| CIFAR-100 one-pattern, three seeds | 210.70 | excludes failed old seed123 job 2878976 |
| proxy top20 unfrozen20 | 19.63 | successful final 20-task array used by earlier manuscript analysis |
| proxy wide38 unfrozen20 | 19.28 | successful final 38-task array used by 20260727 package |
| unique total, including both proxy analyses | 1885.91 | no reward full-reward double counting |

Accounting decisions:

- Cancelled zero-sample attempts, superseded pending jobs, smoke jobs, and failed diagnostic proxy retries are excluded.
- Reward-ablation job `2751655` is retained: it was manually stopped after the aligned 800 usable rows were produced, although Slurm records `FAILED 1:0`.
- Qwen seed777 uses both `2791929` (760/800, timeout) and `2793568` (continuation to 800/800); both allocations contributed to the final condition.
- CIFAR-100 old seed123 job `2878976` is excluded from formal condition cost because it was superseded after the shared filesystem outage. It may be reported separately as wasted allocation if desired.
- Proxy costs report successful final evaluations. Failed retries are diagnostic/wasted allocation and are not mixed into condition cost.
