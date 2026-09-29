# Six-seed CIFAR-10 RL allocated GPU-hours

GPU-hours are Slurm allocation hours: elapsed wall time multiplied by allocated GPUs.
They are not device-utilisation measurements. Cancelled zero-sample attempts and pending jobs are excluded.

| Condition | Seed | Job | Hardware | GPUs | Elapsed | GPU-hours |
|---|---:|---:|---|---:|---:|---:|
| 1pattern | 42 | 2723084 | H100 | 4 | 17:04:21 | 68.29 |
| 1pattern | 114 | 2776908 | H100 | 4 | 09:50:54 | 39.39 |
| 1pattern | 123 | 2723085 | H100 | 4 | 23:06:23 | 92.43 |
| 1pattern | 514 | 2776910 | H100 | 4 | 11:06:49 | 44.45 |
| 1pattern | 777 | 2723086 | H100 | 4 | 18:02:23 | 72.16 |
| 1pattern | 919 | 2777842 | H100 | 4 | 14:19:35 | 57.31 |
| 4pattern | 42 | 2697817 | L40S | 4 | 1-15:07:06 | 156.47 |
| 4pattern | 114 | 2776904 | H100 | 4 | 01:57:05 | 7.81 |
| 4pattern | 123 | 2697840 | H100 | 4 | 21:43:00 | 86.87 |
| 4pattern | 514 | 2776906 | H100 | 4 | 11:43:32 | 46.90 |
| 4pattern | 777 | 2697842 | L40S | 4 | 2-01:22:03 | 197.47 |
| 4pattern | 919 | 2777817 | H100 | 4 | 09:31:49 | 38.12 |

## Totals

| Group | Jobs | GPU-hours |
|---|---:|---:|
| 1pattern | 6 | 374.03 |
| 4pattern | 6 | 533.64 |
| H100 | 10 | 553.72 |
| L40S | 2 | 353.94 |
| all | 12 | 907.67 |
