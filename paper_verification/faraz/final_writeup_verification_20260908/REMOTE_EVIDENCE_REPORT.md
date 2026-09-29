# Julia2 remote evidence addendum — 2026-09-09

This addendum records read-only inspection of `julia2` under `/home/s471802/nn-gpt`, `/home/s471802/nn-dataset`, and `/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs`. It does not modify the original audit report.

## Results

| Gap | Remote result | Status |
|---|---|---|
| wide38 task 0 | Task 0 was completed by `2968490_0`, not by replacement array `2968494`; 1 L40S × 1,920 s = 0.5333333333 GPU-hours | VERIFIED; previous total corrected |
| added-seed NNGPT provenance | Six `20260627` `run_state.json` records name `c91714dbe7dad1d02a9080243945bbf8e8ec9300` | base commit VERIFIED; clean execution tree NOT VERIFIED |
| NN Dataset commit | Current remote checkout is `6dbf65b9a3bf6ed93cb1f655d2110276312f1157`; run logs prove use of `/home/s471802/nn-dataset`, but no historical run manifest pins its commit | NOT VERIFIED for experiment-time commit |
| learned Imagenette | Multiple learned one-pattern Imagenette trajectories exist; a Slurm-COMPLETED 1,000-attempt example is identified below. No completed protocol-matched learned one-/four-pattern pair was found | prior blanket absence interpretation corrected; matched pair still NOT VERIFIED |
| 6×800 to 3×400 reason | No primary planning/scheduling record establishing the reason was recovered from the inspected remote home records | NOT VERIFIED |

## wide38 accounting correction

The manifest row with `selection_index=0` names candidate `4pattern_seed42_743`. The result file `results/00_4pattern_seed42_743.json` has `task_id=0` and the same candidate. The stdout for `proxy-wide38-smoke-2968490_0` ends by writing that result. Fresh Slurm accounting returned:

```text
2968490_0|2968490|proxy-wide38-smoke|COMPLETED|1920|...gres/gpu:l40s=1,gres/gpu=1...|2026-07-27T05:25:28|2026-07-27T05:57:28|0:0|jnfat02
```

The successful replacement array `2968494` contains tasks 1–37 only. Its retained 37 rows total 69,423 GPU-seconds. Therefore:

- complete successful wide38: `(69,423 + 1,920) / 3,600 = 19.8175` GPU-hours;
- corrected unique extended total: `(6,789,290 + 1,920) / 3,600 = 1886.4472222222...`, or **1886.45** rounded once.

The earlier 1885.91 versus 1885.92 explanation remains a valid rounding-order explanation for the incomplete 37-row subtotal, but neither was the complete 38-result total.

The first array `2968450_0`–`2968450_37` failed after 2–5 seconds each due to the superseded launch error. These failed attempts are not added here because the previous extended accounting intentionally retained successful final evaluations. Counting them would change the declared scope and must be reported separately if desired.

Recompute with:

```bash
python3 final_writeup_verification_20260908/scripts/remote_recompute_wide38_task0.py
```

## Learned Imagenette evidence

Run `20260605_0925_dscoder_imagenette_h100` provides a direct completed learned one-pattern example:

- Slurm job `2666209` is `COMPLETED`, exit `0:0`;
- its raw `generation_samples.jsonl` contains exactly 1,000 rows and has SHA-256 `9e43736e9a4b1a69c3047d5a620a7ade743a7c9800520ee4fab9127c4dbbe2fc`;
- the config uses dataset `imagenette`, prompt prefix `rl-bb-test1`, and a one-epoch formal evaluator;
- the run state calls it “DS A9 RL on imagenette” and records base commit `c9276599c8dd1df4def862b21dd3a766638bccd5`.

This is historical one-pattern learned evidence, not the requested final matched five-epoch one-/four-pattern comparison. The top-level `/data/.../parallel_runs` inventory contains many Imagenette runs and completed one-pattern trajectories, but no run config with a four-pattern training prefix that forms the requested matched pair. The safe statement remains: **No completed matched learned Imagenette conditions were found in the inspected Julia2 archive evidence scope.**

## NNGPT and NN Dataset provenance

All six remotely retained `run_state.json` files for seeds 114/514/919 and both conditions record `commit_hash=c91714dbe7dad1d02a9080243945bbf8e8ec9300`. This fills the empty Git fields in their generated `run_config.json` only at the submission metadata level. The job used an rsynced `/tmp/.../nn-gpt` work directory, while the archive has no source snapshot or diff. Exact byte identity with a clean commit remains `NOT VERIFIED`.

The seed-114 job stdout states both `nn-dataset root: /home/s471802/nn-dataset` and the imported/fallback paths. The current server checkout is `6dbf65b9a3bf6ed93cb1f655d2110276312f1157`, but Git history contains many commits around the June runs and no retrieved run metadata records the NN Dataset hash. Do not substitute the current checkout for historical provenance.

## Imagenette evidence

Remote `/data` contains real learned one-pattern Imagenette runs. One directly checked completed example is:

- run: `20260605_0925_dscoder_imagenette_h100`;
- `run_state.json`: ended, job `2666209`, H100, commit declaration `c9276599c8dd1df4def862b21dd3a766638bccd5`;
- raw trajectory: 1,000 rows;
- prompt prefix: `rl-bb-test1` (one-pattern learned condition);
- evaluation: Imagenette `trainvaltest`, reward-eval population, formal epoch setting `1`.

Other observed learned one-pattern Imagenette trajectories include completed/raw counts of 1,000, 832, 800, and partial runs. These historical runs do not match the final CIFAR-10 five-epoch, two-condition, matched-seed design. Configuration inspection found Imagenette runs using `rl-bb-test1`; no corresponding completed four-pattern configuration and no matched one-/four-pattern pair were found in the inspected `/data/.../parallel_runs` archive.

Safe statement: **“Historical learned one-pattern Imagenette runs are present on Julia2. No completed matched learned one-/four-pattern Imagenette conditions were found in the inspected Julia2 archive.”** The rule-constrained sampler remains a separate existing result.

## Remaining limitations

- The threshold-sensitivity search is handled in a separate remote evidence addendum.
- The reason for reducing the planned SFT-only scope remains supported by correspondence/delivery prose rather than a recovered primary scheduling record.
- Absence claims are limited to the inspected Julia2 user-owned archive and metadata; the second server was not needed for the Julia2-owned experiment outputs.
