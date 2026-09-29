# Code availability and fixed-commit check

> Updated 2026-09-09 with read-only Julia2 run-state evidence. Repository/commit conclusions below retain the clean-tree caveat.

Status: **PARTIAL**. Repository identity and fixed-commit availability are directly verified. Remote `run_state.json` fills the previously blank base-commit declarations for the six added-seed runs, but no archived source snapshot/diff establishes a clean fixed tree; the historical NN Dataset commit is also not pinned.

## NNGPT repository

| Check | Result | Evidence |
|---|---|---|
| Canonical repository | `https://github.com/ABrain-One/nn-gpt.git` | Local `upstream` fetch/push remote from `git remote -v` |
| User fork | `https://github.com/CSXizhang/nn-gpt.git` | Local `origin` fetch/push remote |
| Fixed object | `c91714dbe7dad1d02a9080243945bbf8e8ec9300` is a `commit` | `git cat-file -t` |
| Commit subject/date | `Simplify reward memory preflight`, 2026-06-08T22:39:18+02:00 | `git show -s` |
| Parent | `dd7ebb5d0ee2070a1d8b4d52d43c1c624650a842` | `git show -s` |
| Containing local/remote experiment branch | `experiment/four-pattern-reward-ablation-821f` | `git branch -a --contains` |
| Current checkout relation | Current HEAD is `6e7ea0949584e77448c47a66fd5fe98808d0b751`; fixed commit is **not** its ancestor | `git merge-base --is-ancestor` returned false |

The fixed commit contains `ab/gpt/TuneRL.py`, `ab/gpt/TuneRLSft.py`, `ab/gpt/util/Reward.py`, and the reward-ablation Slurm launcher. Source reconstruction for the reward table must therefore use `git show c91714dbe7...:<path>`, as requested. The commit itself changed only `ab/gpt/util/Reward.py`; its existence alone does not prove that every experiment used it.

## Six-seed raw experiment provenance

The archive `/Users/zhangxi/code/RL/faraz_delivery_20260726/article_raw_data_archive_20260718_resend_20260726.zip` contains all 12 `generation_samples.jsonl` files and all 12 matching `run_config.json` files under `article_raw_data_archive_20260704/01_six_seed_robustness/`.

The raw configs divide into two groups:

- Seeds 42/123/777 for both one-pattern and four-pattern record `git.commit = c91714dbe7...`, but also record `git.dirty = true`. Thus `c91714dbe7...` is the base commit, while the raw configs do **not** prove the executed source tree was byte-identical to that commit.
- Seeds 114/514/919 for both conditions contain empty Git branch/commit fields. Six remotely retained matching `run_state.json` files each record `commit_hash=c91714dbe7...`, strengthening base-commit provenance at submission-metadata level.

All twelve raw configs independently agree on `reward.variant = full_reward`, evaluator `formal_epochs = "5"`, and `NNGPT_RL_RESUME_STAGE = stage2_formal_explore`. This directly ties the cohort to the main five-epoch full-reward execution mode. It does not resolve the dirty/missing source-tree identity described above.

The remote run states and derived archive records attribute all twelve runs to `c91714dbe7...`. The six new run states are primary run metadata rather than source snapshots: jobs executed an rsynced `/tmp/.../nn-gpt` tree, and no retained diff proves byte identity. The strongest safe statement is: **the cohort is archived against base commit `c91714dbe7...`; exact clean-tree provenance remains unverified (dirty=true in six raw configs; source snapshots/diffs absent for the other six).**

Other direct raw config evidence:

- The rule-constrained Imagenette evaluation config at `/Users/zhangxi/code/RL/delivery_20260727/01_rule_sampler/raw/imagenette/run_config.json` records the full NNGPT commit `c91714dbe7dad1d02a9080243945bbf8e8ec9300`.
- The same is true for the delivered rule-sampler source config and other delivered rule-sampler dataset configs.

It must not be described as the unique code commit for all experiments in the paper.

Git ancestry checks directly verify that the semantic split fix `6186ebe2c14e23e21444954b4d533abf0e603db4` and smoke-prevalidation weights fix `85c3b6ed10868a47fdbfd268fb0021c0c217a2d3` are ancestors of `c91714dbe7...`.

## NN Dataset repository

| Repository | Commit | Role | Evidence |
|---|---|---|---|
| NNGPT (`ABrain-One/nn-gpt`) | `c91714dbe7dad1d02a9080243945bbf8e8ec9300` base commit; exact execution trees **NOT VERIFIED** | Fixed reward/runtime source; base provenance for six-seed CIFAR-10 | Git object/remotes; six raw configs name commit with dirty flag; six matching remote run states name commit; no source snapshot/diff |
| NN Dataset (`ABrain-One/nn-dataset`) | **NOT VERIFIED** | Dataset loader/evaluator dependency | Canonical remote is verified; current local/remote checkouts differ and neither can substitute for a historical hash; retrieved run logs name `/home/s471802/nn-dataset` but pin no commit |

The current NN Dataset checkout cannot be substituted for historical experiment provenance. The final paper/email should identify its experiment commit only after a raw environment manifest, lock record, or run config that pins it is supplied.
