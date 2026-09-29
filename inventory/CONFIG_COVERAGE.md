# Configuration coverage for selected paper runs

`config_provenance.tsv` points to 211 exact configuration, script, and
candidate-manifest members in the release packages. All 45 selected run
directories have at least one configuration source. Forty-two have one or more
`run_config.json` files; the separate combined one-epoch audit has a
`run_manifest.json`; two proxy analyses are specified by archived scripts and
candidate manifests. The TSV is a search index, not a replacement for the
complete JSON or scripts inside the tar packages.

The training `run_config.json` files record the code commit, branch, dirty
worktree flag, seed, model and tokenizer source, adapter initialization path,
prompt and sampling configuration, reward variant, evaluator dataset/split,
and runtime settings. Auxiliary evaluation JSONs record their source candidate
file, evaluation settings, and commit where available. Use the archived member
at `config_path` for the exact values; a blank TSV cell means the field is
not asserted by this index.

The two proxy directories have no `run_config.json` and no proven exact code
commit. For the top-20 reevaluation, read `scripts/prepare_manifest.py`,
`scripts/run_proxy_candidate.py`, `proxy_manifest_top20.jsonl`, and the
archived Slurm logs. For the wide-38 reevaluation, read
`prepare_proxy_wide_manifest.py`, `scripts/run_proxy_candidate.py`,
`proxy_manifest_wide38.jsonl`, and the Slurm logs. These files identify the
source candidates, selection rule, evaluation implementation, and results;
the precise source commit remains unresolved.

The combined one-epoch SFT-only/RL-after audit is a job manifest and script
snapshot, not a standalone raw-result directory. Its `baseline/run_manifest.json`
names commit `c91714dbe7dad1d02a9080243945bbf8e8ec9300`, seed 42,
the four conditions, generation job IDs, evaluation job 2741706, and summary
job 2741707. `baseline/slurm/eval.sh` records the four candidate source paths,
dataset and one-epoch evaluation setting. The directory has no copied
evaluation result of its own; related calculations are in the Faraz evidence
package. Do not interpret the three files in this directory as complete raw
evaluation output.

For paper table recomputation, use the candidate JSONL and evaluator outputs
in the selected-run packages, the source calculations in
`paper_verification/faraz/`, and the Slurm export. The 29 stage adapters and
three SFT adapters are deliberately excluded, so retraining or regenerating
LLM samples from saved weights is outside this archive's scope. Historical
dirty-working-tree patches for some runs were not recovered, so matching a
recorded commit alone may not recreate their executable code exactly.
