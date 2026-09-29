# Remote threshold-sensitivity evidence search

Audit date: 2026-09-09 (Asia/Shanghai)

## Result

**PARTIAL SEARCH / NOT VERIFIED.** No generator or result artifact for `threshold_sensitivity_70_80_90` was found by the completed searches, but several bounded content passes did not complete. Therefore this evidence cannot establish a complete server-wide absence. The concentration field, generating function/source line, and “at most four of twelve runs” count remain `NOT VERIFIED`.

## Completed searches

- Filename search across `/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs`: exit 0, no match.
- NNGPT git-history pickaxe for the exact artifact name: exit 0, no commit.
- Nearby `paper_followup_20260727` listing: exit 0; its four files do not include the requested analysis.
- Archive enumeration under `/data/42-julia-hpc-ai-cv-students/s471802` and `/home/s471802`: exit 0 and **eight files found**. Seven are dataset or installed-package assets. The plausible runtime archive `/home/s471802/tunerl_runtime_5007863e.tgz` (SHA-256 `09ed4995f349cf1154eb1f05b3ddade21b5a4229fe6258808a897c37794fb628`) was checked by member name; grep exit 1 means no matching member name.

The eight enumerated archives are preserved verbatim in `raw_search/archive_search.stdout.txt` and listed in `remote_threshold_manifest.json`.

## Incomplete or nonconclusive searches

- Broad content search: exit 124. Its emitted paths are tokenizer-vocabulary false positives; timeout prevents a complete no-match conclusion.
- Source-only content search: exit 124 with no emitted match; timeout prevents a complete no-match conclusion.
- Summary-content search: interrupted before a status file was written; empty stdout is not treated as a completed no-match result.
- Initial shell-history search: exit 2 because `/home/s471802/.zsh_history` was absent. Its empty stdout is not treated as conclusive. The current reproduction script checks readability per history file and records the remote command status, but that corrected command was not substituted for the retained raw result.

Large model-weight bodies and unbounded raw `generation_samples.jsonl` contents were excluded. These scope limits mean this is a scoped partial search, not proof that the analysis never existed.

## Safe paper/email statement

`DO NOT CLAIM YET`: the available local evidence and completed portions of the Julia2 search do not identify whether the threshold analysis used `graph_hash` (Exact Forward-Graph) or `actual_structure_signature` (Family+Block), and do not independently reproduce the “at most four of twelve runs” count.

Continue to use distinct terminology:

- `actual_structure_signature`: Family+Block Eff./Top-1
- `graph_hash`: Exact Forward-Graph Eff./Top-1

Raw command, stdout, stderr, and status evidence is under `remote_evidence/threshold/raw_search/`. The current read-only reproduction script is `scripts/remote_threshold_search.sh`; a future run must be assessed from each recorded status and must not convert timeout/error/missing-status cases into clean no-match results.
