#!/usr/bin/env python3
"""Read-only provenance and terminology checks for final-paper verification."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import zipfile
from collections import Counter
from pathlib import Path


ROOT = Path("/Users/zhangxi/code/RL")
NNGPT = ROOT / "nn-gpt"
NNDATASET = ROOT / "nn-dataset"
SFT = ROOT / "faraz_followup_delivery_20260808/sft_only/raw"
SAMPLER = ROOT / "delivery_20260727/01_rule_sampler/raw/imagenette"
COMMIT = "c91714dbe7dad1d02a9080243945bbf8e8ec9300"
RAW_ARCHIVE = ROOT / "faraz_delivery_20260726/article_raw_data_archive_20260718_resend_20260726.zip"


def git(repo: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], text=True, capture_output=True, check=False
    )
    if check and result.returncode:
        raise RuntimeError(result.stderr.strip())
    return result.stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sft_concentration() -> list[dict]:
    result = []
    for run_dir in sorted(SFT.glob("4pattern_seed*")):
        path = run_dir / "generation_samples.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        formal = [
            r["api_result"]
            for r in rows
            if r.get("api_result", {}).get("formal_success_candidate") is True
        ]
        entry = {"run": run_dir.name, "attempted_n": len(rows), "formal_success_n": len(formal)}
        for key in ("family_hash", "cnn_signature", "actual_structure_signature"):
            counts = Counter(str(r[key]) for r in formal if r.get(key) not in (None, ""))
            entry[key] = {
                "covered_n": sum(counts.values()),
                "top1_count": max(counts.values()),
                "top1_share": max(counts.values()) / sum(counts.values()),
            }
        entry["sha256"] = sha256(path)
        result.append(entry)
    return result


def find_named_or_text(root: Path, needles: tuple[str, ...]) -> list[str]:
    pattern = "|".join(needles)
    result = subprocess.run(
        [
            "rg", "-l", "-i", pattern, str(root),
            "--glob", "!**/.git/**", "--glob", "!**/*.jsonl",
            "--glob", "*.py", "--glob", "*.json", "--glob", "*.csv",
            "--glob", "*.md", "--glob", "*.tex", "--glob", "*.txt",
            "--glob", "!**/final_writeup_verification_20260908/**",
        ],
        text=True, capture_output=True, check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr.strip())
    return sorted(line for line in result.stdout.splitlines() if line)


def find_zip_members(roots: tuple[Path, ...], needles: tuple[str, ...]) -> list[str]:
    hits = []
    lowered = tuple(n.lower() for n in needles)
    for root in roots:
        if not root.exists():
            continue
        for archive in root.rglob("*.zip"):
            with zipfile.ZipFile(archive) as handle:
                for info in handle.infolist():
                    matched = any(n in info.filename.lower() for n in lowered)
                    if (not matched and info.file_size <= 20_000_000
                            and Path(info.filename).suffix.lower() in {".py", ".json", ".csv", ".md", ".tex", ".txt"}
                            and not info.filename.lower().endswith(".jsonl")):
                        body = handle.read(info).decode("utf-8", errors="replace").lower()
                        matched = any(n in body for n in lowered)
                    if matched:
                        hits.append(f"{archive}!{info.filename}")
    return sorted(hits)


def zip_inventory(roots: tuple[Path, ...]) -> list[str]:
    return sorted(
        str(path)
        for root in roots if root.exists()
        for path in root.rglob("*.zip")
    )


def summarize_sampler_raw() -> dict:
    path = SAMPLER / "generation_samples.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    formal = [
        row for row in rows
        if row.get("api_result", {}).get("formal_success_candidate") is True
    ]
    return {
        "attempted_n": len(rows),
        "formal_success_field": "api_result.formal_success_candidate is True",
        "formal_success_n": len(formal),
        "formal_success_rate": len(formal) / len(rows),
    }


def six_seed_provenance_rows() -> list[dict]:
    path = ROOT / "faraz_delivery_20260726/run_provenance_audit_20260726.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if row.get("cohort") == "six_seed_robustness"]
    return [
        {
            "condition": row.get("condition_or_model"),
            "seed": row.get("seed"),
            "reported_run_id": row.get("run_id"),
            "archive_commit": row.get("primary_commit"),
            "config_commit": row.get("run_config_commit"),
            "config_commit_dirty": row.get("run_config_dirty"),
            "commit_evidence": row.get("commit_evidence"),
        }
        for row in selected
    ]


def six_seed_raw_zip_configs() -> list[dict]:
    prefix = "article_raw_data_archive_20260704/01_six_seed_robustness/"
    result = []
    with zipfile.ZipFile(RAW_ARCHIVE) as handle:
        for name in sorted(handle.namelist()):
            if not (name.startswith(prefix) and name.endswith("/run_config.json")):
                continue
            config = json.loads(handle.read(name))
            git_info = config.get("git") if isinstance(config.get("git"), dict) else {}
            result.append({
                "member": name,
                "git_commit": git_info.get("commit"),
                "git_dirty": git_info.get("dirty"),
                "branch": git_info.get("branch"),
                "reward_variant": (config.get("reward") or {}).get("variant"),
                "formal_epochs": (config.get("evaluator") or {}).get("formal_epochs"),
                "resume_stage": ((config.get("reward") or {}).get("env") or {}).get("NNGPT_RL_RESUME_STAGE"),
            })
    return result


def independent_generation_audit() -> dict:
    member = (
        "article_raw_data_archive_20260704/03_independent_generation_audit/"
        "sft_only_heldout_test/generation_samples.jsonl"
    )
    baseline_member = (
        "article_raw_data_archive_20260704/03_independent_generation_audit/"
        "sft_only_heldout_test/baseline_table.md"
    )
    structural_member = (
        "article_raw_data_archive_20260704/03_independent_generation_audit/"
        "sft_only_heldout_test/structural_diversity_summary.json"
    )
    with zipfile.ZipFile(RAW_ARCHIVE) as handle:
        raw_bytes = handle.read(member)
        rows = [json.loads(line) for line in raw_bytes.decode().splitlines() if line.strip()]
    selected = [row for row in rows if row.get("setting") == "sft_only_fourpattern_current"]
    formal = [row["api_result"] for row in selected
              if row.get("api_result", {}).get("formal_success_candidate") is True]
    result = {
        "raw_member": member,
        "raw_member_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "raw_setting_filter": "setting == sft_only_fourpattern_current",
        "formal_success_filter": "api_result.formal_success_candidate is True",
        "baseline_table_member": baseline_member,
        "baseline_table_header_line": 1,
        "baseline_table_row_line": 3,
        "baseline_table_row_label": "sft_only_fourpattern_current",
        "baseline_table_column_label": "CNN top-1 share",
        "structural_summary_member": structural_member,
        "structural_summary_json_path": "sft_only_fourpattern_current.cnn.top1_share",
        "attempted_n": len(selected),
        "formal_success_n": len(formal),
    }
    for key in ("cnn_signature", "family_hash", "actual_structure_signature", "graph_hash"):
        counts = Counter(str(row[key]) for row in formal if row.get(key) not in (None, ""))
        result[key] = {
            "covered_n": sum(counts.values()),
            "top1_count": max(counts.values()),
            "top1_share": max(counts.values()) / sum(counts.values()),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    search_roots = (ROOT, Path("/Users/zhangxi/Desktop/example-cvpr"))
    sampler_config = json.loads((SAMPLER / "run_config.json").read_text())
    sampler_summary = json.loads((SAMPLER / "summary_metrics.json").read_text())
    output = {
        "nn_gpt": {
            "remotes": git(NNGPT, "remote", "-v").splitlines(),
            "fixed_commit": COMMIT,
            "cat_file_type": git(NNGPT, "cat-file", "-t", COMMIT),
            "commit_metadata": git(NNGPT, "show", "-s", "--format=%H%n%P%n%aI%n%s", COMMIT).splitlines(),
            "branches_containing_commit": git(NNGPT, "branch", "-a", "--contains", COMMIT).splitlines(),
            "head": git(NNGPT, "rev-parse", "HEAD"),
            "fixed_commit_is_ancestor_of_head": subprocess.run(
                ["git", "-C", str(NNGPT), "merge-base", "--is-ancestor", COMMIT, "HEAD"]
            ).returncode == 0,
            "semantic_split_fix_is_ancestor_of_fixed_commit": subprocess.run(
                ["git", "-C", str(NNGPT), "merge-base", "--is-ancestor",
                 "6186ebe2c14e23e21444954b4d533abf0e603db4", COMMIT]
            ).returncode == 0,
            "smoke_prevalidation_fix_is_ancestor_of_fixed_commit": subprocess.run(
                ["git", "-C", str(NNGPT), "merge-base", "--is-ancestor",
                 "85c3b6ed10868a47fdbfd268fb0021c0c217a2d3", COMMIT]
            ).returncode == 0,
        },
        "nn_dataset": {
            "remotes": git(NNDATASET, "remote", "-v").splitlines(),
            "current_head_only": git(NNDATASET, "rev-parse", "HEAD"),
            "experiment_commit": None,
        },
        "four_pattern_sft_5epoch_concentration": sft_concentration(),
        "six_seed_derived_provenance_rows": six_seed_provenance_rows(),
        "six_seed_raw_zip_configs": six_seed_raw_zip_configs(),
        "independent_generation_audit_recompute": independent_generation_audit(),
        "search_scope": {
            "roots": [str(path) for path in search_roots],
            "zip_archives_opened": zip_inventory(search_roots),
            "zip_member_content_limit_bytes": 20_000_000,
            "zip_member_content_extensions": [".py", ".json", ".csv", ".md", ".tex", ".txt"],
            "jsonl_content_searched": False,
        },
        "threshold_artifact_hits": find_named_or_text(
            ROOT, ("threshold_sensitivity_70_80_90", "threshold sensitivity", "four of twelve")
        ),
        "threshold_manuscript_hits": find_named_or_text(
            Path("/Users/zhangxi/Desktop/example-cvpr"),
            ("threshold_sensitivity_70_80_90", "threshold sensitivity", "four of twelve"),
        ),
        "threshold_zip_member_hits": find_zip_members(
            search_roots,
            ("threshold_sensitivity_70_80_90", "threshold sensitivity", "four of twelve"),
        ),
        "thirty_point_thirty_file_hits": sorted(set(
            find_named_or_text(ROOT, ("30\\.30", "0\\.303", "independent_generation_audit"))
            + find_named_or_text(Path("/Users/zhangxi/Desktop/example-cvpr"),
                                 ("30\\.30", "0\\.303", "independent_generation_audit"))
        )),
        "thirty_point_thirty_zip_member_hits": find_zip_members(
            search_roots, ("30.30", "0.303", "independent_generation_audit")
        ),
        "imagenette_rule_sampler": {
            "generation_samples": str(SAMPLER / "generation_samples.jsonl"),
            "generation_samples_sha256": sha256(SAMPLER / "generation_samples.jsonl"),
            "config": sampler_config,
            "summary": sampler_summary,
            "raw_recompute": summarize_sampler_raw(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
