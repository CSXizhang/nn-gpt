#!/usr/bin/env python3
"""Rebuild reward evidence from a fixed nn-gpt commit and raw JSONL logs.

This script is read-only with respect to the repository, manuscript, and raw logs.
It obtains all implementation text through ``git show COMMIT:path``.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Iterator


COMMIT = "c91714dbe7dad1d02a9080243945bbf8e8ec9300"
SOURCE_PATHS = ("ab/gpt/TuneRL.py", "ab/gpt/TuneRLSft.py")
TARGETS = {
    "r_formal_success_signal": 0.28,
    "r_dense": 0.09515141,
    "r_repeat_family": -0.055,
    "r_plain_fuse_penalty": -0.11,
}
REQUIRED_CONSTANTS = {
    "FORMAL_SUCCESS_SIGNAL_BONUS", "TARGET_STRUCTURE_MATCH_BONUS",
    "STAGE23_REPEATED_BLOCK_REWARD_CAP", "STAGE2_DENSE_SCALE",
    "STAGE3_DENSE_SCALE", "STAGE2_PREV_GROUP_SCALE",
    "STAGE2_BEST_GROUP_SCALE", "STAGE2_BACKBONE_PREV_GROUP_SCALE",
    "STAGE2_BACKBONE_BEST_GROUP_SCALE", "REPEAT_FAMILY_PENALTY",
    "PLAIN_FUSE_PENALTY", "STAGE2_REPEAT_FAMILY_SCALE",
    "STAGE2_PLAIN_FUSE_SCALE",
}


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout


def source_at_commit(repo: Path, path: str) -> str:
    return git(repo, "show", f"{COMMIT}:{path}")


def constants(text: str) -> dict[str, Any]:
    tree = ast.parse(text)
    out: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            for target in targets:
                if isinstance(target, ast.Name):
                    try:
                        out[target.id] = ast.literal_eval(value)
                    except (ValueError, TypeError):
                        pass
    return out


def dicts(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from dicts(child)


def close(value: Any, target: float) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isclose(
        float(value), target, rel_tol=0.0, abs_tol=5e-8
    )


def scan_logs(search_roots: list[Path]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    seen: set[Path] = set()
    seen_hits: set[tuple[Any, ...]] = set()
    per_field_limit = 8
    for root in search_roots:
        for path in root.rglob("generation_samples.jsonl"):
            if all(sum(h["field"] == field for h in hits) >= per_field_limit for field in TARGETS):
                return hits
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            with path.open(encoding="utf-8", errors="strict") as handle:
                for line_no, line in enumerate(handle, 1):
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError(f"invalid JSON in {path}:{line_no}: {exc}") from exc
                    for obj in dicts(row):
                        for field, target in TARGETS.items():
                            if sum(h["field"] == field for h in hits) >= per_field_limit:
                                continue
                            if close(obj.get(field), target):
                                hit = {
                                    "field": field,
                                    "target": target,
                                    "path": str(path),
                                    "line": line_no,
                                    "candidate_id": row.get("candidate_id"),
                                    "setting": row.get("setting"),
                                    "stage": obj.get("current_stage_name"),
                                    "reward_target_value": obj.get("reward_target_value"),
                                    "frozen_train_acc": obj.get("frozen_train_acc", obj.get("train_acc")),
                                    "target_structure_match": obj.get("target_structure_match"),
                                    "formal_success_candidate": obj.get("formal_success_candidate"),
                                    "value": obj.get(field),
                                }
                                identity = (field, str(path), line_no, hit["candidate_id"], hit["stage"])
                                if identity not in seen_hits:
                                    seen_hits.add(identity)
                                    hits.append(hit)
    return hits


def scan_zip_logs(archives: list[Path], wanted_fields: set[str]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for archive in archives:
        if not archive.exists():
            continue
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                if not member.endswith("generation_samples.jsonl"):
                    continue
                with zf.open(member) as raw:
                    for line_no, raw_line in enumerate(raw, 1):
                        try:
                            row = json.loads(raw_line)
                        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                            raise RuntimeError(f"invalid JSON in {archive}!{member}:{line_no}: {exc}") from exc
                        for obj in dicts(row):
                            for field in list(wanted_fields):
                                if close(obj.get(field), TARGETS[field]):
                                    hits.append({
                                        "field": field, "target": TARGETS[field],
                                        "path": f"{archive}!{member}", "line": line_no,
                                        "candidate_id": row.get("candidate_id"), "setting": row.get("setting"),
                                        "stage": obj.get("current_stage_name"),
                                        "reward_target_value": obj.get("reward_target_value"),
                                        "frozen_train_acc": obj.get("frozen_train_acc", obj.get("train_acc")),
                                        "target_structure_match": obj.get("target_structure_match"),
                                        "formal_success_candidate": obj.get("formal_success_candidate"),
                                        "value": obj.get(field),
                                    })
                                    wanted_fields.remove(field)
                                    break
                            if not wanted_fields:
                                return hits
    return hits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/Users/zhangxi/code/RL/nn-gpt"))
    parser.add_argument("--workspace", type=Path, default=Path("/Users/zhangxi/code/RL"))
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "reward_audit_evidence.json")
    args = parser.parse_args()

    commit_type = git(args.repo, "cat-file", "-t", COMMIT).strip()
    if commit_type != "commit":
        raise RuntimeError(f"{COMMIT} resolved as {commit_type!r}, expected 'commit'")
    sources = {path: source_at_commit(args.repo, path) for path in SOURCE_PATHS}
    tune_constants = constants(sources["ab/gpt/TuneRL.py"])
    missing_constants = sorted(REQUIRED_CONSTANTS - set(tune_constants))
    if missing_constants:
        raise RuntimeError(f"required fixed-commit constants not parsed: {missing_constants}")
    hits = scan_logs([
        args.workspace / "nn-gpt" / "experiment_inputs",
        args.workspace / "delivery_20260727",
        args.workspace / "faraz_delivery_20260726",
        args.workspace / "faraz_followup_delivery_20260808",
        args.workspace / "reply_20260802",
    ])
    # Also take one direct hit per requested field from the original article
    # archive.  This avoids relying only on later copied delivery trees.
    hits.extend(scan_zip_logs([
            args.workspace / "faraz_delivery_20260726" / "article_raw_data_archive_20260718_resend_20260726.zip",
            args.workspace / "data_20260726.zip",
            args.workspace / "delivery_20260727.zip",
            args.workspace / "followup_delivery_20260808.zip",
        ], set(TARGETS)))

    dense_hits = [h for h in hits if h["field"] == "r_dense"]
    for hit in dense_hits:
        if hit["reward_target_value"] is None or hit["frozen_train_acc"] is None:
            hit["formula_recomputed"] = None
            continue
        inner = min(0.35, max(0.02, 0.03 + 0.28 * float(hit["reward_target_value"]) + 0.04 * max(0.0, float(hit["frozen_train_acc"]) - 0.50)))
        scale = 0.50 if hit["stage"] == "stage2_formal_explore" else 0.70
        hit["formula_recomputed"] = scale * inner
        hit["formula_matches"] = close(hit["formula_recomputed"], float(hit["value"]))

    evidence = {
        "commit": COMMIT,
        "commit_type": commit_type,
        "source_access": [f"git show {COMMIT}:{path}" for path in SOURCE_PATHS],
        "constants": tune_constants,
        "raw_log_hits": hits,
    }
    provenance_archive = args.workspace / "faraz_delivery_20260726" / "article_raw_data_archive_20260718_resend_20260726.zip"
    provenance_member = "article_raw_data_archive_20260704/01_six_seed_robustness/1pattern_seed42/run_config.json"
    with zipfile.ZipFile(provenance_archive) as zf:
        seed42_config = json.loads(zf.read(provenance_member))
    evidence["seed42_run_provenance"] = {
        "path": f"{provenance_archive}!{provenance_member}",
        "git": seed42_config["git"],
        "reward": seed42_config["reward"],
    }
    manuscript = Path("/Users/zhangxi/Desktop/example-cvpr/Paper_draft_en.tex")
    evidence["manuscript"] = {
        "path": str(manuscript),
        "sha256": hashlib.sha256(manuscript.read_bytes()).hexdigest(),
        "mtime_ns": manuscript.stat().st_mtime_ns,
        "table_lines": [151, 173],
        "scope_note": "Locally available numbered table; not independently established as the submission-final source.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(evidence, indent=2, sort_keys=True, default=lambda value: sorted(value) if isinstance(value, set) else repr(value)) + "\n")
    print(json.dumps({
        "commit_type": commit_type,
        "output": str(args.out),
        "hit_counts": {field: sum(h["field"] == field for h in hits) for field in TARGETS},
    }, indent=2))


if __name__ == "__main__":
    main()
