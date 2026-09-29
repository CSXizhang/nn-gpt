#!/usr/bin/env python3
"""Recompute SFT-only and matched post-RL structural metrics from raw JSONL.

This script is offline and read-only with respect to experiment inputs. Post-RL
inputs are streamed directly from the delivered ZIP archive.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any


WORKSPACE = Path("/Users/zhangxi/code/RL")
DELIVERY = WORKSPACE / "faraz_followup_delivery_20260808" / "sft_only" / "raw"
POST_ZIP = WORKSPACE / "faraz_delivery_20260726" / "article_raw_data_archive_20260718_resend_20260726.zip"
POST_PREFIX = "article_raw_data_archive_20260704/01_six_seed_robustness"
OUT = WORKSPACE / "final_writeup_verification_20260908"
FIELDS = (
    "family_hash",
    "actual_structure_signature",
    "backbone_signature",
    "block_signature",
    "cnn_signature",
    "backbone_cnn_signature",
    "graph_hash",
)


def load_lines(lines: Any, source: str) -> list[dict[str, Any]]:
    rows = []
    for line_no, raw in enumerate(lines, 1):
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        if raw.strip():
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise RuntimeError(f"{source}:{line_no}: JSON value is not an object")
            rows.append(value)
    return rows


def load_file(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return load_lines(handle, str(path))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def api(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("api_result")
    return value if isinstance(value, dict) else {}


def formal(row: dict[str, Any]) -> bool:
    # Exact packaged recomputation rule: explicit boolean gate, not accuracy presence.
    return api(row).get("formal_success_candidate") is True


def metric(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = []
    for row in rows:
        result = api(row)
        value = result.get(field)
        if field == "backbone_cnn_signature" and value in (None, ""):
            backbone, cnn = result.get("backbone_signature"), result.get("cnn_signature")
            value = f"{backbone}::{cnn}" if backbone not in (None, "") and cnn not in (None, "") else None
        if value not in (None, ""):
            values.append(str(value))
    counts = Counter(values)
    n = len(values)
    if not n:
        return {"covered_n": 0, "unique_count": 0, "top1_value": None,
                "top1_count": 0, "top1_share": None, "shannon_entropy": None,
                "effective_number": None}
    top_value, top_count = counts.most_common(1)[0]
    probabilities = [count / n for count in counts.values()]
    entropy = -sum(p * math.log(p) for p in probabilities)
    return {"covered_n": n, "unique_count": len(counts), "top1_value": top_value,
            "top1_count": top_count, "top1_share": top_count / n,
            "shannon_entropy": entropy, "effective_number": math.exp(entropy)}


def indices(rows: list[dict[str, Any]]) -> list[int | None]:
    output = []
    for row in rows:
        candidate_id = str(row.get("candidate_id", ""))
        try:
            output.append(int(candidate_id.rsplit("-", 1)[1]))
        except (IndexError, ValueError):
            output.append(None)
    return output


def summarize(condition: str, seed: int, stage: str, rows: list[dict[str, Any]],
              source: str, checksum: str, config: dict[str, Any]) -> dict[str, Any]:
    all_indices = indices(rows)
    selected = rows if stage == "SFT-only" else rows[-100:]
    selected_indices = indices(selected)
    formal_rows = [row for row in selected if formal(row)]
    ids = [row.get("candidate_id") for row in rows]
    present_indices = [x for x in all_indices if x is not None]
    row_fingerprints = [hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
                        for row in rows]
    return {
        "condition": condition,
        "seed": seed,
        "stage": stage,
        "source": source,
        "source_sha256": checksum,
        "raw_rows": len(rows),
        "candidate_ids_present": sum(value is not None for value in ids),
        "candidate_ids_unique": len({value for value in ids if value is not None}),
        "row_fingerprints_unique": len(set(row_fingerprints)),
        "candidate_index_min": min(present_indices) if present_indices else None,
        "candidate_index_max": max(present_indices) if present_indices else None,
        "candidate_indices_monotonic": (all_indices == sorted(all_indices)) if present_indices else None,
        "attempted_window": "all 400 rows in file" if stage == "SFT-only" else "file-order rows 701-800 (last 100 of 800)",
        "window_n": len(selected),
        "window_index_min": min((x for x in selected_indices if x is not None), default=None),
        "window_index_max": max((x for x in selected_indices if x is not None), default=None),
        "formal_success_n": len(formal_rows),
        "formal_success_rate": len(formal_rows) / len(selected),
        "formal_rule": "api_result.formal_success_candidate is exactly true",
        "metrics_population": "formal-success rows within attempted window",
        "metrics": {field: metric(formal_rows, field) for field in FIELDS},
        "metric_notes": {"backbone_cnn_signature": "derived as backbone_signature::cnn_signature when no stored combined field exists"},
        "config": config,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    consistency_checks = []
    for condition in ("1pattern", "4pattern"):
        for seed in (42, 123, 777):
            directory = DELIVERY / f"{condition}_seed{seed}"
            source = directory / "generation_samples.jsonl"
            config = {
                "generation": json.loads((directory / "generation_run_config.json").read_text()),
                "evaluation": json.loads((directory / "evaluation_run_config.json").read_text()),
            }
            rows = load_file(source)
            if len(rows) != 400:
                raise RuntimeError(f"{source}: expected 400 rows, found {len(rows)}")
            result = summarize(condition, seed, "SFT-only", rows, str(source),
                               sha256_file(source), config)
            results.append(result)
            for field in FIELDS:
                if result["metrics"][field]["covered_n"] != result["formal_success_n"]:
                    raise RuntimeError(f"{source}: {field} coverage differs from formal-success N")
            archived = next(iter(json.loads((directory / "structural_diversity_summary.json").read_text()).values()))
            mappings = {
                "backbone": "backbone_signature",
                "cnn": "cnn_signature",
                "backbone_cnn": "backbone_cnn_signature",
                "forward_graph": "graph_hash",
            }
            for archived_name, recomputed_name in mappings.items():
                expected = archived[archived_name]
                actual = result["metrics"][recomputed_name]
                consistency_checks.append({
                    "condition": condition, "seed": seed,
                    "archived_metric": archived_name, "recomputed_field": recomputed_name,
                    "unique_match": expected["unique"] == actual["unique_count"],
                    "top1_count_match": expected["top1_count"] == actual["top1_count"],
                    "effective_abs_difference": abs(expected["effective_number"] - actual["effective_number"]),
                })
                if not (consistency_checks[-1]["unique_match"] and
                        consistency_checks[-1]["top1_count_match"] and
                        consistency_checks[-1]["effective_abs_difference"] <= 1e-12):
                    raise RuntimeError(f"{source}: archived {archived_name} summary mismatch")

    zip_checksum = sha256_file(POST_ZIP)
    with zipfile.ZipFile(POST_ZIP) as archive:
        for condition in ("1pattern", "4pattern"):
            for seed in (42, 123, 777):
                base = f"{POST_PREFIX}/{condition}_seed{seed}"
                member = f"{base}/generation_samples.jsonl"
                config_member = f"{base}/run_config.json"
                with archive.open(member) as handle:
                    rows = load_lines(handle, f"{POST_ZIP}!{member}")
                if len(rows) != 800:
                    raise RuntimeError(f"{member}: expected 800 rows, found {len(rows)}")
                with archive.open(config_member) as handle:
                    config = json.load(handle)
                member_hash = hashlib.sha256(archive.read(member)).hexdigest()
                result = summarize(condition, seed, "post-RL", rows,
                                   f"{POST_ZIP}!{member}", member_hash,
                                   {"run": config, "container_zip_sha256": zip_checksum})
                results.append(result)
                for field in FIELDS:
                    if result["metrics"][field]["covered_n"] != result["formal_success_n"]:
                        raise RuntimeError(f"{member}: {field} coverage differs from formal-success N")

    sft_last100_sensitivity = []
    for condition in ("1pattern", "4pattern"):
        for seed in (42, 123, 777):
            source = DELIVERY / f"{condition}_seed{seed}" / "generation_samples.jsonl"
            kept = [row for row in load_file(source)[-100:] if formal(row)]
            sft_last100_sensitivity.append({
                "condition": condition, "seed": seed,
                "attempted_window": "SFT-only last 100 file rows",
                "formal_success_n": len(kept),
                "family_hash": metric(kept, "family_hash"),
                "actual_structure_signature": metric(kept, "actual_structure_signature"),
            })

    output = {
        "method": {
            "formal_success": "api_result.formal_success_candidate is True",
            "entropy": "H = -sum_i p_i ln(p_i), natural logarithm",
            "effective_number": "exp(H)",
            "post_rl_window": "last 100 file-order rows (positions 701..800); archived RL rows have no candidate_id, so row fingerprints rather than candidate IDs were checked for duplicates",
        },
        "runs": results,
        "archived_summary_consistency_checks": consistency_checks,
        "secondary_sft_last100_window_sensitivity": sft_last100_sensitivity,
    }
    (OUT / "sft_family_recompute.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fieldnames = [
        "Condition", "Seed", "Stage", "Attempted window", "Formal-success N",
        "Family top-1", "Family eff.", "Family+Block top-1", "Family+Block eff.",
    ]
    with (OUT / "sft_family_pre_post.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            fam = result["metrics"]["family_hash"]
            family_block = result["metrics"]["actual_structure_signature"]
            writer.writerow({
                "Condition": result["condition"], "Seed": result["seed"],
                "Stage": result["stage"], "Attempted window": result["attempted_window"],
                "Formal-success N": result["formal_success_n"],
                "Family top-1": fam["top1_share"], "Family eff.": fam["effective_number"],
                "Family+Block top-1": family_block["top1_share"],
                "Family+Block eff.": family_block["effective_number"],
            })


if __name__ == "__main__":
    main()
