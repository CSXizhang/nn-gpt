#!/usr/bin/env python3
"""Recompute the historical 70/80/90 threshold table from the delivered ZIP.

This preserves the corrected 2026-07-04 session logic while redirecting all
I/O to the audit directory and adding field-coverage accounting.
"""
import csv
import json
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = ROOT.parents[0] / "faraz_delivery_20260726" / "article_raw_data_archive_20260718_resend_20260726.zip"
PREFIX = "article_raw_data_archive_20260704/01_six_seed_robustness"
THRESHOLDS = (0.70, 0.80, 0.90)
METRICS = (("family", "family_hash"), ("descriptor", "descriptor_key"), ("graph", "graph_hash"))
RUNS = tuple(f"{condition}pattern_seed{seed}" for condition in (1, 4) for seed in (42, 123, 777, 114, 514, 919))


def api(row):
    return row.get("api_result") if isinstance(row.get("api_result"), dict) else {}


def key_for(item, field):
    value = item.get(field)
    source = field if value else ""
    if field == "graph_hash" and not value:
        if item.get("signature"):
            value, source = item["signature"], "signature"
        elif item.get("actual_structure_signature"):
            value, source = item["actual_structure_signature"], "actual_structure_signature"
    if field == "descriptor_key" and not value and item.get("actual_structure_signature"):
        value, source = item["actual_structure_signature"], "actual_structure_signature"
    if field == "family_hash" and not value:
        if item.get("family_id"):
            value, source = item["family_id"], "family_id"
        elif item.get("family_expr"):
            value, source = item["family_expr"], "family_expr"
    return (str(value), source) if value is not None else ("", "")


def main():
    records = []
    coverage = []
    with zipfile.ZipFile(ZIP_PATH) as archive:
        for folder in RUNS:
            member = f"{PREFIX}/{folder}/generation_samples.jsonl"
            rows = [json.loads(line) for line in archive.read(member).decode().splitlines() if line.strip()]
            condition = "1-pattern" if folder.startswith("1") else "4-pattern"
            seed = folder.split("seed", 1)[1]
            role = "original" if seed in {"42", "123", "777"} else "added"
            rec = {"condition": condition, "seed": seed, "role": role, "run_id": folder, "rows": len(rows)}
            sources = {name: Counter() for name, _ in METRICS}
            denominators = {name: [] for name, _ in METRICS}
            formal_counts = []
            onsets = {(name, threshold): "never" for name, _ in METRICS for threshold in THRESHOLDS}
            for start in range(0, len(rows), 100):
                chunk = rows[start:start + 100]
                successful = [api(row) for row in chunk if api(row).get("formal_success_candidate")]
                formal_counts.append(len(successful))
                for name, field in METRICS:
                    values = []
                    for item in successful:
                        value, source = key_for(item, field)
                        if value:
                            values.append(value)
                            sources[name][source] += 1
                    denominators[name].append(len(values))
                    if len(values) < 20:
                        continue
                    share = Counter(values).most_common(1)[0][1] / len(values)
                    for threshold in THRESHOLDS:
                        if onsets[(name, threshold)] == "never" and share >= threshold:
                            onsets[(name, threshold)] = str(start + len(chunk))
            for name, _ in METRICS:
                for threshold in THRESHOLDS:
                    rec[f"{name}_top1_ge_{int(threshold * 100)}_onset"] = onsets[(name, threshold)]
            records.append(rec)
            coverage.append({
                "condition": condition,
                "seed": int(seed),
                "rows": len(rows),
                "formal_success_by_window": formal_counts,
                "metric_valid_denominator_by_window": denominators,
                "field_source_counts": {name: dict(counts) for name, counts in sources.items()},
            })

    output = ROOT / "session_evidence" / "threshold_sensitivity_recomputed.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["condition", "seed", "role", "run_id", "rows"] + [
        f"{name}_top1_ge_{int(threshold * 100)}_onset"
        for name, _ in METRICS for threshold in THRESHOLDS
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    (ROOT / "session_evidence" / "threshold_field_coverage.json").write_text(
        json.dumps({"input_zip": str(ZIP_PATH), "runs": coverage}, indent=2) + "\n", encoding="utf-8"
    )
    print(f"runs={len(records)} rows={sum(row['rows'] for row in records)}")
    print(output)


if __name__ == "__main__":
    main()
