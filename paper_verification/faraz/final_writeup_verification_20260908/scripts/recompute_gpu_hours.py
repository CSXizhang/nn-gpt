#!/usr/bin/env python3
"""Recompute allocated GPU-hours from retained raw Slurm-accounting snapshots."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_HALF_UP, getcontext
from fractions import Fraction
from pathlib import Path

getcontext().prec = 40


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"job_id", "gpu_count", "elapsed_seconds", "gpu_type"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"missing required Slurm fields in {path}")
    return rows


def hours(row: dict[str, str]) -> Fraction:
    return Fraction(int(row["elapsed_seconds"]) * int(row["gpu_count"]), 3600)


def total(rows: list[dict[str, str]]) -> Fraction:
    return sum((hours(row) for row in rows), Fraction())


def decimal_text(value: Fraction, places: int = 15) -> str:
    decimal = Decimal(value.numerator) / Decimal(value.denominator)
    return format(decimal, f".{places}f")


def rounded_2(value: Fraction) -> str:
    decimal = Decimal(value.numerator) / Decimal(value.denominator)
    return format(decimal.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP), ".2f")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def serialise(value: Fraction) -> dict[str, object]:
    gpu_seconds = value * 3600
    if gpu_seconds.denominator != 1:
        raise ValueError("GPU-seconds must be integral")
    return {
        "gpu_seconds": gpu_seconds.numerator,
        "exact_fraction_gpu_hours": f"{value.numerator}/{value.denominator}",
        "decimal_gpu_hours_15dp": decimal_text(value),
        "rounded_gpu_hours_2dp": rounded_2(value),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-sacct", type=Path, required=True)
    parser.add_argument("--extended-sacct", type=Path, required=True)
    parser.add_argument("--wide-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    primary = read_csv(args.primary_sacct)
    extended = read_csv(args.extended_sacct)
    with args.wide_manifest.open(encoding="utf-8") as handle:
        wide_manifest = [json.loads(line) for line in handle if line.strip()]

    for source_name, rows in (("primary", primary), ("extended", extended)):
        duplicate_ids = sorted(job_id for job_id, count in Counter(row["job_id"] for row in rows).items() if count != 1)
        if duplicate_ids:
            raise ValueError(f"duplicate job_id values within {source_name} snapshot: {duplicate_ids}")

    # Verify the retained decimal field but always calculate from integer allocation data.
    recorded_mismatches = []
    for row in extended:
        if "gpu_hours" in row and row["gpu_hours"]:
            expected = Decimal(hours(row).numerator) / Decimal(hours(row).denominator)
            if abs(expected - Decimal(row["gpu_hours"])) > Decimal("0.0000005"):
                recorded_mismatches.append(row["job_id"])
    if recorded_mismatches:
        raise ValueError(f"recorded gpu_hours mismatch: {recorded_mismatches}")

    primary_by_condition = defaultdict(list)
    primary_by_hardware = defaultdict(list)
    for row in primary:
        primary_by_condition[row["condition"]].append(row)
        primary_by_hardware[row["gpu_type"]].append(row)

    ext_by_cohort = defaultdict(list)
    ext_by_condition = defaultdict(list)
    for row in extended:
        ext_by_cohort[row["cohort"]].append(row)
        ext_by_condition[(row["cohort"], row["condition"])].append(row)

    ablation = ext_by_cohort["reward_ablation"]
    ablation_overlap = [row for row in ablation if row["overlaps_six_seed_cifar10"] == "yes"]
    ablation_unique = [row for row in ablation if row["overlaps_six_seed_cifar10"] == "no"]
    primary_ids = {row["job_id"] for row in primary}
    extended_ids = {row["job_id"] for row in extended}
    overlap_ids = {row["job_id"] for row in ablation_overlap}
    if not overlap_ids.issubset(primary_ids):
        raise ValueError(f"ablation overlap IDs absent from primary snapshot: {overlap_ids-primary_ids}")
    if primary_ids & extended_ids != overlap_ids:
        raise ValueError("actual cross-snapshot duplicate IDs differ from declared overlap IDs")
    primary_by_id = {row["job_id"]: row for row in primary}
    extended_by_id = {row["job_id"]: row for row in extended}
    consistency_fields = ("elapsed_seconds", "gpu_count", "gpu_type")
    for job_id in overlap_ids:
        mismatched = [field for field in consistency_fields if primary_by_id[job_id][field] != extended_by_id[job_id][field]]
        if mismatched:
            raise ValueError(f"cross-snapshot field mismatch for {job_id}: {mismatched}")

    components = {
        "primary_six_seed_cifar10": total(primary),
        "reward_ablation_incremental_unique": total(ablation_unique),
        "qwen_cifar10": total(ext_by_cohort["qwen_cifar10"]),
        "cifar100_one_pattern": total(ext_by_cohort["cifar100_1pattern"]),
        "proxy_top20": total(ext_by_condition[("proxy_retraining", "top20_unfrozen20")]),
        "proxy_wide38": total(ext_by_condition[("proxy_retraining", "wide38_unfrozen20")]),
    }
    grand = sum(components.values(), Fraction())
    rounded_component_sum = sum(Decimal(rounded_2(value)) for value in components.values())
    wide_rows = ext_by_condition[("proxy_retraining", "wide38_unfrozen20")]
    wide_task_ids = sorted(int(row["job_id"].rsplit("_", 1)[1]) for row in wide_rows)
    if any("selection_index" not in row for row in wide_manifest):
        raise ValueError("wide manifest row lacks selection_index")
    expected_wide_task_ids = sorted(int(row["selection_index"]) for row in wide_manifest)
    if len(expected_wide_task_ids) != len(set(expected_wide_task_ids)):
        raise ValueError("duplicate selection_index in wide manifest")

    result = {
        "definition": "allocated GPU-hours = elapsed_seconds * allocated_gpu_count / 3600",
        "not_measured": "actual GPU utilisation hours",
        "inputs": {
            "primary_sacct": {"path": str(args.primary_sacct.resolve()), "sha256": sha256(args.primary_sacct), "rows": len(primary)},
            "extended_sacct": {"path": str(args.extended_sacct.resolve()), "sha256": sha256(args.extended_sacct), "rows": len(extended)},
            "wide_manifest": {
                "path": str(args.wide_manifest.resolve()),
                "sha256": sha256(args.wide_manifest),
                "rows": len(wide_manifest),
                "task_id_field": "selection_index",
            },
        },
        "primary": {
            "one_pattern": serialise(total(primary_by_condition["1pattern"])),
            "four_pattern": serialise(total(primary_by_condition["4pattern"])),
            "total": serialise(total(primary)),
            "H100": serialise(total(primary_by_hardware["H100"])),
            "L40S": serialise(total(primary_by_hardware["L40S"])),
            "job_ids": [row["job_id"] for row in primary],
        },
        "extended": {
            "reward_ablation_all_nine_rows": serialise(total(ablation)),
            "reward_ablation_overlap_with_primary": serialise(total(ablation_overlap)),
            "reward_ablation_incremental_unique": serialise(total(ablation_unique)),
            "reward_ablation_overlap_job_ids": sorted(overlap_ids),
            "reward_ablation_incremental_job_ids": [row["job_id"] for row in ablation_unique],
            "qwen_cifar10": serialise(components["qwen_cifar10"]),
            "qwen_job_ids": [row["job_id"] for row in ext_by_cohort["qwen_cifar10"]],
            "cifar100_one_pattern": serialise(components["cifar100_one_pattern"]),
            "cifar100_job_ids": [row["job_id"] for row in ext_by_cohort["cifar100_1pattern"]],
            "proxy_top20": serialise(components["proxy_top20"]),
            "proxy_top20_job_ids": [row["job_id"] for row in ext_by_condition[("proxy_retraining", "top20_unfrozen20")]],
            "proxy_wide38": serialise(components["proxy_wide38"]),
            "proxy_wide38_job_ids": [row["job_id"] for row in wide_rows],
            "proxy_wide38_retained_sacct_rows": len(wide_rows),
            "proxy_wide38_expected_task_ids_from_manifest": expected_wide_task_ids,
            "proxy_wide38_retained_task_ids": wide_task_ids,
            "proxy_wide38_missing_task_ids": sorted(set(expected_wide_task_ids) - set(wide_task_ids)),
            "proxy_wide38_completeness": "NOT VERIFIED: manifest defines 38 tasks but retained sacct snapshot has 37 rows and lacks task 0",
        },
        "unique_grand_total": {
            "scope": "deduplicated sum of supplied retained accounting rows; not a verified complete extended total because wide-manifest task 0 lacks a retained sacct row",
            **serialise(grand),
        },
        "rounding_check": {
            "sum_of_components_after_each_is_rounded_to_2dp": format(rounded_component_sum, ".2f"),
            "round_exact_grand_total_to_2dp": rounded_2(grand),
            "difference_gpu_hours": format(rounded_component_sum - Decimal(rounded_2(grand)), ".2f"),
        },
        "distinct_accounting_records": {
            "primary_rows": len(primary),
            "extended_rows": len(extended),
            "overlap_rows": len(ablation_overlap),
            "unique_rows_after_overlap_deduplication": len(primary) + len(extended) - len(ablation_overlap),
        },
        "per_job": [
            {
                "source": "primary" if source == 0 else "extended",
                "job_id": row["job_id"],
                "cohort": row.get("cohort", "six_seed_cifar10_rl"),
                "condition": row["condition"],
                "seed": row.get("seed", ""),
                "gpu_type": row["gpu_type"],
                "gpu_count": int(row["gpu_count"]),
                "elapsed_seconds": int(row["elapsed_seconds"]),
                **serialise(hours(row)),
            }
            for source, rows in enumerate((primary, extended))
            for row in rows
        ],
    }
    output = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
