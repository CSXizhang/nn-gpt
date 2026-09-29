#!/usr/bin/env python3
"""Recompute allocated GPU-hours from the retained Slurm records."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "08_sacct_records.csv"


def main() -> None:
    with SOURCE.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    totals = defaultdict(float)
    for row in rows:
        expected = int(row["elapsed_seconds"]) * int(row["gpu_count"]) / 3600
        recorded = float(row["gpu_hours"])
        if abs(expected - recorded) > 5e-7:
            raise ValueError(f"GPU-hour mismatch for {row['job_id']}: {expected} != {recorded}")
        totals[(row["cohort"], row["condition"])] += expected

    for (cohort, condition), gpu_hours in sorted(totals.items()):
        print(f"{cohort},{condition},{gpu_hours:.2f}")


if __name__ == "__main__":
    main()
