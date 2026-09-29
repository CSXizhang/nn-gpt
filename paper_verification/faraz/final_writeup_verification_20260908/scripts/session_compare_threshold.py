#!/usr/bin/env python3
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = ROOT / "session_evidence" / "historical_threshold_output.csv"
RECOMPUTED = ROOT / "session_evidence" / "threshold_sensitivity_recomputed.csv"
KEY = ("condition", "seed", "role")
FIELDS = ("rows",) + tuple(
    f"{metric}_top1_ge_{threshold}_onset"
    for metric in ("family", "descriptor", "graph")
    for threshold in (70, 80, 90)
)


def load(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12, (path, len(rows))
    mapped = {tuple(row[field] for field in KEY): row for row in rows}
    assert len(mapped) == 12, (path, "duplicate keys")
    return mapped


historical = load(HISTORICAL)
recomputed = load(RECOMPUTED)
assert historical.keys() == recomputed.keys()
for key in historical:
    for field in FIELDS:
        assert historical[key][field] == recomputed[key][field], (
            key, field, historical[key][field], recomputed[key][field]
        )
print(f"PASS: 12 rows x {len(FIELDS)} checked fields match")
