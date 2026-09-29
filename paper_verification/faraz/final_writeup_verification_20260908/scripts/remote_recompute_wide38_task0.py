#!/usr/bin/env python3
import csv
import json
import re
from decimal import Decimal, getcontext
from fractions import Fraction
from pathlib import Path

from recompute_gpu_hours import read_csv, total

getcontext().prec = 28
ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "remote_evidence" / "wide38"


def main() -> None:
    rows = list(csv.DictReader(
        (EVIDENCE / "sacct_2968450_2968490_2968494_raw.txt").open(),
        delimiter="|",
    ))
    rows = [{k: v for k, v in row.items() if k} for row in rows]
    task0 = [row for row in rows if row["JobID"] == "2968490_0"]
    assert len(task0) == 1
    task0 = task0[0]
    assert task0["State"] == "COMPLETED"
    gpu_match = re.search(r"(?:^|,)gres/gpu=(\d+)(?:,|$)", task0["AllocTRES"])
    assert gpu_match
    task0_gpu_count = int(gpu_match.group(1))
    assert task0["ElapsedRaw"] == "1920"

    manifests = [json.loads(line) for line in (EVIDENCE / "proxy_manifest_wide38.jsonl").open()]
    manifest0 = [row for row in manifests if row["selection_index"] == 0]
    assert len(manifest0) == 1
    result0 = json.loads((EVIDENCE / "00_4pattern_seed42_743.json").read_text())
    assert result0["task_id"] == 0
    assert result0["candidate"]["candidate_id"] == manifest0[0]["candidate_id"]
    log = (EVIDENCE / "proxy-wide38-smoke-2968490_0.out").read_text()
    assert "wrote " in log and "results/00_4pattern_seed42_743.json" in log

    primary = read_csv(ROOT.parent / "delivery_20260727" / "06_gpu_hours" / "sacct_six_seed_cifar10_rl_20260728.csv")
    extended = read_csv(ROOT.parent / "reply_20260802" / "08_sacct_records.csv")
    retained_wide = [row for row in extended if row["cohort"] == "proxy_retraining" and row["condition"] == "wide38_unfrozen20"]
    successful_live = {row["JobID"]: row for row in rows if row["JobID"].startswith("2968494_") and row["State"] == "COMPLETED"}
    assert {row["job_id"] for row in retained_wide} == set(successful_live)
    for row in retained_wide:
        live = successful_live[row["job_id"]]
        live_gpu = re.search(r"(?:^|,)gres/gpu=(\d+)(?:,|$)", live["AllocTRES"])
        assert live_gpu
        assert int(row["elapsed_seconds"]) == int(live["ElapsedRaw"])
        assert int(row["gpu_count"]) == int(live_gpu.group(1))

    original_wide = total(retained_wide)
    overlap = {row["job_id"] for row in extended if row.get("overlaps_six_seed_cifar10") == "yes"}
    original_grand = total(primary) + total([row for row in extended if row["job_id"] not in overlap])
    task0_seconds = int(task0["ElapsedRaw"])
    task0_hours = Fraction(task0_seconds * task0_gpu_count, 3600)
    corrected_wide = original_wide + task0_hours
    corrected_grand = original_grand + task0_hours
    result = {
        "definition": "allocated GPU-hours = elapsed wall time * allocated GPU count / 3600",
        "task0": {
            "job_id": task0["JobID"],
            "job_id_raw": task0["JobIDRaw"],
            "state": task0["State"],
            "elapsed_seconds": task0_seconds,
            "allocated_gpu_count": task0_gpu_count,
            "gpu_type": "L40S",
            "gpu_hours": str(Decimal(task0_hours.numerator) / Decimal(task0_hours.denominator)),
            "candidate_id": result0["candidate"]["candidate_id"],
            "result_file": "00_4pattern_seed42_743.json",
        },
        "wide38": {
            "previous_37_task_gpu_seconds": int(original_wide * 3600),
            "corrected_38_task_gpu_seconds": int(corrected_wide * 3600),
            "corrected_gpu_hours": str(Decimal(corrected_wide.numerator) / Decimal(corrected_wide.denominator)),
        },
        "extended_unique_total": {
            "previous_supplied_row_gpu_seconds": int(original_grand * 3600),
            "corrected_gpu_seconds": int(corrected_grand * 3600),
            "corrected_gpu_hours": str(Decimal(corrected_grand.numerator) / Decimal(corrected_grand.denominator)),
            "rounded_2dp": str((Decimal(corrected_grand.numerator) / Decimal(corrected_grand.denominator)).quantize(Decimal("0.01"))),
        },
        "scope_note": "Excludes failed 2968450 array allocations, consistent with the retained successful-result accounting scope.",
    }
    output = EVIDENCE / "gpu_task0_recomputed.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
