#!/usr/bin/env python3
"""Recompute the bounded FINAL_RESULTS accuracy/count checks from raw JSONL."""

import json
import statistics
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def accuracy(api):
    # Exact precedence of delivery recompute_final_summaries.py:40-48.
    value = api.get("test_acc")
    if not isinstance(value, (int, float)):
        value = api.get("formal_horizon_test_acc")
    if isinstance(value, dict):
        value = value.get("5", value.get(5))
    return value if isinstance(value, (int, float)) else None


def summarize(path: Path):
    rows = load(path)
    formal = [r["api_result"] for r in rows if r.get("api_result", {}).get("formal_success_candidate") is True]
    values = [accuracy(a) for a in formal]
    assert all(v is not None for v in values)
    test_values = [a.get("test_acc") for a in formal]
    horizon_values = [(a.get("formal_horizon_test_acc") or {}).get("5") for a in formal]
    test_n = sum(isinstance(v, (int, float)) for v in test_values)
    horizon5_n = sum(isinstance(v, (int, float)) for v in horizon_values)
    equal_n = sum(isinstance(t, (int, float)) and isinstance(h, (int, float)) and t == h for t, h in zip(test_values, horizon_values))
    # These packages always use test_acc, exactly as the delivery function does.
    # We record horizon availability/equality because horizon['5'] is not a
    # valid silent substitute: a small number of rows differ.
    assert test_n == len(formal)
    return {
        "input": str(path),
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "attempted_n": len(rows),
        "formal_success_n": len(formal),
        "accuracy_function": "delivery recompute_final_summaries.py:40-48: api_result.test_acc first; formal_horizon_test_acc['5'] only if test_acc non-numeric",
        "test_acc_numeric_n": test_n,
        "formal_horizon_test_acc_5_numeric_n": horizon5_n,
        "test_acc_equals_horizon5_n": equal_n,
        "fallback_used_n": len(formal) - test_n,
        "accuracy_n": len(values),
        "mean_accuracy": sum(values) / len(values),
        "accuracy_values": values,
    }


def main():
    out = {"formula": "seed mean = sum(formal-success 5-epoch accuracies)/N; seed-mean condition accuracy = arithmetic mean of three seed means; pooled = arithmetic mean over all formal-success accuracies"}
    seed_rows = {}
    for condition in ("1pattern", "4pattern"):
        items = []
        for seed in (42, 123, 777):
            path = ROOT / "faraz_followup_delivery_20260808" / "sft_only" / "raw" / f"{condition}_seed{seed}" / "generation_samples.jsonl"
            item = summarize(path)
            item["seed"] = seed
            items.append(item)
        values = [v for item in items for v in item.pop("accuracy_values")]
        out[condition] = {
            "seeds": items,
            "attempted_n": sum(i["attempted_n"] for i in items),
            "formal_success_n": sum(i["formal_success_n"] for i in items),
            "seed_mean_accuracy": sum(i["mean_accuracy"] for i in items) / 3,
            "seed_sd_accuracy_sample": statistics.stdev(i["mean_accuracy"] for i in items),
            "pooled_accuracy": sum(values) / len(values),
        }
        seed_rows[condition] = out[condition]
    out["four_minus_one_seed_mean_accuracy"] = out["4pattern"]["seed_mean_accuracy"] - out["1pattern"]["seed_mean_accuracy"]

    c100 = []
    for seed in (42, 123, 777):
        path = ROOT / "faraz_followup_delivery_20260808" / "cifar100_four_pattern" / "raw" / f"seed{seed}" / "generation_samples.jsonl"
        item = summarize(path)
        item["seed"] = seed
        rows = load(path)
        item["positive_reward_n"] = sum((r.get("reward") or 0) > 0 for r in rows)
        c100.append(item)
    c100_values = [v for seed in (42, 123, 777) for v in summarize(ROOT / "faraz_followup_delivery_20260808" / "cifar100_four_pattern" / "raw" / f"seed{seed}" / "generation_samples.jsonl")["accuracy_values"]]
    for item in c100:
        item.pop("accuracy_values")
    out["cifar100_four_pattern_completed_package"] = {
        "seeds": c100,
        "attempted_n": sum(i["attempted_n"] for i in c100),
        "formal_success_n": sum(i["formal_success_n"] for i in c100),
        "positive_reward_n": sum(i["positive_reward_n"] for i in c100),
        "seed_mean_accuracy": sum(i["mean_accuracy"] for i in c100) / 3,
        "seed_sd_accuracy_sample": statistics.stdev(i["mean_accuracy"] for i in c100),
        "pooled_accuracy": sum(c100_values) / len(c100_values),
        "scope_note": "Only the three completed packaged JSONLs are included; no old interrupted path is merged into these counts.",
    }
    target = Path(__file__).resolve().parents[1] / "secondary_consistency_recomputed.json"
    target.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({
        "sft_one": {k: out["1pattern"][k] for k in ("attempted_n", "formal_success_n", "seed_mean_accuracy", "pooled_accuracy")},
        "sft_four": {k: out["4pattern"][k] for k in ("attempted_n", "formal_success_n", "seed_mean_accuracy", "pooled_accuracy")},
        "four_minus_one": out["four_minus_one_seed_mean_accuracy"],
        "cifar100_four_completed": {k: out["cifar100_four_pattern_completed_package"][k] for k in ("attempted_n", "formal_success_n", "positive_reward_n", "seed_mean_accuracy", "pooled_accuracy")},
        "output": str(target),
    }, indent=2))


if __name__ == "__main__":
    main()
