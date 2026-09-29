import argparse
import collections
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def api(row):
    return row.get("api_result") or {}


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def acc(row):
    result = api(row)
    horizon = result.get("formal_horizon_test_acc") or {}
    if isinstance(horizon, dict) and numeric(horizon.get("1")):
        return float(horizon["1"])
    for key in ("frozen_test_acc", "test_acc", "val_metric"):
        if numeric(result.get(key)):
            return float(result[key])
    return math.nan


def formal_success(row):
    result = api(row)
    return bool(
        result.get("formal_success_candidate")
        or (
            result.get("built_ok")
            and result.get("forward_ok")
            and result.get("forward_shape_ok")
            and result.get("backward_ok")
            and not math.isnan(acc(row))
        )
    )


def mean(values):
    clean = [value for value in values if numeric(value) and not math.isnan(value)]
    return statistics.mean(clean) if clean else math.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="Accuracy grouped by 10 reward batches")
    parser.add_argument("--group-size", type=int, default=10)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    groups = collections.defaultdict(list)
    for index, row in enumerate(rows):
        batch_index = api(row).get("reward_batch_index")
        try:
            batch_index = int(float(batch_index))
        except Exception:
            batch_index = index // 4
        groups[batch_index // args.group_size].append(row)

    x_values = sorted(groups)
    x_labels = [
        f"{group * args.group_size}-{group * args.group_size + args.group_size - 1}"
        for group in x_values
    ]
    acc_mean = []
    acc_median = []
    acc_max = []
    formal_rate = []
    for group in x_values:
        values = [acc(row) for row in groups[group] if formal_success(row) and not math.isnan(acc(row))]
        acc_mean.append(mean(values))
        acc_median.append(statistics.median(values) if values else math.nan)
        acc_max.append(max(values) if values else math.nan)
        formal_rate.append(100.0 * sum(formal_success(row) for row in groups[group]) / len(groups[group]))

    fig, ax_acc = plt.subplots(figsize=(12, 6), constrained_layout=True)
    ax_acc.plot(x_values, acc_mean, marker="o", linewidth=2.5, label="Mean acc", color="#1565c0")
    ax_acc.plot(x_values, acc_median, marker="s", linewidth=2.0, label="Median acc", color="#7b1fa2")
    ax_acc.plot(x_values, acc_max, marker="D", linewidth=2.0, label="Max acc", color="#ef6c00")
    ax_acc.set_ylim(0.82, 0.93)
    ax_acc.set_ylabel("1-epoch frozen test accuracy")
    ax_acc.set_xlabel("Reward batch group")
    ax_acc.set_xticks(x_values)
    ax_acc.set_xticklabels(x_labels, rotation=25, ha="right")
    ax_acc.grid(True, alpha=0.3)

    ax_rate = ax_acc.twinx()
    ax_rate.plot(
        x_values,
        formal_rate,
        marker="^",
        linestyle="--",
        label="Formal success rate",
        color="#2e7d32",
    )
    ax_rate.set_ylim(0, 105)
    ax_rate.set_ylabel("Formal success rate (%)")

    lines_a, labels_a = ax_acc.get_legend_handles_labels()
    lines_b, labels_b = ax_rate.get_legend_handles_labels()
    ax_acc.legend(lines_a + lines_b, labels_a + labels_b, loc="lower left")
    fig.suptitle(args.title)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    print(output)
    print(f"rows={len(rows)} groups={len(x_values)}")
    print("mean", acc_mean)
    print("median", acc_median)
    print("max", acc_max)
    print("formal_rate", formal_rate)


if __name__ == "__main__":
    main()
