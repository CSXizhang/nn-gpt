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
        value = result.get(key)
        if numeric(value):
            return float(value)
    return math.nan


def reward(row):
    value = row.get("reward")
    if numeric(value):
        return float(value)
    value = api(row).get("reward")
    return float(value) if numeric(value) else math.nan


def completion_len(row):
    result = api(row)
    for key in (
        "completion_token_count",
        "completion_token_length",
        "completion_len",
        "num_completion_tokens",
    ):
        value = result.get(key)
        if numeric(value):
            return float(value)
    text = row.get("completion") or ""
    return float(len(text.split())) if text else math.nan


def formal_success(row):
    result = api(row)
    return bool(
        result.get("formal_success_candidate")
        or (
            result.get("built_ok")
            and result.get("forward_ok")
            and result.get("forward_shape_ok")
            and result.get("backward_ok")
            and not result.get("timed_out")
            and not math.isnan(acc(row))
        )
    )


def executable(row):
    result = api(row)
    return bool(result.get("built_ok") and result.get("forward_ok") and result.get("forward_shape_ok"))


def pair(row):
    result = api(row)
    signature = result.get("backbone_signature")
    if signature:
        return str(signature)
    names = result.get("backbone_model_names") or []
    return str(tuple(names)) if isinstance(names, list) and names else ""


def block(row):
    return api(row).get("block_signature") or row.get("block_signature") or ""


def mean(values):
    clean = [value for value in values if numeric(value) and not math.isnan(value)]
    return statistics.mean(clean) if clean else math.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="RL grouped by 10 reward batches")
    parser.add_argument("--group-size", type=int, default=10)
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    rows = [json.loads(line) for line in inp.read_text().splitlines() if line.strip()]

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

    reward_mean = [mean([reward(row) for row in groups[group]]) for group in x_values]
    acc_mean = [mean([acc(row) for row in groups[group] if formal_success(row)]) for group in x_values]
    acc_max = [max([acc(row) for row in groups[group] if formal_success(row)] or [math.nan]) for group in x_values]
    success_rate = [100.0 * sum(formal_success(row) for row in groups[group]) / len(groups[group]) for group in x_values]
    exec_rate = [100.0 * sum(executable(row) for row in groups[group]) / len(groups[group]) for group in x_values]
    length_mean = [mean([completion_len(row) for row in groups[group]]) for group in x_values]
    positive_rate = [100.0 * sum(reward(row) > 0 for row in groups[group]) / len(groups[group]) for group in x_values]
    unique_pairs = [len({pair(row) for row in groups[group] if formal_success(row) and pair(row)}) for group in x_values]
    unique_blocks = [len({block(row) for row in groups[group] if formal_success(row) and block(row)}) for group in x_values]
    dominant_block_share = []
    for group in x_values:
        blocks = [block(row) for row in groups[group] if formal_success(row) and block(row)]
        if blocks:
            dominant_block_share.append(100.0 * collections.Counter(blocks).most_common(1)[0][1] / len(blocks))
        else:
            dominant_block_share.append(math.nan)

    fig, axes = plt.subplots(3, 2, figsize=(16, 14), constrained_layout=True)
    fig.suptitle(args.title, fontsize=18)

    ax = axes[0, 0]
    ax.plot(x_values, reward_mean, marker="o", color="#ef6c00", label="Reward mean")
    ax.plot(x_values, positive_rate, marker="s", color="#2e7d32", label="Positive reward (%)")
    ax.axhline(0, color="black", linewidth=1, alpha=0.5)
    ax.set_title("Reward")
    ax.set_xlabel("Reward batch group")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(x_values, acc_mean, marker="o", label="Mean frozen test acc", color="#6a1b9a")
    ax.plot(x_values, acc_max, marker="D", label="Max frozen test acc", color="#8e24aa", alpha=0.75)
    ax.set_ylim(0.82, 0.95)
    ax.set_title("Accuracy among formal-success samples")
    ax.set_xlabel("Reward batch group")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(x_values, success_rate, marker="o", label="Formal success", color="#6a1b9a")
    ax.plot(x_values, exec_rate, marker="o", label="Executable", color="#1565c0")
    ax.set_ylim(0, 105)
    ax.set_title("Validity rates")
    ax.set_xlabel("Reward batch group")
    ax.set_ylabel("Rate (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(x_values, unique_pairs, marker="o", label="Unique backbone pairs", color="#2e7d32")
    ax.plot(x_values, unique_blocks, marker="o", label="Unique block signatures", color="#00897b")
    ax.set_title("Within-group diversity")
    ax.set_xlabel("Reward batch group")
    ax.set_ylabel("Unique count")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2, 0]
    ax.plot(x_values, dominant_block_share, marker="o", color="#c62828")
    ax.set_ylim(0, 105)
    ax.set_title("Dominant block share among valid samples")
    ax.set_xlabel("Reward batch group")
    ax.set_ylabel("Share (%)")
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.plot(x_values, length_mean, marker="o", color="#455a64")
    ax.set_title("Completion length mean")
    ax.set_xlabel("Reward batch group")
    ax.set_ylabel("Token count when available")
    ax.grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xticks(x_values)
        ax.set_xticklabels(x_labels, rotation=30, ha="right")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(out)
    print(f"rows={len(rows)} groups={len(x_values)}")


if __name__ == "__main__":
    main()
