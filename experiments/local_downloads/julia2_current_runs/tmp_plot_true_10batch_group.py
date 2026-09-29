import collections
import json
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


SRC = Path(
    "/home/s471802/nn-gpt/parallel_runs/"
    "20260528_1847_struct1_v2_a18_stage2_full_h100_rl_w4/"
    "rl_output/generation_samples.jsonl"
)
OUT = Path(
    "/home/s471802/nn-gpt/parallel_runs/"
    "20260528_1847_struct1_v2_a18_stage2_full_h100_rl_w4/"
    "rl_output/reward_dashboard_true_10batch_group.png"
)


def api_result(row):
    return row.get("api_result") or {}


def num(row, key):
    value = api_result(row).get(key)
    if isinstance(value, (int, float)) and not math.isnan(float(value)):
        return float(value)
    return math.nan


def flag(row, key):
    value = api_result(row).get(key)
    if value is True:
        return 1.0
    if value is False:
        return 0.0
    return math.nan


def signature(row, key):
    value = api_result(row).get(key)
    if isinstance(value, list):
        return " + ".join(map(str, value))
    if value not in (None, ""):
        return str(value)
    return None


def mean(values):
    valid = [value for value in values if isinstance(value, (int, float)) and not math.isnan(value)]
    return statistics.mean(valid) if valid else math.nan


rows = [json.loads(line) for line in SRC.read_text().splitlines() if line.strip()]
groups = collections.defaultdict(list)
for index, row in enumerate(rows):
    reward_batch = api_result(row).get("reward_batch_index")
    if not isinstance(reward_batch, (int, float)):
        reward_batch = index // 4 + 1
    group_id = (int(reward_batch) - 1) // 10 + 1
    groups[group_id].append(row)

xs = []
nvals = []
reward = []
acc = []
target = []
formal = []
executable = []
block_unique = []
cnn_unique = []
backbone_unique = []
family_unique = []

for group_id in sorted(groups):
    group = groups[group_id]
    xs.append(group_id)
    nvals.append(len(group))
    reward.append(mean([float(row.get("reward")) for row in group if isinstance(row.get("reward"), (int, float))]))
    acc.append(mean([num(row, "frozen_test_acc") for row in group]))
    target.append(mean([num(row, "formal_reward_target_value") for row in group]))
    formal.append(mean([flag(row, "formal_success_candidate") for row in group]) * 100.0)
    executable.append(mean([flag(row, "executable_candidate") for row in group]) * 100.0)
    block_unique.append(len({signature(row, "block_signature") for row in group if signature(row, "block_signature")}))
    cnn_unique.append(len({signature(row, "cnn_signature") for row in group if signature(row, "cnn_signature")}))
    backbone_unique.append(len({signature(row, "backbone_signature") for row in group if signature(row, "backbone_signature")}))
    family_unique.append(len({signature(row, "family_id") for row in group if signature(row, "family_id")}))

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=True)
fig.suptitle("Struct1 v2 A18 RL - true 10 reward batches per group", fontsize=15, fontweight="bold")

axes[0].plot(xs, reward, marker="o", label="Mean reward", color="#ef6c00")
axes[0].axhline(0, color="#777", lw=1)
axes[0].set_ylabel("Reward")
axes[0].legend(loc="best")

axes[1].plot(xs, acc, marker="o", label="Mean frozen test acc", color="#1565c0")
axes[1].plot(xs, target, marker="s", label="Mean reward target", color="#2e7d32")
axes[1].set_ylabel("Accuracy")
axes[1].set_ylim(0.78, 0.95)
axes[1].legend(loc="best")

axes[2].plot(xs, formal, marker="o", label="Formal success %", color="#6a1b9a")
axes[2].plot(xs, executable, marker="s", label="Executable %", color="#00897b")
axes[2].set_ylabel("Rate (%)")
axes[2].set_ylim(0, 105)
axes[2].legend(loc="best")

axes[3].plot(xs, block_unique, marker="o", label="Unique block", color="#c62828")
axes[3].plot(xs, cnn_unique, marker="s", label="Unique CNN", color="#ad6d00")
axes[3].plot(xs, backbone_unique, marker="^", label="Unique backbone", color="#455a64")
axes[3].plot(xs, family_unique, marker="d", label="Unique family", color="#5e35b1")
axes[3].set_ylabel("Unique count / group")
axes[3].set_xlabel("Group index (10 reward batches = 40 samples)")
axes[3].legend(loc="best", ncol=2)

for axis in axes:
    axis.grid(True, alpha=0.3)
    axis.spines[["top", "right"]].set_visible(False)

for index, (x_value, sample_count) in enumerate(zip(xs, nvals)):
    if sample_count != 40:
        axes[3].annotate(
            f"n={sample_count}",
            (x_value, block_unique[index]),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8,
        )

fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT, dpi=180)
print(f"saved={OUT}")
print(f"rows={len(rows)} groups={len(xs)} last_group_n={nvals[-1] if nvals else 0}")
