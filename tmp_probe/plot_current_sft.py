import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


epoch_root = Path(os.environ["EPOCH_ROOT"])
output = Path(os.environ["PLOT_OUTPUT"])

stats = defaultdict(lambda: {"success": 0, "exists": 0, "total": 0, "acc": []})

for a_dir in sorted(epoch_root.glob("A*")):
    if not a_dir.is_dir():
        continue
    match = re.fullmatch(r"A(\d+)", a_dir.name)
    if not match:
        continue
    cycle = int(match.group(1))
    synth = a_dir / "synth_nn"
    if not synth.exists():
        continue
    b_dirs = sorted([p for p in synth.glob("B*") if p.is_dir()])
    stats[cycle]["total"] = len(b_dirs)
    for b_dir in b_dirs:
        json_path = b_dir / "1.json"
        error_path = b_dir / "error.txt"
        if json_path.exists():
            stats[cycle]["success"] += 1
            try:
                data = json.loads(json_path.read_text())
                row = data[0] if isinstance(data, list) else data
                acc = float(row.get("accuracy", 0.0)) * 100.0
                stats[cycle]["acc"].append(max(0.0, min(100.0, acc)))
            except Exception:
                pass
        elif error_path.exists() and "NN already exists" in error_path.read_text(errors="ignore"):
            stats[cycle]["exists"] += 1

cycles = sorted(stats)
valid_rate = []
known_rate = []
best_acc = []
mean_acc = []
median_acc = []
low_err = []
high_err = []
success_counts = []
exists_counts = []

for cycle in cycles:
    row = stats[cycle]
    total = row["total"] or 0
    success = row["success"]
    exists = row["exists"]
    acc = np.array(row["acc"], dtype=float)
    valid_rate.append(success / total * 100 if total else 0)
    known_rate.append((success + exists) / total * 100 if total else 0)
    success_counts.append(success)
    exists_counts.append(exists)
    if acc.size:
        best_acc.append(float(np.max(acc)))
        mean = float(np.mean(acc))
        mean_acc.append(mean)
        median_acc.append(float(np.median(acc)))
        if acc.size >= 2:
            sem = float(np.std(acc, ddof=1) / math.sqrt(acc.size))
            ci = 1.96 * sem
        else:
            ci = 0.0
        low_err.append(max(0.0, mean - max(0.0, mean - ci)))
        high_err.append(max(0.0, min(100.0, mean + ci) - mean))
    else:
        best_acc.append(np.nan)
        mean_acc.append(np.nan)
        median_acc.append(np.nan)
        low_err.append(0.0)
        high_err.append(0.0)

fig, ax = plt.subplots(figsize=(14, 7), dpi=150)
ax.grid(True, linestyle="--", alpha=0.3)
ax.plot(cycles, valid_rate, marker="o", linewidth=2.4, label="Valid generation rate")
ax.plot(cycles, known_rate, marker="s", linewidth=2.0, linestyle="--", label="Valid + duplicate-known rate")
ax.errorbar(cycles, mean_acc, yerr=[low_err, high_err], fmt="o", color="#f57c00", ecolor="#ffb74d", capsize=4, label="Mean acc +/- 95% CI")
ax.plot(cycles, best_acc, marker="D", color="#d84315", linewidth=1.8, label="Best acc")
ax.scatter(cycles, median_acc, marker="_", s=220, color="#b71c1c", linewidths=3, label="Median acc")

for cycle, success, exists in zip(cycles, success_counts, exists_counts):
    if cycle % 2 == 0 or cycle == cycles[-1]:
        ax.text(cycle, min(98, valid_rate[cycles.index(cycle)] + 3), f"{success}/30", ha="center", fontsize=8)
    if exists:
        ax.text(cycle, 4, f"dup {exists}", ha="center", fontsize=7, color="#7b1fa2")

ax.set_title("1-pattern aligned SFT current run")
ax.set_xlabel("Cycle")
ax.set_ylabel("Percent")
ax.set_ylim(0, 100)
ax.set_xticks(cycles)
ax.legend(loc="lower right", ncol=2)
fig.tight_layout()
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output)
print(output)
