#!/usr/bin/env python3
"""Seed-level diagnostics for the 5-epoch article RL runs."""

from __future__ import annotations

import ast
import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUN_BASE = Path("/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs")
OUT_DIR = Path("/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/article_diagnostics_20260627")

RUNS = [
    {
        "condition": "1-pattern",
        "seed": "42",
        "run_id": "20260618_1pattern_a9_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed42",
        "role": "original",
    },
    {
        "condition": "1-pattern",
        "seed": "123",
        "run_id": "20260618_1pattern_a9_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed123",
        "role": "original",
    },
    {
        "condition": "1-pattern",
        "seed": "777",
        "run_id": "20260618_1pattern_a9_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed777",
        "role": "original",
    },
    {
        "condition": "1-pattern",
        "seed": "114",
        "run_id": "20260627_1pattern_a9_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed114",
        "role": "added",
    },
    {
        "condition": "1-pattern",
        "seed": "514",
        "run_id": "20260627_1pattern_a9_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed514",
        "role": "added",
    },
    {
        "condition": "4-pattern",
        "seed": "42",
        "run_id": "20260612_4pattern_a18_cifar10_e5_p3500_c1200_s100_l40s_excl_full_full_reward_seed42",
        "role": "original",
    },
    {
        "condition": "4-pattern",
        "seed": "123",
        "run_id": "20260612_4pattern_a18_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed123",
        "role": "original",
    },
    {
        "condition": "4-pattern",
        "seed": "777",
        "run_id": "20260612_4pattern_a18_cifar10_e5_p3500_c1200_s100_l40s_excl_full_full_reward_seed777",
        "role": "original",
    },
    {
        "condition": "4-pattern",
        "seed": "114",
        "run_id": "20260627_4pattern_a18_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed114",
        "role": "added",
    },
    {
        "condition": "4-pattern",
        "seed": "514",
        "run_id": "20260627_4pattern_a18_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed514",
        "role": "added",
    },
    {
        "condition": "4-pattern",
        "seed": "919",
        "run_id": "20260627_4pattern_a18_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed919",
        "role": "reserve_running",
    },
]


R_FIELDS = [
    "r_dense",
    "r_formal_success_signal",
    "r_target_structure_penalty",
    "r_repeat_family",
    "r_structure_group",
    "r_structure_archive",
    "r_descriptor_diversity",
    "r_cnn_diversity",
    "r_block_diversity",
    "r_trainset_novelty",
    "r_generalization",
    "r_batch_elite",
    "r_plain_fuse_penalty",
    "r_template_penalty",
    "r_no_progress_penalty",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(errors="ignore") as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def api(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("api_result")
    return value if isinstance(value, dict) else {}


def sample_path(run_id: str) -> Path:
    root = RUN_BASE / run_id
    candidates = [
        root / "rl_output" / "generation_samples.jsonl",
        root / "rl_output" / "sft" / "generation_samples.jsonl",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def scalar(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return float(value)
    return None


def mean(values: list[Any]) -> float | None:
    vals = [scalar(v) for v in values]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def fmt_num(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def acc_value(a: dict[str, Any]) -> float | None:
    horizon = a.get("formal_horizon_test_acc")
    if isinstance(horizon, dict):
        for key in ("5", 5, "1", 1):
            if key in horizon:
                out = scalar(horizon[key])
                if out is not None:
                    return out
    for key in ("test_acc", "val_metric", "reward_target_value"):
        out = scalar(a.get(key))
        if out is not None:
            return out
    return None


def norm_label(value: Any, fallback: str = "NA") -> str:
    if value is None:
        return fallback
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)[:240]
    return str(value)


def entropy_from_counts(counts: Counter[str]) -> float | None:
    total = sum(counts.values())
    if total <= 0:
        return None
    ent = 0.0
    for count in counts.values():
        p = count / total
        ent -= p * math.log(p)
    return math.exp(ent)


def top_share(values: list[str]) -> float | None:
    vals = [v for v in values if v and v != "NA"]
    if not vals:
        return None
    counts = Counter(vals)
    return counts.most_common(1)[0][1] / len(vals)


def top_label(values: list[str]) -> str:
    vals = [v for v in values if v and v != "NA"]
    if not vals:
        return "NA"
    return Counter(vals).most_common(1)[0][0]


def classify_failure(a: dict[str, Any]) -> str:
    error_type = norm_label(a.get("error_type"), "")
    error_stage = norm_label(a.get("error_stage"), "")
    mismatch = norm_label(a.get("target_structure_mismatch_reasons"), "")
    if error_type == "SyntaxError":
        return "syntax_template_or_format_failure"
    if error_type:
        if error_stage == "cpu_prevalidate":
            return f"cpu_prevalidate_{error_type}"
        return error_type
    if "target_parallel_triple_but_block_dead" in mismatch:
        return "target_structure_dead_block"
    if a.get("formal_success_candidate") is True and a.get("target_pattern_match") is False:
        return "formal_success_target_mismatch"
    if a.get("formal_success_candidate") is True and scalar(a.get("reward")) is not None and float(a["reward"]) <= 0:
        return "formal_success_nonpositive_reward"
    if a.get("formal_success_candidate") is True:
        return "formal_success"
    if a.get("built_ok") is False:
        return "build_or_prevalidation_failure"
    return "unclassified"


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    apis = [api(row) for row in rows]
    rewards = [a.get("reward") for a in apis]
    accs = [acc_value(a) for a in apis]
    accs = [v for v in accs if v is not None]
    built = sum(a.get("built_ok") is True for a in apis)
    formal = sum(a.get("formal_success_candidate") is True for a in apis)
    positive = sum(scalar(a.get("reward")) is not None and float(a["reward"]) > 0 for a in apis)
    timeouts = sum(a.get("timed_out") is True or a.get("error_type") in {"timeout", "LearnTimeException"} for a in apis)
    family_vals = [norm_label(a.get("family_hash") or a.get("family_id")) for a in apis if a.get("formal_success_candidate") is True]
    graph_vals = [norm_label(a.get("graph_hash")) for a in apis if a.get("formal_success_candidate") is True]
    desc_vals = [norm_label(a.get("descriptor_key")) for a in apis if a.get("formal_success_candidate") is True]
    if not family_vals:
        family_vals = [norm_label(a.get("family_hash") or a.get("family_id")) for a in apis]
        graph_vals = [norm_label(a.get("graph_hash")) for a in apis]
        desc_vals = [norm_label(a.get("descriptor_key")) for a in apis]
    failures = Counter(classify_failure(a) for a in apis)
    return {
        "n": len(rows),
        "built": built,
        "formal": formal,
        "positive_reward": positive,
        "timeouts": timeouts,
        "mean_reward": mean(rewards),
        "mean_acc": mean(accs),
        "max_acc": max(accs) if accs else None,
        "family_top1": top_share(family_vals),
        "graph_top1": top_share(graph_vals),
        "descriptor_top1": top_share(desc_vals),
        "family_eff": entropy_from_counts(Counter(family_vals)),
        "graph_eff": entropy_from_counts(Counter(graph_vals)),
        "descriptor_eff": entropy_from_counts(Counter(desc_vals)),
        "dominant_family": top_label(family_vals),
        "dominant_graph": top_label(graph_vals),
        "dominant_descriptor": top_label(desc_vals),
        "dominant_failure": failures.most_common(1)[0][0] if failures else "NA",
        "dominant_failure_count": failures.most_common(1)[0][1] if failures else 0,
    }


def window_slices(rows: list[dict[str, Any]], size: int = 100) -> list[tuple[int, int, list[dict[str, Any]]]]:
    out = []
    for start in range(0, len(rows), size):
        chunk = rows[start : start + size]
        if chunk:
            out.append((start + 1, start + len(chunk), chunk))
    return out


def reward_component_row(meta: dict[str, str], start: int, end: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    apis = [api(row) for row in rows]
    out: dict[str, Any] = {
        **meta,
        "window_start": start,
        "window_end": end,
        "n": len(rows),
    }
    for key in ["reward", "test_acc", "formal_reward_target_value"]:
        out[key] = mean([a.get(key) for a in apis])
    for key in R_FIELDS:
        out[key] = mean([a.get(key) for a in apis])
    return out


def extract_policy_trace(run_id: str) -> list[dict[str, Any]]:
    root = RUN_BASE / run_id
    paths = list((root / "slurm").glob("*.out")) + [root / "rl_output" / "training_progress.log"]
    trace: list[dict[str, Any]] = []
    seen = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open(errors="ignore") as handle:
            for line in handle:
                text = line.strip()
                if not (text.startswith("{") and "entropy" in text and "reward" in text):
                    continue
                try:
                    data = ast.literal_eval(text)
                except Exception:
                    continue
                item = {
                    "source": path.name,
                    "step_index": len(trace) + 1,
                    "policy_entropy": scalar(data.get("entropy")),
                    "trainer_reward": scalar(data.get("reward")),
                    "trainer_loss": scalar(data.get("loss")),
                    "kl": scalar(data.get("kl")),
                    "epoch": scalar(data.get("epoch")),
                }
                key = (item["source"], item["step_index"], item["policy_entropy"], item["trainer_reward"])
                if key in seen:
                    continue
                seen.add(key)
                trace.append(item)
    return trace


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            cleaned = {}
            for key in fieldnames:
                value = row.get(key)
                if isinstance(value, float):
                    cleaned[key] = f"{value:.8g}"
                else:
                    cleaned[key] = value
            writer.writerow(cleaned)


def latex_table(rows: list[dict[str, Any]]) -> str:
    header = (
        "\\begin{tabular}{lllrrrrrl}\n"
        "\\toprule\n"
        "Condition & Seed & Role & Samples & Formal & Mean Acc. & Mean Reward & Pos. Reward & Dominant outcome \\\\\n"
        "\\midrule\n"
    )
    body = []
    for row in rows:
        body.append(
            f"{row['condition']} & {row['seed']} & {row['role']} & "
            f"{row['samples']} & {row['final_formal']}/{row['final_n']} & "
            f"{fmt_num(row['final_mean_acc'], 3)} & {fmt_num(row['final_mean_reward'], 3)} & "
            f"{row['final_positive_reward']}/{row['final_n']} & {row['dominant_outcome']} \\\\"
        )
    footer = "\n\\bottomrule\n\\end{tabular}\n"
    return header + "\n".join(body) + footer


def write_plots(window_rows: list[dict[str, Any]], policy_rows: list[dict[str, Any]]) -> list[str]:
    outputs: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        (OUT_DIR / "plot_error.txt").write_text(str(exc))
        return outputs

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in window_rows:
        grouped[f"{row['condition']} seed{row['seed']}"].append(row)

    plt.figure(figsize=(12, 7))
    for label, rows in grouped.items():
        rows = sorted(rows, key=lambda x: int(x["window_end"]))
        xs = [int(r["window_end"]) for r in rows]
        ys = [float(r["family_top1"]) if r.get("family_top1") not in (None, "NA", "") else math.nan for r in rows]
        plt.plot(xs, ys, marker="o", linewidth=1.4, label=label)
    plt.axhline(0.8, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Generated candidates")
    plt.ylabel("Dominant family share")
    plt.title("Dominant-family trajectory by seed")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    out = OUT_DIR / "dominant_family_trajectory.png"
    plt.savefig(out, dpi=180)
    outputs.append(str(out))
    plt.close()

    plt.figure(figsize=(12, 7))
    for label, rows in grouped.items():
        rows = sorted(rows, key=lambda x: int(x["window_end"]))
        xs = [int(r["window_end"]) for r in rows]
        ys = [float(r["mean_reward"]) if r.get("mean_reward") not in (None, "NA", "") else math.nan for r in rows]
        plt.plot(xs, ys, marker="o", linewidth=1.4, label=label)
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Generated candidates")
    plt.ylabel("Mean reward")
    plt.title("Reward trajectory by seed")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    out = OUT_DIR / "reward_trajectory.png"
    plt.savefig(out, dpi=180)
    outputs.append(str(out))
    plt.close()

    grouped_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in policy_rows:
        grouped_policy[f"{row['condition']} seed{row['seed']}"].append(row)
    if grouped_policy:
        plt.figure(figsize=(12, 7))
        for label, rows in grouped_policy.items():
            rows = sorted(rows, key=lambda x: int(x["step_index"]))
            xs = [int(r["step_index"]) for r in rows]
            ys = [float(r["policy_entropy"]) if r.get("policy_entropy") not in (None, "NA", "") else math.nan for r in rows]
            plt.plot(xs, ys, linewidth=1.2, label=label)
        plt.xlabel("Trainer log record")
        plt.ylabel("Policy entropy")
        plt.title("Post-hoc policy entropy trace")
        plt.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        out = OUT_DIR / "policy_entropy_trace.png"
        plt.savefig(out, dpi=180)
        outputs.append(str(out))
        plt.close()
    return outputs


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    reward_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []

    for run in RUNS:
        path = sample_path(run["run_id"])
        rows = read_jsonl(path)
        meta = {
            "condition": run["condition"],
            "seed": run["seed"],
            "role": run["role"],
            "run_id": run["run_id"],
        }
        all_summary = summarize_rows(rows)
        final_rows = rows[-100:] if len(rows) >= 100 else rows
        final_summary = summarize_rows(final_rows)
        completed = len(rows) >= 800

        seed_rows.append(
            {
                **meta,
                "sample_path": str(path),
                "completed": completed,
                "samples": len(rows),
                "all_formal": all_summary["formal"],
                "all_built": all_summary["built"],
                "all_positive_reward": all_summary["positive_reward"],
                "all_mean_acc": all_summary["mean_acc"],
                "all_max_acc": all_summary["max_acc"],
                "all_mean_reward": all_summary["mean_reward"],
                "final_n": final_summary["n"],
                "final_formal": final_summary["formal"],
                "final_built": final_summary["built"],
                "final_positive_reward": final_summary["positive_reward"],
                "final_mean_acc": final_summary["mean_acc"],
                "final_max_acc": final_summary["max_acc"],
                "final_mean_reward": final_summary["mean_reward"],
                "final_family_top1": final_summary["family_top1"],
                "final_graph_top1": final_summary["graph_top1"],
                "final_descriptor_top1": final_summary["descriptor_top1"],
                "final_family_eff": final_summary["family_eff"],
                "final_graph_eff": final_summary["graph_eff"],
                "final_descriptor_eff": final_summary["descriptor_eff"],
                "dominant_outcome": final_summary["dominant_failure"],
                "dominant_outcome_count": final_summary["dominant_failure_count"],
                "dominant_family": final_summary["dominant_family"],
                "dominant_graph": final_summary["dominant_graph"],
                "dominant_descriptor": final_summary["dominant_descriptor"],
            }
        )

        for start, end, chunk in window_slices(rows):
            summary = summarize_rows(chunk)
            window_rows.append(
                {
                    **meta,
                    "window_start": start,
                    "window_end": end,
                    "n": summary["n"],
                    "built": summary["built"],
                    "formal": summary["formal"],
                    "positive_reward": summary["positive_reward"],
                    "mean_acc": summary["mean_acc"],
                    "max_acc": summary["max_acc"],
                    "mean_reward": summary["mean_reward"],
                    "family_top1": summary["family_top1"],
                    "graph_top1": summary["graph_top1"],
                    "descriptor_top1": summary["descriptor_top1"],
                    "family_eff": summary["family_eff"],
                    "graph_eff": summary["graph_eff"],
                    "descriptor_eff": summary["descriptor_eff"],
                    "dominant_failure": summary["dominant_failure"],
                    "dominant_failure_count": summary["dominant_failure_count"],
                    "dominant_family": summary["dominant_family"],
                    "dominant_graph": summary["dominant_graph"],
                    "dominant_descriptor": summary["dominant_descriptor"],
                }
            )
            reward_rows.append(reward_component_row(meta, start, end, chunk))
            counts = Counter(classify_failure(api(row)) for row in chunk)
            for mode, count in counts.most_common():
                failure_rows.append(
                    {
                        **meta,
                        "window_start": start,
                        "window_end": end,
                        "failure_mode": mode,
                        "count": count,
                        "n": len(chunk),
                        "share": count / len(chunk) if chunk else None,
                    }
                )

        for item in extract_policy_trace(run["run_id"]):
            policy_rows.append({**meta, **item})

    # Collapse onset table.
    onset_rows = []
    for seed_row in seed_rows:
        meta = {k: seed_row[k] for k in ("condition", "seed", "role", "run_id")}
        relevant = [r for r in window_rows if r["run_id"] == seed_row["run_id"]]
        row = {**meta}
        for metric in ("family_top1", "descriptor_top1", "graph_top1"):
            onset = "never"
            for item in sorted(relevant, key=lambda x: int(x["window_end"])):
                value = scalar(item.get(metric))
                if value is not None and value >= 0.8:
                    onset = str(item["window_end"])
                    break
            row[f"{metric}_ge_80_onset"] = onset
        onset_rows.append(row)

    write_csv(OUT_DIR / "seed_outcomes.csv", seed_rows)
    write_csv(OUT_DIR / "window_metrics.csv", window_rows)
    write_csv(OUT_DIR / "failure_mode_counts.csv", failure_rows)
    write_csv(OUT_DIR / "reward_components_by_window.csv", reward_rows)
    write_csv(OUT_DIR / "policy_entropy_trace.csv", policy_rows)
    write_csv(OUT_DIR / "collapse_onsets.csv", onset_rows)

    (OUT_DIR / "seed_outcomes_table.tex").write_text(latex_table(seed_rows))
    plots = write_plots(window_rows, policy_rows)

    summary_lines = [
        "# Article Seed Diagnostics 2026-06-27",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Seed Outcomes",
        "",
        "| Condition | Seed | Role | Samples | Final formal | Final acc | Final reward | Positive reward | Dominant outcome |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in seed_rows:
        summary_lines.append(
            f"| {row['condition']} | {row['seed']} | {row['role']} | {row['samples']} | "
            f"{row['final_formal']}/{row['final_n']} | {fmt_num(row['final_mean_acc'], 4)} | "
            f"{fmt_num(row['final_mean_reward'], 4)} | {row['final_positive_reward']}/{row['final_n']} | "
            f"{row['dominant_outcome']} |"
        )
    summary_lines.extend(
        [
            "",
            "## Collapse Onsets",
            "",
            "| Condition | Seed | Family >=80% | Descriptor >=80% | Graph >=80% |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in onset_rows:
        summary_lines.append(
            f"| {row['condition']} | {row['seed']} | {row['family_top1_ge_80_onset']} | "
            f"{row['descriptor_top1_ge_80_onset']} | {row['graph_top1_ge_80_onset']} |"
        )
    summary_lines.extend(["", "## Files", ""])
    for name in [
        "seed_outcomes.csv",
        "window_metrics.csv",
        "failure_mode_counts.csv",
        "reward_components_by_window.csv",
        "policy_entropy_trace.csv",
        "collapse_onsets.csv",
        "seed_outcomes_table.tex",
    ]:
        summary_lines.append(f"- `{OUT_DIR / name}`")
    for plot in plots:
        summary_lines.append(f"- `{plot}`")
    (OUT_DIR / "summary.md").write_text("\n".join(summary_lines) + "\n")

    print(OUT_DIR)
    print("seed_rows", len(seed_rows))
    print("window_rows", len(window_rows))
    print("failure_rows", len(failure_rows))
    print("policy_rows", len(policy_rows))


if __name__ == "__main__":
    main()
