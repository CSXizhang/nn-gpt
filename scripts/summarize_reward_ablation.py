#!/usr/bin/env python3
import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple


FAILURE_CATEGORIES = (
    "format",
    "parse",
    "build",
    "forward",
    "shape-channel",
    "train-step",
    "timeout",
)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _api(record: Dict[str, Any]) -> Dict[str, Any]:
    value = record.get("api_result")
    return value if isinstance(value, dict) else {}


def _optional_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed):
        return None
    return parsed


def _reward(record: Dict[str, Any]) -> float:
    value = _optional_float(record.get("reward"))
    if value is not None:
        return value
    value = _optional_float(_api(record).get("reward"))
    return float(value) if value is not None else 0.0


def _formal_success(record: Dict[str, Any]) -> bool:
    api = _api(record)
    if "formal_success_candidate" in api:
        return bool(api.get("formal_success_candidate"))
    try:
        epochs_completed = int(api.get("epochs_completed", 0) or 0)
    except (TypeError, ValueError):
        epochs_completed = 0
    return bool(
        api.get("built_ok")
        and api.get("forward_shape_ok")
        and api.get("backward_ok")
        and epochs_completed >= 1
    )


def _acc1(record: Dict[str, Any]) -> Optional[float]:
    api = _api(record)
    horizons = api.get("formal_horizon_test_acc")
    if isinstance(horizons, dict):
        for key in ("1", 1):
            value = _optional_float(horizons.get(key))
            if value is not None:
                return value
    for key in ("frozen_test_acc", "test_acc", "val_metric"):
        value = _optional_float(api.get(key))
        if value is not None:
            return value
    return None


def _counter_value(record: Dict[str, Any], key: str) -> str:
    api = _api(record)
    value = api.get(key)
    if value is None:
        value = (api.get("open_discovery") or {}).get(key) if isinstance(api.get("open_discovery"), dict) else None
    if value is None:
        return ""
    return str(value)


def _effective_number(values: Iterable[str]) -> float:
    counts = Counter(value for value in values if value)
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        prob = float(count) / float(total)
        entropy -= prob * math.log(prob)
    return math.exp(entropy)


def _top_share(values: Iterable[str]) -> float:
    counts = Counter(value for value in values if value)
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return float(max(counts.values())) / float(total)


def _text_blob(record: Dict[str, Any]) -> str:
    api = _api(record)
    parts: List[str] = []
    for mapping in (record, api):
        for key in (
            "error",
            "error_type",
            "error_stage",
            "error_context",
            "error_hint",
            "generation_error",
        ):
            value = mapping.get(key)
            if value is not None:
                parts.append(str(value))
    frozen_eval = api.get("frozen_eval")
    if isinstance(frozen_eval, dict):
        for key in ("error", "error_type", "error_stage", "error_context", "error_hint"):
            value = frozen_eval.get(key)
            if value is not None:
                parts.append(str(value))
    return " ".join(parts).lower()


def _failure_mode(record: Dict[str, Any]) -> str:
    api = _api(record)
    raw = api.get("raw_extraction") if isinstance(api.get("raw_extraction"), dict) else {}
    blob = _text_blob(record)
    completion = str(record.get("completion") or "")

    if bool(api.get("timed_out")) or "timeout" in blob or "timed out" in blob or "time limit" in blob:
        return "timeout"
    if (
        record.get("generation_error")
        or not completion.strip()
        or raw.get("xml_tag_exact") is False
        or raw.get("exact_block_signature") is False
        or raw.get("exact_init_signature") is False
        or raw.get("exact_forward_signature") is False
        or "tags missing" in blob
        or "raw_completion" in blob
    ):
        return "format"
    open_discovery = api.get("open_discovery") if isinstance(api.get("open_discovery"), dict) else {}
    parse_ok = api.get("parse_ok", open_discovery.get("parse_ok"))
    if parse_ok is False or any(token in blob for token in ("syntax", "parse", "ast", "indent")):
        return "parse"
    if api.get("built_ok") is False or any(token in blob for token in ("build", "instantiate", "importerror", "nameerror")):
        return "build"
    if (
        api.get("forward_shape_ok") is False
        or any(
            token in blob
            for token in (
                "shape",
                "channel",
                "dimension",
                "mat1",
                "mat2",
                "logits",
                "num_classes",
                "in_channels",
            )
        )
    ):
        return "shape-channel"
    if api.get("forward_ok") is False or "forward" in blob:
        return "forward"
    return "train-step"


def _cost_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    formal_evals = 0
    ten_epoch_reevals = 0
    eval_seconds = 0.0
    for record in rows:
        api = _api(record)
        try:
            epochs_completed = int(api.get("epochs_completed", 0) or 0)
        except (TypeError, ValueError):
            epochs_completed = 0
        if epochs_completed >= 1 or api.get("formal_horizon_test_acc"):
            formal_evals += 1
        horizons = api.get("formal_horizon_test_acc")
        formal_epochs = api.get("formal_reward_epochs") or []
        if (isinstance(horizons, dict) and "10" in horizons) or 10 in formal_epochs:
            ten_epoch_reevals += 1
        seconds = _optional_float(api.get("estimated_total_seconds"))
        if seconds is not None:
            eval_seconds += seconds
    return {
        "generated_samples": len(rows),
        "formal_evals": formal_evals,
        "ten_epoch_reevaluations": ten_epoch_reevals,
        "rough_gpu_hours": eval_seconds / 3600.0,
    }


def _summarize_variant(variant: str, path: Path) -> Tuple[Dict[str, Any], Dict[str, int], Dict[str, Any]]:
    rows = _read_jsonl(path)
    successes = [record for record in rows if _formal_success(record)]
    diversity_pool = successes
    acc_values = [value for value in (_acc1(record) for record in successes) if value is not None]
    backbone_values = [_counter_value(record, "backbone_signature") for record in diversity_pool]
    module_values = [_counter_value(record, "block_signature") for record in diversity_pool]
    graph_values = [_counter_value(record, "graph_hash") for record in diversity_pool]
    failure_counts = Counter(_failure_mode(record) for record in rows if not _formal_success(record))
    for category in FAILURE_CATEGORIES:
        failure_counts.setdefault(category, 0)
    summary = {
        "variant": variant,
        "samples": len(rows),
        "formal_success": f"{len(successes)}/{len(rows)}",
        "formal_success_count": len(successes),
        "formal_success_rate": (len(successes) / len(rows)) if rows else 0.0,
        "acc1_mean": mean(acc_values) if acc_values else None,
        "acc1_max": max(acc_values) if acc_values else None,
        "positive_reward": f"{sum(1 for record in rows if _reward(record) > 0.0)}/{len(rows)}",
        "positive_reward_count": sum(1 for record in rows if _reward(record) > 0.0),
        "positive_reward_rate": (sum(1 for record in rows if _reward(record) > 0.0) / len(rows)) if rows else 0.0,
        "backbone_eff": _effective_number(backbone_values),
        "module_eff": _effective_number(module_values),
        "graph_eff": _effective_number(graph_values),
        "top_backbone_share": _top_share(backbone_values),
        "top_module_share": _top_share(module_values),
        "top_graph_share": _top_share(graph_values),
    }
    return summary, dict(failure_counts), _cost_summary(rows)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown_table(headers: List[str], rows: List[List[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(value) for value in row) + " |")
    return "\n".join(lines)


def _parse_variant_path(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    variant, path = value.split("=", 1)
    return variant.strip(), Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize four-pattern reward ablation samples.")
    parser.add_argument("inputs", nargs="+", help="variant=path/to/generation_samples.jsonl")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    summaries: List[Dict[str, Any]] = []
    failures: Dict[str, Dict[str, int]] = {}
    costs: Dict[str, Dict[str, Any]] = {}
    for item in args.inputs:
        variant, path = _parse_variant_path(item)
        summary, failure_counts, cost_summary = _summarize_variant(variant, path)
        summaries.append(summary)
        failures[variant] = failure_counts
        costs[variant] = cost_summary

    metric_headers = [
        "variant",
        "samples",
        "formal success",
        "acc1 mean",
        "acc1 max",
        "positive reward",
        "backbone eff",
        "module eff",
        "graph eff",
        "top backbone share",
        "top module share",
        "top graph share",
    ]
    metric_rows = [
        [
            row["variant"],
            row["samples"],
            row["formal_success"],
            row["acc1_mean"],
            row["acc1_max"],
            row["positive_reward"],
            row["backbone_eff"],
            row["module_eff"],
            row["graph_eff"],
            row["top_backbone_share"],
            row["top_module_share"],
            row["top_graph_share"],
        ]
        for row in summaries
    ]
    print("## Small-scale reward ablation")
    print(_markdown_table(metric_headers, metric_rows))
    print()

    failure_headers = ["variant", *FAILURE_CATEGORIES]
    failure_rows = [[variant, *(failures[variant].get(category, 0) for category in FAILURE_CATEGORIES)] for variant in failures]
    print("## Failure-mode breakdown")
    print(_markdown_table(failure_headers, failure_rows))
    print()

    cost_headers = ["variant", "generated samples", "formal evals", "10-epoch reevaluations", "rough GPU hours"]
    cost_rows = [
        [
            variant,
            cost["generated_samples"],
            cost["formal_evals"],
            cost["ten_epoch_reevaluations"],
            cost["rough_gpu_hours"],
        ]
        for variant, cost in costs.items()
    ]
    print("## Compute cost estimate")
    print(_markdown_table(cost_headers, cost_rows))

    if args.json_out is not None:
        payload = {"summaries": summaries, "failures": failures, "costs": costs}
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
