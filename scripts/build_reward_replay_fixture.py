#!/usr/bin/env python3
"""Build a compact real-training reward replay input fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_SOURCES = (
    "downloads/julia2_current_runs/elitefix_generation_samples.jsonl",
    "downloads/julia2_current_runs/a12_w10_data/generation_samples.jsonl",
    "downloads/julia2_current_runs/struct1_v2_a18_stage2_full_h100_rl_w4_generation_samples.jsonl",
    "../outputs/manual-20260518-rl-stats/baseline_generation_samples.jsonl",
    "../outputs/manual-20260518-rl-stats/a3_generation_samples.jsonl",
)


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _api(record: Dict[str, Any]) -> Dict[str, Any]:
    api = record.get("api_result")
    return api if isinstance(api, dict) else {}


def _category_keys(record: Dict[str, Any]) -> List[str]:
    api = _api(record)
    open_discovery = api.get("open_discovery") if isinstance(api.get("open_discovery"), dict) else {}
    reward = float(record.get("reward", api.get("reward", 0.0)) or 0.0)
    keys = ["positive" if reward > 0.0 else "non_positive"]
    if api.get("formal_success_candidate"):
        keys.append("formal_success")
    if not api.get("built_ok"):
        keys.append("not_built")
    if api.get("target_structure_match") is False:
        keys.append("target_mismatch")
    if api.get("xml_incomplete_length_cap"):
        keys.append("xml_incomplete_length_cap")
    if api.get("r_length_compactness") or api.get("r_repeated_line_penalty"):
        keys.append("compactness")
    for name in ("descriptor", "cnn", "block"):
        flag = f"{name}_reward_cap_applied"
        if api.get(flag) or open_discovery.get(flag):
            keys.append(f"{name}_cap")
    if api.get("r_batch_elite") or open_discovery.get("r_batch_elite"):
        keys.append("batch_elite")
    return keys


def _compact_record(record: Dict[str, Any], *, source: str, source_index: int, categories: List[str]) -> Dict[str, Any]:
    api = _api(record)
    return {
        "source": source,
        "source_index": source_index,
        "categories": categories,
        "prompt": str(record.get("prompt") or ""),
        "completion": str(record.get("completion") or record.get("raw_completion") or ""),
        "accuracy": (
            record.get("accuracy")
            if record.get("accuracy") is not None
            else api.get("seed_accuracy_baseline")
            if api.get("seed_accuracy_baseline") is not None
            else api.get("accuracy_baseline")
        ),
        "api_result": api,
    }


def build_fixture(root: Path, output: Path, *, per_category: int) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    counts: Dict[str, int] = {}
    sources = [root / item for item in DEFAULT_SOURCES]

    for source_path in sources:
        if not source_path.exists():
            continue
        for index, record in enumerate(_read_jsonl(source_path)):
            categories = _category_keys(record)
            wanted = [category for category in categories if counts.get(category, 0) < per_category]
            if not wanted:
                continue
            key = (str(record.get("prompt") or ""), str(record.get("completion") or record.get("raw_completion") or ""))
            if key in seen:
                continue
            seen.add(key)
            for category in wanted:
                counts[category] = counts.get(category, 0) + 1
            selected.append(
                _compact_record(
                    record,
                    source=str(source_path.relative_to(root) if source_path.is_relative_to(root) else source_path),
                    source_index=index,
                    categories=categories,
                )
            )

    _write_jsonl(output, selected)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("test/fixtures/reward_replay/current_reward_inputs.jsonl"))
    parser.add_argument("--per-category", type=int, default=4)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output = args.output if args.output.is_absolute() else root / args.output
    rows = build_fixture(root, output, per_category=max(1, int(args.per_category)))
    print(f"wrote={len(rows)} output={output}")


if __name__ == "__main__":
    main()
