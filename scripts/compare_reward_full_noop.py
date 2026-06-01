#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple


NUMERIC_PATHS = (
    ("record", ("reward",)),
    ("api", ("reward",)),
    ("api", ("r_primary",)),
    ("api", ("r_tiebreak",)),
    ("api", ("r_trainset_novelty",)),
    ("api", ("r_structure_group",)),
    ("api", ("r_structure_archive",)),
    ("api", ("r_descriptor_diversity",)),
    ("api", ("r_cnn_diversity",)),
    ("api", ("r_block_diversity",)),
    ("api", ("r_prev_backbone_group",)),
    ("api", ("r_best_backbone_group",)),
    ("api", ("backbone_reward_target_gain",)),
    ("api", ("archive_snapshot_backbone_freq",)),
    ("api", ("archive_snapshot_cnn_freq",)),
    ("api", ("archive_snapshot_block_freq",)),
    ("api", ("archive_snapshot_backbone_cnn_freq",)),
    ("api", ("archive_snapshot_backbone_block_freq",)),
    ("api", ("batch_same_backbone_cnn_count",)),
    ("api", ("batch_same_backbone_block_count",)),
    ("api", ("global_descriptor_archive_reward",)),
    ("api", ("global_cnn_archive_reward",)),
    ("api", ("block_archive_reward",)),
    ("api", ("r_repeat_family",)),
    ("api", ("r_no_progress_penalty",)),
    ("api", ("reward_target_value",)),
    ("api", ("formal_reward_target_value",)),
    ("open_discovery", ("r_primary",)),
    ("open_discovery", ("r_tiebreak",)),
    ("open_discovery", ("r_trainset_novelty",)),
    ("open_discovery", ("r_structure_group",)),
    ("open_discovery", ("r_structure_archive",)),
    ("open_discovery", ("r_descriptor_diversity",)),
    ("open_discovery", ("r_cnn_diversity",)),
    ("open_discovery", ("r_block_diversity",)),
    ("open_discovery", ("r_prev_backbone_group",)),
    ("open_discovery", ("r_best_backbone_group",)),
    ("open_discovery", ("backbone_reward_target_gain",)),
    ("open_discovery", ("archive_snapshot_backbone_freq",)),
    ("open_discovery", ("archive_snapshot_cnn_freq",)),
    ("open_discovery", ("archive_snapshot_block_freq",)),
    ("open_discovery", ("archive_snapshot_backbone_cnn_freq",)),
    ("open_discovery", ("archive_snapshot_backbone_block_freq",)),
    ("open_discovery", ("batch_same_backbone_cnn_count",)),
    ("open_discovery", ("batch_same_backbone_block_count",)),
    ("open_discovery", ("global_descriptor_archive_reward",)),
    ("open_discovery", ("global_cnn_archive_reward",)),
    ("open_discovery", ("block_archive_reward",)),
    ("open_discovery", ("r_repeat_family",)),
    ("open_discovery", ("r_no_progress_penalty",)),
    ("open_discovery", ("reward_target_value",)),
    ("open_discovery", ("formal_reward_target_value",)),
)

STRING_PATHS = (
    ("api", ("backbone_signature",)),
    ("api", ("cnn_signature",)),
    ("api", ("block_signature",)),
    ("api", ("backbone_block_pair_key",)),
    ("api", ("reward_variant",)),
    ("api", ("target_structure_match",)),
    ("open_discovery", ("backbone_signature",)),
    ("open_discovery", ("cnn_signature",)),
    ("open_discovery", ("block_signature",)),
    ("open_discovery", ("backbone_block_pair_key",)),
    ("open_discovery", ("reward_variant",)),
)

LIST_PATHS = (
    ("api", ("strong_repeat_penalty_reasons",)),
    ("api", ("target_structure_mismatch_reasons",)),
    ("open_discovery", ("strong_repeat_penalty_reasons",)),
)

BOOL_PATHS = (
    ("positive_reward", ()),
    ("api", ("descriptor_reward_cap_applied",)),
    ("api", ("cnn_reward_cap_applied",)),
    ("api", ("block_reward_cap_applied",)),
    ("api", ("signature_reward_cap_applied",)),
    ("api", ("xml_incomplete_length_cap",)),
    ("api", ("stage1_fixed_failure_reward",)),
    ("open_discovery", ("descriptor_reward_cap_applied",)),
    ("open_discovery", ("cnn_reward_cap_applied",)),
    ("open_discovery", ("block_reward_cap_applied",)),
    ("open_discovery", ("signature_reward_cap_applied",)),
)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _api(record: Dict[str, Any]) -> Dict[str, Any]:
    value = record.get("api_result")
    return value if isinstance(value, dict) else {}


def _open_discovery(record: Dict[str, Any]) -> Dict[str, Any]:
    value = _api(record).get("open_discovery")
    return value if isinstance(value, dict) else {}


def _container(record: Dict[str, Any], label: str) -> Dict[str, Any]:
    if label == "record":
        return record
    if label == "api":
        return _api(record)
    if label == "open_discovery":
        return _open_discovery(record)
    raise ValueError(label)


def _get(record: Dict[str, Any], label: str, path: Tuple[str, ...]) -> Tuple[bool, Any]:
    if label == "positive_reward":
        return True, _as_float(record.get("reward", _api(record).get("reward"))) > 0.0
    current: Any = _container(record, label)
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return False, None
        current = current[key]
    return True, current


def _as_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return math.nan
    return parsed


def _same_float(left: Any, right: Any, tol: float) -> bool:
    left_float = _as_float(left)
    right_float = _as_float(right)
    if math.isnan(left_float) and math.isnan(right_float):
        return True
    if math.isnan(left_float) or math.isnan(right_float):
        return False
    return abs(left_float - right_float) <= tol


def _same_json_value(left: Any, right: Any) -> bool:
    return json.dumps(left, sort_keys=True, ensure_ascii=False) == json.dumps(right, sort_keys=True, ensure_ascii=False)


def _compare_pair(index: int, left: Dict[str, Any], right: Dict[str, Any], tol: float) -> List[str]:
    mismatches: List[str] = []
    for label, path in NUMERIC_PATHS:
        left_has, left_value = _get(left, label, path)
        right_has, right_value = _get(right, label, path)
        if not left_has and not right_has:
            continue
        if left_has != right_has:
            mismatches.append(f"#{index} {label}.{'.'.join(path)} presence {left_has}!={right_has}")
            continue
        if not _same_float(left_value, right_value, tol):
            mismatches.append(
                f"#{index} {label}.{'.'.join(path)} {left_value!r}!={right_value!r}"
            )
    for label, path in BOOL_PATHS:
        left_has, left_value = _get(left, label, path)
        right_has, right_value = _get(right, label, path)
        if not left_has and not right_has:
            continue
        if left_has != right_has:
            mismatches.append(f"#{index} {label}.{'.'.join(path)} presence {left_has}!={right_has}")
            continue
        if bool(left_value) != bool(right_value):
            mismatches.append(
                f"#{index} {label}.{'.'.join(path)} {bool(left_value)!r}!={bool(right_value)!r}"
            )
    for label, path in STRING_PATHS:
        left_has, left_value = _get(left, label, path)
        right_has, right_value = _get(right, label, path)
        if not left_has and not right_has:
            continue
        if left_has != right_has:
            mismatches.append(f"#{index} {label}.{'.'.join(path)} presence {left_has}!={right_has}")
            continue
        if str(left_value) != str(right_value):
            mismatches.append(
                f"#{index} {label}.{'.'.join(path)} {left_value!r}!={right_value!r}"
            )
    for label, path in LIST_PATHS:
        left_has, left_value = _get(left, label, path)
        right_has, right_value = _get(right, label, path)
        if not left_has and not right_has:
            continue
        if left_has != right_has:
            mismatches.append(f"#{index} {label}.{'.'.join(path)} presence {left_has}!={right_has}")
            continue
        if not _same_json_value(left_value, right_value):
            mismatches.append(
                f"#{index} {label}.{'.'.join(path)} {left_value!r}!={right_value!r}"
            )
    return mismatches


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare baseline 821f reward logs with ablation full no-op logs."
    )
    parser.add_argument("baseline", type=Path)
    parser.add_argument("full_variant", type=Path)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--max-mismatches", type=int, default=40)
    args = parser.parse_args()

    baseline_rows = _read_jsonl(args.baseline)
    full_rows = _read_jsonl(args.full_variant)
    mismatches: List[str] = []
    if len(baseline_rows) != len(full_rows):
        mismatches.append(f"row_count {len(baseline_rows)}!={len(full_rows)}")
    for index, (left, right) in enumerate(zip(baseline_rows, full_rows), start=1):
        mismatches.extend(_compare_pair(index, left, right, args.tol))
        if len(mismatches) >= args.max_mismatches:
            break

    if mismatches:
        print(f"NO-OP CHECK FAILED: compared={min(len(baseline_rows), len(full_rows))} mismatches={len(mismatches)}")
        for item in mismatches[: args.max_mismatches]:
            print(item)
        raise SystemExit(1)
    print(f"NO-OP CHECK PASSED: compared={len(baseline_rows)} tol={args.tol:g}")


if __name__ == "__main__":
    main()
