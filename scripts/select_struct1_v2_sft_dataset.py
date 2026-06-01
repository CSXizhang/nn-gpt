#!/usr/bin/env python3
"""Select and migrate rl-bb-struct1-v2 SFT records into nn-dataset."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REQUIRED_STAT_FIELDS = {"uid", "transform", "duration", "accuracy"}
PATTERNS = (
    "A_to_Fractal_plus_B",
    "B_to_Fractal_plus_A",
    "Fractal_to_DualBackbone",
    "A_to_Fractal_to_B",
)


@dataclass(frozen=True)
class Candidate:
    name: str
    code_path: Path
    stat_path: Path
    canonical_code: str
    accuracy: float
    pattern: str
    graph_hash: str
    block_signature: str
    backbone_signature: str


def _canonicalize_python(code: str) -> str:
    tree = ast.parse(code)
    return ast.unparse(tree).strip() + "\n"


def _code_name(prefix: str, canonical_code: str) -> str:
    compact = re.sub(r"\s", "", canonical_code)
    return f"{prefix}-{hashlib.md5(compact.encode()).hexdigest()}"


def _load_stat(path: Path) -> tuple[bool, float | None, str | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else [payload]
    for row in rows:
        if isinstance(row, dict) and REQUIRED_STAT_FIELDS.issubset(row):
            return True, float(row["accuracy"]), str(row["transform"])
    return False, None, None


def _stat_paths_for_name(source_root: Path, name: str) -> Iterable[Path]:
    rel = f"img-classification_cifar-10_acc_{name}/1.json"
    yield source_root / "out/nngpt/new_lemur/train" / rel
    yield source_root / "out/nngpt/new_lemur/stat/train" / rel
    yield source_root / "out/nngpt/stat/train" / rel
    yield source_root / "out/nngpt/new_lemur" / "train" / rel


def _discover(args: argparse.Namespace) -> tuple[list[Candidate], dict[str, list[str]]]:
    import ab.gpt.TuneRL as TuneRL
    import ab.gpt.util.SFTUtil as SFTUtil

    source_root = Path(args.source_root).expanduser().resolve()
    skipped: dict[str, list[str]] = defaultdict(list)
    candidates: list[Candidate] = []
    seen_code_hashes: set[str] = set()

    for code_path in sorted((source_root / "out/nngpt/new_lemur/nn").glob(f"{args.source_prefix}-*.py")):
        name = code_path.stem
        stat_path = next((path for path in _stat_paths_for_name(source_root, name) if path.exists()), None)
        if stat_path is None:
            skipped["missing_stat"].append(str(code_path))
            continue
        try:
            has_stat, accuracy, transform = _load_stat(stat_path)
        except Exception as exc:
            skipped["bad_stat_json"].append(f"{stat_path}: {exc}")
            continue
        if not has_stat or accuracy is None:
            skipped["incomplete_stat"].append(str(stat_path))
            continue
        if args.required_transform and transform != args.required_transform:
            skipped["wrong_transform"].append(f"{name}:{transform}")
            continue
        if accuracy < float(args.min_acc):
            skipped["low_accuracy"].append(f"{name}:{accuracy:.4f}")
            continue
        try:
            canonical_code = _canonicalize_python(code_path.read_text(encoding="utf-8"))
        except Exception as exc:
            skipped["ast_parse"].append(f"{code_path}: {exc}")
            continue
        code_hash = hashlib.md5(re.sub(r"\s", "", canonical_code).encode()).hexdigest()
        if code_hash in seen_code_hashes:
            skipped["duplicate_code"].append(str(code_path))
            continue
        seen_code_hashes.add(code_hash)

        try:
            block_code, init_code, forward_code = SFTUtil.parse_nn_code(canonical_code)
            pattern = SFTUtil.extract_target_pattern_from_code(canonical_code) or ""
        except Exception as exc:
            skipped["section_parse"].append(f"{code_path}: {exc}")
            continue
        if not block_code or not init_code or not forward_code or pattern not in PATTERNS:
            skipped["missing_sections_or_pattern"].append(str(code_path))
            continue
        try:
            section_info = TuneRL.describe_reward_code_sections(
                block_code=block_code,
                init_code=init_code,
                forward_code=forward_code,
            )
            graph_info = section_info.get("graph_info")
            graph_hash = str(getattr(graph_info, "graph_hash", "") or "")
            block_signature = str(section_info.get("block_signature") or "incomplete_block")
            backbone_signature = str(section_info.get("backbone_signature") or "unknown_backbone_pair")
        except Exception as exc:
            skipped["graph_parse"].append(f"{code_path}: {exc}")
            continue
        if not graph_hash or not block_signature or not backbone_signature:
            skipped["missing_signature"].append(str(code_path))
            continue
        candidates.append(
            Candidate(
                name=name,
                code_path=code_path,
                stat_path=stat_path,
                canonical_code=canonical_code,
                accuracy=float(accuracy),
                pattern=pattern,
                graph_hash=graph_hash,
                block_signature=block_signature,
                backbone_signature=backbone_signature,
            )
        )
    return candidates, dict(skipped)


def _effective_number(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    return 1.0 / sum((count / total) ** 2 for count in counter.values())


def _top_share(counter: Counter[str]) -> float:
    total = sum(counter.values())
    return (max(counter.values()) / total) if total else 0.0


def _select(candidates: list[Candidate], args: argparse.Namespace) -> list[Candidate]:
    selected: list[Candidate] = []
    graph_counts: Counter[str] = Counter()
    block_counts: Counter[str] = Counter()
    backbone_counts: Counter[str] = Counter()
    per_pattern_target = int(args.per_pattern_target)

    by_pattern: dict[str, list[Candidate]] = {pattern: [] for pattern in PATTERNS}
    for candidate in candidates:
        by_pattern[candidate.pattern].append(candidate)
    for pattern in PATTERNS:
        pool = sorted(by_pattern[pattern], key=lambda item: item.accuracy, reverse=True)
        pattern_selected = 0
        for candidate in pool:
            if pattern_selected >= per_pattern_target:
                break
            if graph_counts[candidate.graph_hash] >= int(args.max_graph_repeats):
                continue
            if block_counts[candidate.block_signature] >= int(args.max_block_repeats):
                continue
            if backbone_counts[candidate.backbone_signature] >= int(args.max_backbone_repeats):
                continue
            selected.append(candidate)
            pattern_selected += 1
            graph_counts[candidate.graph_hash] += 1
            block_counts[candidate.block_signature] += 1
            backbone_counts[candidate.backbone_signature] += 1
    return selected


def _pattern_counts(candidates: Iterable[Candidate]) -> dict[str, int]:
    counts = Counter(candidate.pattern for candidate in candidates)
    return {pattern: int(counts.get(pattern, 0)) for pattern in PATTERNS}


def _pattern_deficits(selected: Iterable[Candidate], per_pattern_target: int) -> dict[str, int]:
    counts = Counter(candidate.pattern for candidate in selected)
    return {
        pattern: max(0, int(per_pattern_target) - int(counts.get(pattern, 0)))
        for pattern in PATTERNS
    }


def _clean_dest_prefix(repo_root: Path, prefix: str) -> None:
    for code_path in (repo_root / "ab/nn/nn").glob(f"{prefix}-*.py"):
        code_path.unlink()
    for stat_dir in (repo_root / "ab/nn/stat/train").glob(f"img-classification_cifar-10_acc_{prefix}-*"):
        if stat_dir.is_dir():
            shutil.rmtree(stat_dir)


def _write_selected(selected: list[Candidate], args: argparse.Namespace) -> list[dict[str, object]]:
    repo_root = Path(args.repo_root).expanduser().resolve()
    dest_prefix = str(args.dest_prefix)
    if args.clean_dest_prefix:
        _clean_dest_prefix(repo_root, dest_prefix)

    written: list[dict[str, object]] = []
    for candidate in selected:
        dest_name = _code_name(dest_prefix, candidate.canonical_code)
        dest_code = repo_root / "ab/nn/nn" / f"{dest_name}.py"
        dest_stat = repo_root / "ab/nn/stat/train" / f"img-classification_cifar-10_acc_{dest_name}" / "1.json"
        dest_code.parent.mkdir(parents=True, exist_ok=True)
        dest_stat.parent.mkdir(parents=True, exist_ok=True)
        dest_code.write_text(candidate.canonical_code, encoding="utf-8")
        shutil.copy2(candidate.stat_path, dest_stat)
        written.append(
            {
                "source_name": candidate.name,
                "dest_name": dest_name,
                "accuracy": candidate.accuracy,
                "pattern": candidate.pattern,
                "graph_hash": candidate.graph_hash,
                "block_signature": candidate.block_signature,
                "backbone_signature": candidate.backbone_signature,
                "code": str(dest_code),
                "stat": str(dest_stat),
            }
        )
    return written


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    pattern_counts = Counter(str(row["pattern"]) for row in rows)
    graph_counts = Counter(str(row["graph_hash"]) for row in rows)
    block_counts = Counter(str(row["block_signature"]) for row in rows)
    backbone_counts = Counter(str(row["backbone_signature"]) for row in rows)
    accuracies = [float(row["accuracy"]) for row in rows]
    return {
        "selected": len(rows),
        "acc_mean": (sum(accuracies) / len(accuracies)) if accuracies else None,
        "acc_max": max(accuracies) if accuracies else None,
        "pattern_counts": dict(pattern_counts),
        "backbone_eff": _effective_number(backbone_counts),
        "module_eff": _effective_number(block_counts),
        "graph_eff": _effective_number(graph_counts),
        "top_backbone_share": _top_share(backbone_counts),
        "top_module_share": _top_share(block_counts),
        "top_graph_share": _top_share(graph_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default="/home/s471802/nn-gpt")
    parser.add_argument("--repo-root", default="/home/s471802/nn-dataset")
    parser.add_argument("--source-prefix", default="rl-bb-struct1-v2")
    parser.add_argument("--dest-prefix", default="rl-bb-struct1-v2")
    parser.add_argument("--target", type=int, default=420)
    parser.add_argument("--per-pattern-target", type=int, default=105)
    parser.add_argument("--min-acc", type=float, default=0.70)
    parser.add_argument("--max-graph-repeats", type=int, default=2)
    parser.add_argument("--max-block-repeats", type=int, default=5)
    parser.add_argument("--max-backbone-repeats", type=int, default=8)
    parser.add_argument("--required-transform", default="norm_128_flip")
    parser.add_argument("--clean-dest-prefix", action="store_true")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    candidates, skipped = _discover(args)
    selected = _select(candidates, args)
    if len(selected) < int(args.target):
        report = {
            "status": "insufficient_selected",
            "candidate_count": len(candidates),
            "selected_count": len(selected),
            "target": int(args.target),
            "eligible_pattern_counts": _pattern_counts(candidates),
            "selected_pattern_counts": _pattern_counts(selected),
            "pattern_deficits": _pattern_deficits(selected, int(args.per_pattern_target)),
            "skipped": {key: len(value) for key, value in skipped.items()},
            "selected_preview": [candidate.name for candidate in selected[:20]],
        }
        Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise RuntimeError(f"Selected {len(selected)} candidates, need {args.target}; generate an append batch.")

    selected = selected[: int(args.target)]
    written = _write_selected(selected, args)
    report = {
        "status": "ok",
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "written_count": len(written),
        "target": int(args.target),
        "source_prefix": args.source_prefix,
        "dest_prefix": args.dest_prefix,
        "filters": {
            "min_acc": float(args.min_acc),
            "max_graph_repeats": int(args.max_graph_repeats),
            "max_block_repeats": int(args.max_block_repeats),
            "max_backbone_repeats": int(args.max_backbone_repeats),
            "required_transform": args.required_transform,
        },
        "summary": _summary(written),
        "skipped": {key: len(value) for key, value in skipped.items()},
        "written": written,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "candidate_count", "selected_count", "written_count")}, sort_keys=True))
    print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
