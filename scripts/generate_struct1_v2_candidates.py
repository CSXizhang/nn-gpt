#!/usr/bin/env python3
"""Generate balanced four-pattern brute candidates into a synth_nn directory."""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path


FOUR_PATTERN_NAMES = (
    "A_to_Fractal_plus_B",
    "B_to_Fractal_plus_A",
    "Fractal_to_DualBackbone",
    "A_to_Fractal_to_B",
)


def _parse_pattern_counts(raw: str | None, budget: int, allow_partial: bool = False) -> dict[str, int]:
    if raw:
        counts: dict[str, int] = {}
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            name, value = item.split(":", 1)
            name = name.strip()
            if name not in FOUR_PATTERN_NAMES:
                raise ValueError(f"Unsupported pattern {name!r}; expected one of {FOUR_PATTERN_NAMES}")
            counts[name] = int(value)
        missing = [name for name in FOUR_PATTERN_NAMES if name not in counts]
        if missing and not allow_partial:
            raise ValueError(f"Missing pattern counts for: {', '.join(missing)}")
        for name in missing:
            counts[name] = 0
        return counts

    per_pattern, remainder = divmod(int(budget), len(FOUR_PATTERN_NAMES))
    counts = {name: per_pattern for name in FOUR_PATTERN_NAMES}
    for name in FOUR_PATTERN_NAMES[:remainder]:
        counts[name] += 1
    return counts


def _next_model_index(output_dir: Path) -> int:
    max_index = -1
    for path in output_dir.glob("B*"):
        if not path.is_dir():
            continue
        match = re.fullmatch(r"B(\d+)", path.name)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def generate(args: argparse.Namespace) -> None:
    from ab.gpt.brute.fract.backbone import NNAlterBN
    from ab.gpt.util.Const import fract_dir, new_nn_file

    random.seed(int(args.seed))
    output_dir = Path(args.output_synth_dir).expanduser().resolve()
    if output_dir.exists() and not args.append:
        for child in output_dir.glob("B*"):
            if child.is_dir():
                for file_path in child.iterdir():
                    file_path.unlink()
                child.rmdir()
    output_dir.mkdir(parents=True, exist_ok=True)

    pattern_counts = _parse_pattern_counts(
        args.pattern_counts,
        int(args.budget),
        allow_partial=bool(args.allow_partial_pattern_counts),
    )
    patterns = NNAlterBN.DIVERSE_FORWARD_PATTERNS
    helper_code = NNAlterBN.DIVERSE_FORWARD_HELPER
    template = (Path(fract_dir) / "backbone" / "FractalFusion_template.py").read_text(encoding="utf-8")

    available_backbones = NNAlterBN.filter_backbones_by_size(max_params_millions=float(args.max_backbone_params_m))
    if len(available_backbones) < 2:
        raise RuntimeError(f"Need at least two available backbones, found {len(available_backbones)}")
    for backbone_name in available_backbones:
        NNAlterBN.probe_model_output_channels(backbone_name)

    schedule: list[str] = []
    for pattern_name, count in pattern_counts.items():
        schedule.extend([pattern_name] * int(count))
    random.shuffle(schedule)

    start_index = _next_model_index(output_dir) if args.append else 0
    manifest_path = output_dir.parent / "candidate_manifest.jsonl"
    if manifest_path.exists() and not args.append:
        manifest_path.unlink()

    run_config = {
        "phase": "struct1_v2_brute_generate",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seed": int(args.seed),
        "budget": len(schedule),
        "pattern_counts": pattern_counts,
        "patterns": FOUR_PATTERN_NAMES,
        "output_synth_dir": str(output_dir),
        "append": bool(args.append),
        "max_backbone_params_m": float(args.max_backbone_params_m),
        "available_backbones": available_backbones,
    }
    _write_json(output_dir.parent / "generation_config.json", run_config)

    for offset, pattern_name in enumerate(schedule):
        model_index = start_index + offset
        model_dir = output_dir / f"B{model_index:04d}"
        model_dir.mkdir(parents=True, exist_ok=False)

        block_code = NNAlterBN.generate_conv_block()
        bb_a, bb_b = random.sample(available_backbones, 2)
        n_units = random.randint(1, 2)
        cols = random.randint(2, 3)
        forward_code = helper_code + patterns[pattern_name]

        code = (
            template.replace("$$", block_code)
            .replace("?FORWARD", forward_code)
            .replace("?PATTERN", pattern_name)
            .replace("?N", str(n_units))
            .replace("?COLS", str(cols))
            .replace("?bb_a", f'"{bb_a}"')
            .replace("?bb_b", f'"{bb_b}"')
        )
        (model_dir / new_nn_file).write_text(code, encoding="utf-8")
        _append_jsonl(
            manifest_path,
            {
                "model_id": model_dir.name,
                "pattern": pattern_name,
                "backbone_a": bb_a,
                "backbone_b": bb_b,
                "n_units": n_units,
                "cols": cols,
                "code_path": str(model_dir / new_nn_file),
            },
        )
        print(f"[generate] {model_dir.name} pattern={pattern_name} backbones={bb_a},{bb_b}", flush=True)

    print(f"Wrote {len(schedule)} candidates to {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-synth-dir", required=True)
    parser.add_argument("--budget", type=int, default=520)
    parser.add_argument("--pattern-counts", default="")
    parser.add_argument("--allow-partial-pattern-counts", action="store_true")
    parser.add_argument("--seed", type=int, default=600)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--max-backbone-params-m", type=float, default=30.0)
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
