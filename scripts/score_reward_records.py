#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if os.getenv("NNGPT_REWARD_REPLAY_LIGHTWEIGHT", "").strip().lower() in {"1", "true", "yes", "on"}:
    from scripts import reward_replay_stubs

    reward_replay_stubs.install()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _set_default_env(args: argparse.Namespace) -> None:
    os.environ.setdefault("NNGPT_RL_REWARD_VARIANT", args.reward_variant)
    os.environ.setdefault("NNGPT_RL_RESUME_STAGE", "stage2_formal_explore")
    os.environ.setdefault("NNGPT_RL_FORMAL_REWARD_EPOCHS", "1,5,10")
    os.environ.setdefault(
        "NNGPT_SFT_BASE_MODEL_ID",
        "/home/s471802/nn-gpt/out/llm/deepseek-ai/deepseek-coder-6.7b-instruct",
    )
    os.environ.setdefault("NNGPT_SFT_TOKENIZER_ID", "deepseek-ai/deepseek-coder-6.7b-instruct")
    os.environ.setdefault("NNGPT_SFT_RL_NN_PREFIXES", "rl-bb-struct1")
    os.environ.setdefault("NNGPT_SFT_RL_PROMPT_MODE", "sft_aligned")


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-score existing generation records with precomputed eval results.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--reward-variant", default="full")
    args = parser.parse_args()

    _set_default_env(args)

    from ab.gpt import TuneRL, TuneRLSft

    TuneRLSft.configure_sft_runtime()
    TuneRL.apply_resume_stage_override(os.environ["NNGPT_RL_RESUME_STAGE"], log_prefix="[score_reward_records]")

    rows = _read_jsonl(args.input)
    if args.limit is not None:
        rows = rows[: int(args.limit)]
    entries = TuneRL.reward_entries_from_records(rows)
    group_context = TuneRL.default_reward_replay_group_context()
    scored = TuneRL.score_reward_entries(
        entries,
        group_context=group_context,
        archive_snapshot_counts={},
    )
    output_rows = []
    for item in scored:
        result = item["result"]
        output_rows.append(
            {
                "prompt": item.get("prompt", ""),
                "completion": item.get("completion", ""),
                "reward": float(result.get("reward", item.get("score", -2.0))),
                "api_result": result,
            }
        )
    _write_jsonl(args.output, output_rows)
    print(f"rescored={len(output_rows)} output={args.output}")


if __name__ == "__main__":
    main()
