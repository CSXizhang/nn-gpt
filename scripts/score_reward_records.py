#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List


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


def _completion(record: Dict[str, Any]) -> str:
    return str(record.get("completion") or record.get("raw_completion") or "")


def _api(record: Dict[str, Any]) -> Dict[str, Any]:
    value = record.get("api_result")
    return value if isinstance(value, dict) else {}


def _seed_accuracy(record: Dict[str, Any]) -> float:
    api = _api(record)
    for source in (
        record,
        api,
        record.get("candidate") if isinstance(record.get("candidate"), dict) else {},
    ):
        value = source.get("seed_accuracy_baseline")
        if value is not None:
            return float(value)
    return 0.10


def _set_default_env(args: argparse.Namespace) -> None:
    os.environ.setdefault("NNGPT_RL_REWARD_VARIANT", args.reward_variant)
    os.environ.setdefault("NNGPT_RL_RESUME_STAGE", "stage2_formal_explore")
    os.environ.setdefault("NNGPT_RL_FORMAL_REWARD_EPOCHS", "1")
    os.environ.setdefault(
        "NNGPT_SFT_BASE_MODEL_ID",
        "/home/s471802/nn-gpt/out/llm/deepseek-ai/deepseek-coder-6.7b-instruct",
    )
    os.environ.setdefault("NNGPT_SFT_TOKENIZER_ID", "deepseek-ai/deepseek-coder-6.7b-instruct")
    os.environ.setdefault("NNGPT_SFT_RL_NN_PREFIXES", "rl-bb-struct1")
    os.environ.setdefault("NNGPT_SFT_RL_PROMPT_MODE", "sft_aligned")


def _entry_from_record(record: Dict[str, Any], index: int) -> Dict[str, Any]:
    from ab.gpt import TuneRL

    completion = _completion(record)
    block_code, init_code, forward_code = TuneRL.extract_completion_blocks(completion)
    backbone_model_names = TuneRL._extract_backbone_model_names(init_code)
    graph_info = None
    if block_code and init_code and forward_code and "self.pattern" not in forward_code:
        graph_info = TuneRL.extract_graph_info(
            init_code,
            forward_code,
            legacy_patterns=TuneRL.SFTUtil.legacy_patterns,
        )
    backbone_signature = TuneRL.build_backbone_signature(backbone_model_names)
    cnn_signature = (
        str(getattr(graph_info, "cnn_signature", "") or "")
        if graph_info is not None
        else "incomplete_cnn"
    )
    return {
        "rank": 0,
        "local_index": index,
        "global_index": index,
        "completion": completion,
        "prompt": str(record.get("prompt") or ""),
        "graph_info": graph_info,
        "backbone_model_names": backbone_model_names,
        "backbone_signature": backbone_signature,
        "cnn_signature": cnn_signature,
        "prompt_goal_tags": [],
        "goal_key": TuneRL.primary_goal_key([]),
        "seed_accuracy_baseline": _seed_accuracy(record),
        "precomputed_eval_result": dict(_api(record)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-score existing generation records with precomputed eval results.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--reward-variant", default="full")
    args = parser.parse_args()

    _set_default_env(args)

    from ab.gpt import TuneRL, TuneRLSft

    TuneRLSft.patch_sft_runtime()
    TuneRL.apply_resume_stage_override(os.environ["NNGPT_RL_RESUME_STAGE"], log_prefix="[score_reward_records]")

    rows = _read_jsonl(args.input)
    if args.limit is not None:
        rows = rows[: int(args.limit)]
    entries = [_entry_from_record(record, index) for index, record in enumerate(rows)]
    group_context = {
        "group_baseline_train_acc": None,
        "group_baseline_reward_target_acc": None,
        "group_baseline_test_acc": None,
        "best_closed_group_mean_train_acc": None,
        "best_closed_group_mean_reward_target_acc": None,
        "best_closed_group_mean_test_acc": None,
        "best_closed_group_id": None,
        "dominant_family_hash": None,
        "dominant_family_share": 0.0,
        "dominant_descriptor_key": None,
        "dominant_descriptor_share": 0.0,
        "dominant_backbone_signature": None,
        "dominant_backbone_share": 0.0,
        "dominant_cnn_signature": None,
        "dominant_cnn_share": 0.0,
        "dominant_backbone_cnn_pair": None,
        "dominant_backbone_cnn_share": 0.0,
        "reward_batch_index": 0,
        "reward_group_id": 0,
        "group_warmup": False,
        "current_stage_name": TuneRL.current_stage_name,
        "current_stage_index": TuneRL.RL_STAGE_TO_INDEX.get(TuneRL.current_stage_name, 0),
        "generation_total": 0,
        "stage_group_count": 0,
        "recovery_active": False,
    }
    scored = TuneRL._score_reward_entries(
        entries,
        group_context=group_context,
        archive_snapshot_family_counts={},
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
