#!/usr/bin/env python3
"""Generate, evaluate, and summarize baseline architecture candidates.

This runner intentionally separates LLM generation from formal evaluation so
single-rank generation does not reserve idle evaluation GPUs.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import random
import subprocess
import sys
import textwrap
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    try:
        import torch

        if isinstance(value, torch.dtype):
            return str(value)
    except Exception:
        pass
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _repo_commit() -> dict[str, str]:
    def run_git(*args: str) -> str:
        try:
            return subprocess.check_output(["git", *args], text=True).strip()
        except Exception:
            return ""

    return {
        "commit": run_git("rev-parse", "HEAD"),
        "commit_short": run_git("rev-parse", "--short", "HEAD"),
        "commit_subject": run_git("log", "-1", "--pretty=%s"),
        "branch": run_git("branch", "--show-current"),
    }


def _candidate_id(setting: str, index: int) -> str:
    return f"{setting}-{index:04d}"


def _truncate_completion(completion: str) -> str:
    end_tag = "</forward>"
    if end_tag in completion:
        completion = completion.split(end_tag, 1)[0] + end_tag
    return completion.strip()


def _configure_sft_env(args: argparse.Namespace) -> None:
    if getattr(args, "base_model_id", None):
        os.environ["NNGPT_SFT_BASE_MODEL_ID"] = str(args.base_model_id)
    if getattr(args, "tokenizer_id", None):
        os.environ["NNGPT_SFT_TOKENIZER_ID"] = str(args.tokenizer_id)
    if getattr(args, "nn_prefixes", None):
        os.environ["NNGPT_SFT_RL_NN_PREFIXES"] = str(args.nn_prefixes)
    if getattr(args, "prompt_mode", None):
        os.environ["NNGPT_SFT_RL_PROMPT_MODE"] = str(args.prompt_mode)
    if getattr(args, "feedback_char_budget", None) is not None:
        os.environ["NNGPT_SFT_FEEDBACK_CHAR_BUDGET"] = str(args.feedback_char_budget)
    os.environ.setdefault("NNGPT_RL_FORMAL_REWARD_EPOCHS", "1,5,10")


def _configure_eval_runtime() -> None:
    import ab.gpt.TuneRL as TuneRL
    import ab.gpt.TuneRLSft as TuneRLSft
    from ab.gpt.rl_pipeline.completion import clear_extraction_meta_cache, extract_completion_blocks_strict

    TuneRL.current_stage_name = TuneRL.STAGE2_FORMAL_EXPLORE
    TuneRL.extract_completion_blocks = extract_completion_blocks_strict
    TuneRL.clear_extraction_meta_cache = clear_extraction_meta_cache
    TuneRL.evaluate_code_and_reward = TuneRLSft.evaluate_code_and_reward_cifar
    setattr(TuneRL.evaluate_code_and_reward, "_nngpt_eval_cfg_builder", TuneRLSft.build_sft_reward_eval_cfg)


def _load_prompt_dataset(tokenizer: Any, budget: int):
    import ab.gpt.TuneRLSft as TuneRLSft
    import ab.gpt.util.Reward as RewardUtil

    os.environ["NNGPT_SFT_DATASET_LIMIT"] = str(max(1, int(budget)))
    dataset = TuneRLSft.load_rl_dataset_sft(tokenizer)
    RewardUtil.shutdown_eval_worker()
    if len(dataset) <= 0:
        raise RuntimeError("No prompts available for baseline generation")
    return dataset


def _load_generation_model(args: argparse.Namespace):
    import torch
    import ab.gpt.TuneRL as TuneRL
    import ab.gpt.TuneRLSft as TuneRLSft
    import ab.gpt.rl_pipeline.trainer_runtime as TrainerRuntime

    model_source, tokenizer_source, source_mode = TuneRLSft.resolve_sft_model_sources()
    tokenizer = TrainerRuntime.load_tokenizer(tokenizer_source)
    precision = TuneRL.best_mixed_precision()
    train_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = TrainerRuntime.load_quantized_causal_lm(
        model_source=model_source,
        precision=precision,
        train_device=train_device,
        use_deepspeed=False,
    )

    adapter_path = str(args.adapter_path or "").strip()
    adapter_mode = str(args.adapter_mode or "trainable").strip().lower()
    if adapter_path:
        if adapter_mode in {"merge", "merged", "merge_and_unload"}:
            model = TrainerRuntime.maybe_merge_initial_adapter(
                model,
                enabled=True,
                adapter_path=adapter_path,
                label="baseline SFT",
                load_message=f"Loading baseline SFT adapter from {adapter_path} for merge...",
            )
        else:
            model = TrainerRuntime.load_trainable_initial_adapter(
                model,
                enabled=True,
                adapter_path=adapter_path,
                label="baseline SFT",
                adapter_dtype=None,
                load_message=f"Loading baseline SFT adapter from {adapter_path}...",
            )

    TuneRL.align_generation_head_dtype(model, precision["torch_dtype"])
    model.eval()
    return model, tokenizer, {
        "model_source": model_source,
        "tokenizer_source": tokenizer_source,
        "source_mode": source_mode,
        "precision": precision,
        "adapter_path": adapter_path,
        "adapter_mode": adapter_mode if adapter_path else "",
    }


def command_gen_only(args: argparse.Namespace) -> None:
    import torch
    import ab.gpt.TuneRL as TuneRL

    _configure_sft_env(args)
    _configure_eval_runtime()
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = output_dir / "candidates.jsonl"
    if candidates_path.exists() and not args.append:
        candidates_path.unlink()

    model, tokenizer, model_config = _load_generation_model(args)
    dataset = _load_prompt_dataset(tokenizer, int(args.budget))

    generation_config = {
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "top_k": int(args.top_k),
        "max_new_tokens": int(args.max_new_tokens),
        "do_sample": True,
        "stop_strings": ["</forward>"],
    }
    run_config = {
        "phase": "gen_only",
        "setting": args.setting,
        "source_run": args.source_run,
        "seed": int(args.seed),
        "budget": int(args.budget),
        "source_format": "xml_completion",
        "prompt_config": {
            "nn_prefixes": args.nn_prefixes,
            "prompt_mode": args.prompt_mode,
            "feedback_char_budget": int(args.feedback_char_budget),
            **generation_config,
        },
        "adapter_config": {
            "adapter_path": str(args.adapter_path or ""),
            "adapter_mode": str(args.adapter_mode or ""),
        },
        "model_config": model_config,
        "git": _repo_commit(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _write_json(output_dir / "run_config.json", run_config)

    device = next(model.parameters()).device
    for index in range(int(args.budget)):
        item = dataset[index % len(dataset)]
        prompt = str(item["prompt"])
        raw_completion = ""
        candidate_code = ""
        generation_error = None
        try:
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=int(args.max_new_tokens),
                    do_sample=True,
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    top_k=int(args.top_k),
                    stop_strings=["</forward>"],
                    tokenizer=tokenizer,
                    eos_token_id=getattr(tokenizer, "eos_token_id", None),
                    pad_token_id=getattr(tokenizer, "pad_token_id", getattr(tokenizer, "eos_token_id", None)),
                )
            prompt_len = int(inputs["input_ids"].shape[-1])
            raw_completion = tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True)
            raw_completion = _truncate_completion(raw_completion)
            candidate_code = TuneRL.reconstruct_code(raw_completion)
        except Exception as exc:
            generation_error = f"{type(exc).__name__}: {exc}"

        record = {
            "candidate_id": _candidate_id(args.setting, index),
            "setting": args.setting,
            "source_run": args.source_run,
            "prompt_config": dict(run_config["prompt_config"]),
            "adapter_config": dict(run_config["adapter_config"]),
            "source_format": "xml_completion",
            "seed_accuracy_baseline": item.get("accuracy"),
            "target_pattern": item.get("target_pattern"),
            "prompt": prompt,
            "candidate_code": candidate_code,
            "raw_completion": raw_completion,
            "generation_error": generation_error,
        }
        _append_jsonl(candidates_path, record)
        print(f"[gen_only] {record['candidate_id']} error={generation_error is not None}")

    print(f"Wrote candidates: {candidates_path}")


def _source_segment(tree_source: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(tree_source, node)
    if not segment:
        raise ValueError(f"Could not extract source for AST node {type(node).__name__}")
    return textwrap.dedent(segment).strip()


def _completion_from_full_code(code: str) -> str:
    tree = ast.parse(code)
    block_code = ""
    init_code = ""
    forward_code = ""

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "drop_conv3x3_block":
            block_code = _source_segment(code, node)
        if isinstance(node, ast.ClassDef) and node.name == "Net":
            for member in node.body:
                if isinstance(member, ast.FunctionDef) and member.name == "__init__":
                    init_code = _source_segment(code, member)
                elif isinstance(member, ast.FunctionDef) and member.name == "forward":
                    forward_code = _source_segment(code, member)

    return "\n".join(
        [
            "<block>",
            block_code,
            "</block>",
            "<init>",
            init_code,
            "</init>",
            "<forward>",
            forward_code,
            "</forward>",
        ]
    )


def _full_code_sections(code: str) -> tuple[str, str, str]:
    tree = ast.parse(code)
    block_code = ""
    init_code = ""
    forward_code = ""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "drop_conv3x3_block":
            block_code = _source_segment(code, node)
        elif isinstance(node, ast.ClassDef) and node.name == "Net":
            for member in node.body:
                if isinstance(member, ast.FunctionDef) and member.name == "__init__":
                    init_code = _source_segment(code, member)
                elif isinstance(member, ast.FunctionDef) and member.name == "forward":
                    forward_code = _source_segment(code, member)
    return block_code, init_code, forward_code


def _brute_patterns(pattern_set: str) -> tuple[dict[str, str], str]:
    from ab.gpt.brute.fract.backbone import NNAlterBN

    pattern_set = str(pattern_set or "all").lower()
    if pattern_set == "union":
        pattern_set = "all"
    if pattern_set == "basic":
        return dict(NNAlterBN.FORWARD_PATTERNS), ""
    if pattern_set == "diverse":
        return dict(NNAlterBN.DIVERSE_FORWARD_PATTERNS), NNAlterBN.DIVERSE_FORWARD_HELPER
    if pattern_set == "all":
        patterns = dict(NNAlterBN.FORWARD_PATTERNS)
        patterns.update(NNAlterBN.DIVERSE_FORWARD_PATTERNS)
        return patterns, NNAlterBN.DIVERSE_FORWARD_HELPER
    raise ValueError("--pattern-set must be one of: basic, diverse, all, union")


def _balanced_pattern_items(pattern_items: list[tuple[str, str]], budget: int) -> list[tuple[str, str]]:
    if budget <= 0 or not pattern_items:
        return []
    repeated: list[tuple[str, str]] = []
    while len(repeated) < budget:
        repeated.extend(pattern_items)
    selected = repeated[:budget]
    random.shuffle(selected)
    return selected


def command_brute_gen_only(args: argparse.Namespace) -> None:
    from ab.gpt.brute.fract.backbone import NNAlterBN
    from ab.gpt.util.Const import fract_dir

    random.seed(int(args.seed))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = output_dir / "candidates.jsonl"
    if candidates_path.exists() and not args.append:
        candidates_path.unlink()

    patterns, helper_code = _brute_patterns(args.pattern_set)
    if not patterns:
        raise RuntimeError("No brute forward patterns selected")

    # Keep brute generation CPU-only. The historical NNAlterBN filter probes
    # candidate backbones on CUDA, which defeats the two-stage baseline plan.
    import ab.gpt.util.SFTUtil as SFTUtil

    available_backbones = list(SFTUtil.available_backbones)
    if len(available_backbones) < 2:
        raise RuntimeError(f"Need at least two brute backbones, found {len(available_backbones)}")

    template = (Path(fract_dir) / "backbone" / "FractalFusion_template.py").read_text(encoding="utf-8")
    pattern_items = list(patterns.items())
    run_config = {
        "phase": "brute_gen_only",
        "setting": args.setting,
        "source_run": args.source_run,
        "seed": int(args.seed),
        "budget": int(args.budget),
        "source_format": "full_code",
        "brute_config": {
            "pattern_set": args.pattern_set,
            "patterns": [name for name, _ in pattern_items],
            "pattern_schedule": "balanced_round_robin_shuffled",
            "available_backbones": available_backbones,
            "backbone_source": "SFTUtil.available_backbones",
        },
        "git": _repo_commit(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _write_json(output_dir / "run_config.json", run_config)

    pattern_schedule = _balanced_pattern_items(pattern_items, int(args.budget))
    for index, (pattern_name, forward_code) in enumerate(pattern_schedule):
        block_code = NNAlterBN.generate_conv_block()
        bb_a, bb_b = random.sample(available_backbones, 2)
        n_units = random.randint(1, 2)
        cols = random.randint(2, 3)
        final_forward = (helper_code if pattern_name in NNAlterBN.DIVERSE_FORWARD_PATTERNS else "") + forward_code
        candidate_code = (
            template.replace("$$", block_code)
            .replace("?FORWARD", final_forward)
            .replace("?PATTERN", pattern_name)
            .replace("?N", str(n_units))
            .replace("?COLS", str(cols))
            .replace("?bb_a", f'"{bb_a}"')
            .replace("?bb_b", f'"{bb_b}"')
        )
        generation_error = None
        raw_completion = ""
        try:
            raw_completion = _completion_from_full_code(candidate_code)
        except Exception as exc:
            generation_error = f"{type(exc).__name__}: {exc}"

        record = {
            "candidate_id": _candidate_id(args.setting, index),
            "setting": args.setting,
            "source_run": args.source_run,
            "prompt_config": {
                "generator": "brute/fract/backbone/NNAlterBN.py",
                "pattern_set": args.pattern_set,
                "pattern_name": pattern_name,
                "n_units": n_units,
                "cols": cols,
                "backbones": [bb_a, bb_b],
            },
            "adapter_config": {},
            "source_format": "full_code",
            "seed_accuracy_baseline": float(args.seed_accuracy_baseline),
            "candidate_code": candidate_code,
            "raw_completion": raw_completion,
            "generation_error": generation_error,
        }
        _append_jsonl(candidates_path, record)
        print(f"[brute_gen_only] {record['candidate_id']} pattern={pattern_name} error={generation_error is not None}")

    print(f"Wrote candidates: {candidates_path}")


def command_collect_synth(args: argparse.Namespace) -> None:
    random.seed(int(args.seed))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = output_dir / "candidates.jsonl"
    if candidates_path.exists() and not args.append:
        candidates_path.unlink()

    synth_dir = Path(args.synth_dir)
    dirs = [path for path in synth_dir.glob("B*") if path.is_dir() and path.name[1:].isdigit()]
    dirs.sort(key=lambda path: int(path.name[1:]))
    if args.limit is not None:
        dirs = dirs[: int(args.limit)]
    if not dirs:
        raise RuntimeError(f"No B* synth dirs found under {synth_dir}")

    run_config = {
        "phase": "collect_synth",
        "setting": args.setting,
        "source_run": args.source_run,
        "seed": int(args.seed),
        "source_format": args.source_format,
        "synth_dir": str(synth_dir),
        "candidate_count": len(dirs),
        "git": _repo_commit(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _write_json(output_dir / "run_config.json", run_config)

    for index, path in enumerate(dirs):
        raw_completion_path = path / "full_output.txt"
        candidate_code_path = path / "new_nn.py"
        raw_completion = raw_completion_path.read_text(encoding="utf-8") if raw_completion_path.exists() else ""
        candidate_code = candidate_code_path.read_text(encoding="utf-8") if candidate_code_path.exists() else ""
        generation_error = None
        if args.source_format == "xml_completion" and not raw_completion:
            generation_error = "missing full_output.txt"
        if args.source_format == "full_code" and not candidate_code:
            generation_error = "missing new_nn.py"

        record = {
            "candidate_id": _candidate_id(args.setting, index),
            "setting": args.setting,
            "source_run": args.source_run,
            "prompt_config": {
                "collector": "collect_synth",
                "synth_dir": str(synth_dir),
                "source_b_dir": path.name,
            },
            "adapter_config": {},
            "source_format": args.source_format,
            "seed_accuracy_baseline": float(args.seed_accuracy_baseline),
            "candidate_code": candidate_code,
            "raw_completion": raw_completion,
            "generation_error": generation_error,
        }
        _append_jsonl(candidates_path, record)
        print(f"[collect_synth] {record['candidate_id']} source={path.name} error={generation_error is not None}")

    print(f"Wrote candidates: {candidates_path}")


def _failure_result(candidate: dict[str, Any], error: str) -> dict[str, Any]:
    return {
        "error": error,
        "built_ok": False,
        "forward_ok": False,
        "forward_shape_ok": False,
        "backward_ok": False,
        "loss_drop_ok": False,
        "executable_candidate": False,
        "formal_success_candidate": False,
        "seed_accuracy_baseline": candidate.get("seed_accuracy_baseline"),
    }


def _candidate_source_format(candidate: dict[str, Any]) -> str:
    return str(candidate.get("source_format") or "xml_completion").strip().lower()


def _graph_info_from_sections(init_code: str, forward_code: str):
    import ab.gpt.TuneRL as TuneRL

    if not init_code or not forward_code:
        return None
    if "self.pattern" in forward_code:
        return None
    try:
        return TuneRL.extract_graph_info(
            init_code,
            forward_code,
            legacy_patterns=TuneRL.SFTUtil.legacy_patterns,
        )
    except Exception:
        return None


def _full_code_eval_spec(
    candidate: dict[str, Any],
    *,
    index: int,
    batch_last_item: bool,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    import torch
    import ab.gpt.TuneRL as TuneRL
    import ab.gpt.TuneRLSft as TuneRLSft

    code = str(candidate.get("candidate_code") or "")
    if candidate.get("generation_error") or not code:
        return None
    try:
        block_code, init_code, forward_code = _full_code_sections(code)
    except Exception:
        block_code, init_code, forward_code = "", "", ""
    graph_info = _graph_info_from_sections(init_code, forward_code)
    backbone_model_names = TuneRL._extract_backbone_model_names(init_code)
    seed_accuracy_baseline = float(candidate.get("seed_accuracy_baseline") or 0.10)
    prm = {
        "lr": 0.01,
        "batch": 64,
        "dropout": 0.3,
        "momentum": 0.9,
        "transform": TuneRL.FORMAL_REWARD_TRANSFORM,
        "epoch": 1,
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = {
        "code": code,
        "in_shape": (1, 3, 224, 224),
        "out_shape": (10,),
        "prm": prm,
        "device": device,
        "seed_accuracy_baseline": seed_accuracy_baseline,
        "reward_batch_index": 0,
        "completion_index": index,
        "batch_last_item": batch_last_item,
        "cfg": TuneRLSft.build_sft_reward_eval_cfg(
            stage_name=str(TuneRL.current_stage_name),
            in_shape=(1, 3, 224, 224),
            out_shape=(10,),
            prm=prm,
            cfg=None,
            device=device,
        ),
    }
    meta = {
        "graph_info": graph_info,
        "block_code": block_code,
        "init_code": init_code,
        "forward_code": forward_code,
        "backbone_model_names": backbone_model_names,
        "seed_accuracy_baseline": seed_accuracy_baseline,
    }
    return spec, meta


def _augment_full_code_result(
    candidate: dict[str, Any],
    result: dict[str, Any],
    meta: dict[str, Any],
) -> dict[str, Any]:
    import ab.gpt.TuneRL as TuneRL

    res = dict(result or {})
    graph_info = meta.get("graph_info")
    block_code = str(meta.get("block_code") or "")
    backbone_model_names = list(meta.get("backbone_model_names") or [])
    backbone_signature = TuneRL.build_backbone_signature(backbone_model_names)
    block_signature = TuneRL._block_signature_from_code(block_code)
    cnn_signature = (
        str(getattr(graph_info, "cnn_signature", "") or "")
        if graph_info is not None
        else "incomplete_cnn"
    )
    cnn_expr = (
        str(getattr(graph_info, "cnn_expr", "") or "")
        if graph_info is not None
        else "IncompleteCNN"
    )
    graph_hash = str(getattr(graph_info, "graph_hash", "") or "")
    pattern_name = (
        str(getattr(graph_info, "pattern_name", "") or getattr(graph_info, "suggested_pattern_name", "") or "")
        if graph_info is not None
        else str((candidate.get("prompt_config") or {}).get("pattern_name") or "")
    )
    executable_candidate = bool(
        graph_info
        and bool(getattr(graph_info, "parse_ok", False))
        and res.get("built_ok")
        and res.get("forward_shape_ok")
    )
    try:
        has_formal_epoch = int(res.get("epochs_completed", 0) or 0) >= 1
    except (TypeError, ValueError):
        has_formal_epoch = False
    formal_success_candidate = bool(
        executable_candidate
        and (
            res.get("backward_ok")
            or res.get("trained_step_ok")
            or has_formal_epoch
        )
    )
    res.setdefault("seed_accuracy_baseline", meta.get("seed_accuracy_baseline"))
    res["source_format"] = "full_code"
    res["executable_candidate"] = executable_candidate
    res["formal_success_candidate"] = formal_success_candidate
    res["discovery_candidate"] = bool(
        executable_candidate
        and graph_info is not None
        and bool(getattr(graph_info, "parse_ok", False))
    )
    res["backbone_model_names"] = backbone_model_names
    res["backbone_signature"] = backbone_signature
    res["cnn_signature"] = cnn_signature
    res["cnn_expr"] = cnn_expr
    res["block_signature"] = block_signature
    res["graph_hash"] = graph_hash
    res["signature"] = f"{pattern_name}_{graph_hash[:6]}" if graph_hash else pattern_name
    res["family_id"] = str(getattr(graph_info, "family_id", "") or "") if graph_info is not None else ""
    res["family_expr"] = str(getattr(graph_info, "family_expr", "") or "") if graph_info is not None else ""
    res["family_hash"] = str(getattr(graph_info, "family_hash", "") or "") if graph_info is not None else ""
    res["descriptor_key"] = str(getattr(graph_info, "descriptor_key", "") or "") if graph_info is not None else ""
    res["pattern_name"] = pattern_name
    res.setdefault("current_stage_name", TuneRL.current_stage_name)
    res.setdefault("current_stage_index", TuneRL.RL_STAGE_TO_INDEX.get(TuneRL.current_stage_name, 0))
    res.setdefault("stage_uses_formal_eval", True)
    res.setdefault("stage_uses_static_only", False)
    return res


def _evaluate_full_code_candidate(
    candidate: dict[str, Any],
    *,
    index: int,
    total: int,
) -> tuple[float, dict[str, Any], str]:
    import ab.gpt.util.Reward as RewardUtil

    prompt = str(candidate.get("prompt") or "")
    if candidate.get("generation_error"):
        result = _failure_result(candidate, str(candidate["generation_error"]))
        return -2.0, result, prompt
    spec_meta = _full_code_eval_spec(candidate, index=index, batch_last_item=index == total - 1)
    if spec_meta is None:
        result = _failure_result(candidate, "missing candidate_code")
        return -2.0, result, prompt
    spec, meta = spec_meta
    result = RewardUtil.evaluate_code_and_reward_batch([spec])[0]
    api_result = _augment_full_code_result(candidate, result, meta)
    return float(api_result.get("reward", -2.0) or -2.0), api_result, prompt


def _evaluate_candidate(candidate: dict[str, Any], *, index: int, total: int) -> tuple[float, dict[str, Any], str]:
    import ab.gpt.TuneRLSft as TuneRLSft

    if _candidate_source_format(candidate) == "full_code":
        return _evaluate_full_code_candidate(candidate, index=index, total=total)

    completion = str(candidate.get("raw_completion") or "")
    prompt = str(candidate.get("prompt") or "")
    if candidate.get("generation_error"):
        result = _failure_result(candidate, str(candidate["generation_error"]))
        return -2.0, result, prompt
    if not completion:
        result = _failure_result(candidate, "missing raw_completion")
        return -2.0, result, prompt

    result = TuneRLSft.sft_reward_fn(
        completion,
        seed_accuracy_baseline=float(candidate.get("seed_accuracy_baseline") or 0.10),
        reward_batch_index=0,
        reward_group_id=0,
        group_warmup=False,
        completion_index=index,
        batch_last_item=index == total - 1,
    )
    return float(result.get("reward", -2.0) or -2.0), result, prompt


def _build_batch_reward_entry(candidate: dict[str, Any], *, index: int) -> dict[str, Any] | None:
    import ab.gpt.TuneRL as TuneRL

    if _candidate_source_format(candidate) == "full_code":
        spec_meta = _full_code_eval_spec(candidate, index=index, batch_last_item=False)
        if spec_meta is None:
            return None
        spec, meta = spec_meta
        return {
            "rank": 0,
            "local_index": index,
            "global_index": index,
            "source_format": "full_code",
            "direct_eval_spec": spec,
            "direct_eval_meta": meta,
            "precomputed_eval_result": None,
        }

    completion = str(candidate.get("raw_completion") or "")
    if candidate.get("generation_error") or not completion:
        return None

    block_code, init_code, forward_code = TuneRL.extract_completion_blocks(completion)
    backbone_model_names = TuneRL._extract_backbone_model_names(init_code)
    graph_info = None
    if block_code and init_code and forward_code and "self.pattern" not in forward_code:
        graph_info = TuneRL.extract_graph_info(
            init_code,
            forward_code,
            legacy_patterns=TuneRL.SFTUtil.legacy_patterns,
        )

    return {
        "rank": 0,
        "local_index": index,
        "global_index": index,
        "completion": completion,
        "graph_info": graph_info,
        "backbone_model_names": backbone_model_names,
        "backbone_signature": TuneRL.build_backbone_signature(backbone_model_names),
        "cnn_signature": (
            str(getattr(graph_info, "cnn_signature", "") or "")
            if graph_info is not None
            else "incomplete_cnn"
        ),
        "prompt_goal_tags": [],
        "goal_key": TuneRL.primary_goal_key([]),
        "seed_accuracy_baseline": float(candidate.get("seed_accuracy_baseline") or 0.10),
        "precomputed_eval_result": None,
    }


def _evaluate_candidate_batch(
    candidates: list[dict[str, Any]],
    *,
    start_index: int,
    total: int,
) -> list[tuple[float, dict[str, Any], str]]:
    import ab.gpt.TuneRL as TuneRL
    import ab.gpt.TuneRLSft as TuneRLSft
    import ab.gpt.util.Reward as RewardUtil

    entries_by_offset: dict[int, dict[str, Any]] = {}
    precompute_entries: list[dict[str, Any]] = []
    direct_entries: list[tuple[int, dict[str, Any]]] = []
    direct_specs: list[dict[str, Any]] = []
    for offset, candidate in enumerate(candidates):
        index = start_index + offset
        entry = _build_batch_reward_entry(candidate, index=index)
        if entry is None:
            continue
        entries_by_offset[offset] = entry
        if entry.get("source_format") == "full_code":
            direct_entries.append((offset, entry))
            direct_specs.append(dict(entry["direct_eval_spec"]))
        else:
            precompute_entries.append(entry)

    if precompute_entries:
        TuneRL._precompute_eval_results(
            precompute_entries,
            group_context={
                "reward_batch_index": 0,
                "current_stage_name": TuneRL.current_stage_name,
            },
        )
    if direct_specs:
        direct_specs[-1]["batch_last_item"] = True
        direct_results = RewardUtil.evaluate_code_and_reward_batch(direct_specs)
        for (offset, entry), eval_result in zip(direct_entries, direct_results):
            entry["precomputed_eval_result"] = eval_result

    results: list[tuple[float, dict[str, Any], str]] = []
    for offset, candidate in enumerate(candidates):
        index = start_index + offset
        completion = str(candidate.get("raw_completion") or "")
        prompt = str(candidate.get("prompt") or "")
        if candidate.get("generation_error"):
            api_result = _failure_result(candidate, str(candidate["generation_error"]))
            results.append((-2.0, api_result, prompt))
            continue

        entry = entries_by_offset.get(offset)
        if _candidate_source_format(candidate) == "full_code":
            if entry is None or entry.get("precomputed_eval_result") is None:
                api_result = _failure_result(candidate, "missing full-code eval result")
            else:
                api_result = _augment_full_code_result(
                    candidate,
                    entry["precomputed_eval_result"],
                    entry.get("direct_eval_meta") or {},
                )
            results.append((float(api_result.get("reward", -2.0) or -2.0), api_result, prompt))
            continue

        if not completion:
            api_result = _failure_result(candidate, "missing raw_completion")
            results.append((-2.0, api_result, prompt))
            continue

        api_result = TuneRLSft.sft_reward_fn(
            completion,
            seed_accuracy_baseline=float(candidate.get("seed_accuracy_baseline") or 0.10),
            precomputed_eval_result=(
                entry.get("precomputed_eval_result")
                if entry is not None and entry.get("precomputed_eval_result") is not None
                else None
            ),
            graph_info=entry.get("graph_info") if entry is not None else None,
            reward_batch_index=0,
            reward_group_id=0,
            group_warmup=False,
            completion_index=index,
            batch_last_item=index == total - 1,
        )
        results.append((float(api_result.get("reward", -2.0) or -2.0), api_result, prompt))

    return results


def command_eval_only(args: argparse.Namespace) -> None:
    _configure_sft_env(args)
    _configure_eval_runtime()

    try:
        import torch
        gpu_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:
        gpu_count = 0
    import ab.gpt.util.Reward as RewardUtil

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "generation_samples.jsonl"
    if samples_path.exists() and not args.append:
        samples_path.unlink()

    candidates: list[dict[str, Any]] = []
    for path_text in args.candidate_file:
        candidates.extend(_read_jsonl(Path(path_text)))
    if args.limit is not None:
        candidates = candidates[: int(args.limit)]
    if not candidates:
        raise RuntimeError("No candidates provided for eval-only")
    source_format_counts = Counter(_candidate_source_format(candidate) for candidate in candidates)

    run_config = {
        "phase": "eval_only",
        "candidate_files": list(args.candidate_file),
        "seed": int(args.seed),
        "candidate_count": len(candidates),
        "source_formats": dict(source_format_counts),
        "gpu_count": gpu_count,
        "eval_config": {
            "dataset": "cifar-10",
            "transform": "norm_128_flip",
            "resize": 128,
            "batch": 64,
            "formal_reward_epochs": os.environ.get("NNGPT_RL_FORMAL_REWARD_EPOCHS", "1,5,10"),
            "workers_per_gpu": os.environ.get("NNGPT_REWARD_WORKERS_PER_GPU", ""),
            "eval_concurrency": int(args.eval_concurrency),
        },
        "git": _repo_commit(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _write_json(output_dir / "run_config.json", run_config)
    warmup = RewardUtil.prewarm_eval_workers(timeout_seconds=60.0, require_gpu=True)
    print(json.dumps({"reward_worker_warmup": warmup}, ensure_ascii=False, default=_json_default))

    warmup_pool_size = int(warmup.get("pool_size") or 1) if isinstance(warmup, dict) else 1
    eval_concurrency = int(args.eval_concurrency)
    if eval_concurrency <= 0:
        eval_concurrency = max(1, warmup_pool_size)
    eval_concurrency = max(1, eval_concurrency)
    print(f"[eval_only] eval_concurrency={eval_concurrency} warmup_pool_size={warmup_pool_size}")

    for start_index in range(0, len(candidates), eval_concurrency):
        chunk = candidates[start_index : start_index + eval_concurrency]
        if eval_concurrency <= 1:
            chunk_results = [
                _evaluate_candidate(chunk[0], index=start_index, total=len(candidates))
            ]
        else:
            chunk_results = _evaluate_candidate_batch(
                chunk,
                start_index=start_index,
                total=len(candidates),
            )
        for offset, (reward, api_result, prompt) in enumerate(chunk_results):
            index = start_index + offset
            candidate = candidates[index]
            record = {
                "prompt": prompt,
                "completion": candidate.get("raw_completion", ""),
                "reward": reward,
                "api_result": api_result,
                "candidate": candidate,
                "candidate_id": candidate.get("candidate_id"),
                "setting": candidate.get("setting"),
            }
            _append_jsonl(samples_path, record)
            print(
                "[eval_only] "
                f"{index + 1}/{len(candidates)} "
                f"{candidate.get('candidate_id')} "
                f"formal_success={bool(api_result.get('formal_success_candidate'))} "
                f"acc={api_result.get('frozen_test_acc')}"
            )

    RewardUtil.shutdown_eval_worker()
    command_summarize(
        argparse.Namespace(
            input=[f"eval={samples_path}"],
            output_dir=str(output_dir),
            subsample=None,
            seed=args.seed,
        )
    )


def _parse_summary_input(value: str) -> tuple[str | None, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
        return label.strip() or None, Path(path)
    return None, Path(value)


def _formal_success_from_result(result: dict[str, Any]) -> bool:
    if "formal_success_candidate" in result:
        return bool(result.get("formal_success_candidate"))
    try:
        epochs_completed = int(result.get("epochs_completed", 0) or 0)
    except (TypeError, ValueError):
        epochs_completed = 0
    return bool(
        result.get("built_ok")
        and result.get("forward_shape_ok")
        and (
            result.get("backward_ok")
            or result.get("trained_step_ok")
            or epochs_completed >= 1
        )
    )


def _horizon_acc_from_result(
    result: dict[str, Any],
    horizon: int | str,
    *,
    fallback: bool = True,
) -> float | None:
    horizons = result.get("formal_horizon_test_acc")
    if isinstance(horizons, dict):
        for key in (str(horizon), horizon):
            value = horizons.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                return float(value)
    if not fallback:
        return None
    for key in ("frozen_test_acc", "test_acc", "val_metric"):
        value = result.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def _acc_from_result(result: dict[str, Any]) -> float | None:
    for key in ("frozen_test_acc", "formal_reward_target_value", "reward_target_value", "test_acc", "val_metric"):
        value = result.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    horizons = result.get("formal_horizon_test_acc")
    if isinstance(horizons, dict):
        values = [float(v) for v in horizons.values() if isinstance(v, (int, float)) and math.isfinite(float(v))]
        if values:
            return values[-1]
    return None


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return ordered[lower]
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _top_mean(values: list[float], k: int) -> float | None:
    if not values:
        return None
    top = sorted(values, reverse=True)[:k]
    return _mean(top)


def _effective_number(counter: Counter[str]) -> float | None:
    total = sum(counter.values())
    if total <= 0:
        return None
    entropy = 0.0
    for count in counter.values():
        p = count / total
        entropy -= p * math.log(p)
    return math.exp(entropy)


def _diversity(counter: Counter[str], sample_count: int) -> dict[str, Any]:
    total = sum(counter.values())
    top_key, top_count = ("", 0)
    if counter:
        top_key, top_count = counter.most_common(1)[0]
    return {
        "unique": len(counter),
        "normalized_unique_ratio": (len(counter) / sample_count) if sample_count else None,
        "top1": top_key,
        "top1_count": top_count,
        "top1_share": (top_count / total) if total else None,
        "effective_number": _effective_number(counter),
    }


def _record_setting(record: dict[str, Any], fallback_label: str | None) -> str:
    if record.get("setting"):
        return str(record["setting"])
    candidate = record.get("candidate")
    if isinstance(candidate, dict) and candidate.get("setting"):
        return str(candidate["setting"])
    return str(fallback_label or "unknown")


def _failure_mode(record: dict[str, Any]) -> str:
    candidate = record.get("candidate")
    if isinstance(candidate, dict) and candidate.get("generation_error"):
        return str(candidate["generation_error"]).split(":", 1)[0]
    result = record.get("api_result")
    if isinstance(result, dict):
        error = str(result.get("error") or result.get("failure_reason") or "").strip()
        if error:
            return error.split("\n", 1)[0][:160]
        if not bool(result.get("built_ok", False)):
            return "build_failed"
        if not bool(result.get("forward_shape_ok", False)):
            return "forward_shape_failed"
        if not bool(result.get("loss_drop_ok", False)):
            return "loss_drop_failed"
    return "not_formal_success"


def command_summarize(args: argparse.Namespace) -> None:
    rng = random.Random(int(args.seed))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in args.input:
        label, path = _parse_summary_input(item)
        for record in _read_jsonl(path):
            grouped[_record_setting(record, label)].append(record)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    diversity_summary: dict[str, Any] = {}
    failure_summary: dict[str, Any] = {}

    for setting, records in sorted(grouped.items()):
        if args.subsample is not None and len(records) > int(args.subsample):
            records = rng.sample(records, int(args.subsample))
        successes: list[dict[str, Any]] = []
        acc_values: list[float] = []
        failures = Counter()
        format_valid_count = 0
        counters = {
            "backbone": Counter(),
            "cnn": Counter(),
            "backbone_cnn": Counter(),
            "forward_graph": Counter(),
        }

        for record in records:
            candidate = record.get("candidate")
            if isinstance(candidate, dict):
                source_format = _candidate_source_format(candidate)
                if not candidate.get("generation_error"):
                    if source_format == "full_code" and candidate.get("candidate_code"):
                        format_valid_count += 1
                    elif source_format != "full_code" and candidate.get("raw_completion"):
                        format_valid_count += 1
            result = record.get("api_result") if isinstance(record.get("api_result"), dict) else {}
            formal_success = bool(result.get("formal_success_candidate"))
            if formal_success:
                successes.append(record)
                acc = _acc_from_result(result)
                if acc is not None:
                    acc_values.append(acc)
                backbone = str(result.get("backbone_signature") or "")
                cnn = str(result.get("cnn_signature") or "")
                graph = str(result.get("graph_hash") or result.get("signature") or "")
                if backbone:
                    counters["backbone"][backbone] += 1
                if cnn:
                    counters["cnn"][cnn] += 1
                if backbone or cnn:
                    counters["backbone_cnn"][f"{backbone}::{cnn}"] += 1
                if graph:
                    counters["forward_graph"][graph] += 1
            else:
                failures[_failure_mode(record)] += 1

        sample_count = len(records)
        summary[setting] = {
            "samples": sample_count,
            "format_valid_count": format_valid_count,
            "format_valid_rate": (format_valid_count / sample_count) if sample_count else None,
            "formal_success_count": len(successes),
            "formal_success_rate": (len(successes) / sample_count) if sample_count else None,
            "mean_acc": _mean(acc_values),
            "max_acc": max(acc_values) if acc_values else None,
            "median_acc": _quantile(acc_values, 0.5),
            "q1_acc": _quantile(acc_values, 0.25),
            "q3_acc": _quantile(acc_values, 0.75),
            "top5_mean_acc": _top_mean(acc_values, 5),
            "top10_mean_acc": _top_mean(acc_values, 10),
        }
        diversity_summary[setting] = {
            key: _diversity(counter, len(successes))
            for key, counter in counters.items()
        }
        failure_summary[setting] = dict(failures.most_common())

    _write_json(output_dir / "summary_metrics.json", summary)
    _write_json(output_dir / "structural_diversity_summary.json", diversity_summary)
    _write_json(output_dir / "failure_modes.json", failure_summary)
    _write_markdown_table(output_dir / "baseline_table.md", summary, diversity_summary)
    print(f"Wrote summaries under {output_dir}")


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _write_markdown_table(path: Path, summary: dict[str, Any], diversity: dict[str, Any]) -> None:
    headers = [
        "Setting",
        "Samples",
        "Format valid",
        "Correct rate",
        "Mean acc",
        "Max acc",
        "Median",
        "Top-5 mean",
        "Top-10 mean",
        "Backbone top-1 share",
        "CNN top-1 share",
        "Forward graph top-1 share",
        "Backbone unique ratio",
        "Forward graph effective #",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |",
    ]
    for setting, metrics in sorted(summary.items()):
        div = diversity.get(setting, {})
        row = [
            setting,
            _fmt(metrics.get("samples")),
            _fmt(metrics.get("format_valid_rate")),
            _fmt(metrics.get("formal_success_rate")),
            _fmt(metrics.get("mean_acc")),
            _fmt(metrics.get("max_acc")),
            _fmt(metrics.get("median_acc")),
            _fmt(metrics.get("top5_mean_acc")),
            _fmt(metrics.get("top10_mean_acc")),
            _fmt(div.get("backbone", {}).get("top1_share")),
            _fmt(div.get("cnn", {}).get("top1_share")),
            _fmt(div.get("forward_graph", {}).get("top1_share")),
            _fmt(div.get("backbone", {}).get("normalized_unique_ratio")),
            _fmt(div.get("forward_graph", {}).get("effective_number")),
        ]
        lines.append("| " + " | ".join(row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


TOP20_REEVAL_GROUPS = [
    "prompt_only",
    "sft_only_onepattern",
    "sft_only_fourpattern",
    "brute_constrained_random",
    "rl_onepattern_last100",
    "rl_fourpattern_last100",
]


def _candidate_from_selected_baseline(
    record: dict[str, Any],
    *,
    group: str,
    rank: int,
    acc1: float,
    source_jsonl: Path,
    source_line_index: int,
) -> dict[str, Any]:
    candidate = dict(record.get("candidate") or {})
    original_candidate_id = str(candidate.get("candidate_id") or record.get("candidate_id") or "")
    candidate["original_setting"] = str(candidate.get("setting") or record.get("setting") or group)
    candidate["original_candidate_id"] = original_candidate_id
    candidate["candidate_id"] = f"{group}-top20-{rank:04d}"
    candidate["setting"] = group
    candidate["selection_group"] = group
    candidate["selection_rank"] = rank
    candidate["selection_acc1"] = acc1
    candidate["source_jsonl"] = str(source_jsonl)
    candidate["source_line_index"] = source_line_index
    candidate["source_window"] = "all"
    return candidate


def _candidate_from_selected_rl(
    record: dict[str, Any],
    *,
    group: str,
    source_run: str,
    rank: int,
    acc1: float,
    source_jsonl: Path,
    source_line_index: int,
) -> dict[str, Any]:
    result = record.get("api_result") if isinstance(record.get("api_result"), dict) else {}
    completion = str(record.get("completion") or "")
    return {
        "candidate_id": f"{group}-top20-{rank:04d}",
        "setting": group,
        "source_run": source_run,
        "prompt_config": {
            "source": "rl_trajectory_top20_reeval",
            "selection_horizon": 1,
            "source_window": "last100",
        },
        "adapter_config": {
            "source": source_run,
        },
        "candidate_code": "",
        "raw_completion": completion,
        "generation_error": None,
        "source_format": "xml_completion",
        "seed_accuracy_baseline": result.get("seed_accuracy_baseline", 0.10),
        "prompt": record.get("prompt", ""),
        "original_setting": source_run,
        "original_candidate_id": str(record.get("candidate_id") or ""),
        "selection_group": group,
        "selection_rank": rank,
        "selection_acc1": acc1,
        "source_jsonl": str(source_jsonl),
        "source_line_index": source_line_index,
        "source_window": "last100",
    }


def _select_top_successes(
    indexed_records: list[tuple[int, dict[str, Any]]],
    *,
    group: str,
    limit: int,
    horizon: int = 1,
) -> list[tuple[int, dict[str, Any], float]]:
    selected: list[tuple[int, dict[str, Any], float]] = []
    for line_index, record in indexed_records:
        result = record.get("api_result") if isinstance(record.get("api_result"), dict) else {}
        if not _formal_success_from_result(result):
            continue
        acc = _horizon_acc_from_result(result, horizon, fallback=True)
        if acc is None:
            continue
        selected.append((line_index, record, acc))
    selected.sort(key=lambda item: (-item[2], item[0]))
    return selected[:limit]


def command_build_top20_reeval_queue(args: argparse.Namespace) -> None:
    baseline_path = Path(args.baseline_jsonl)
    rl_one_path = Path(args.rl_onepattern_jsonl)
    rl_four_path = Path(args.rl_fourpattern_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    queue_path = output_dir / "candidates_top20_10epoch.jsonl"
    if queue_path.exists():
        queue_path.unlink()

    baseline_rows = list(enumerate(_read_jsonl(baseline_path), start=1))
    baseline_by_setting: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for line_index, record in baseline_rows:
        baseline_by_setting[_record_setting(record, None)].append((line_index, record))

    group_candidates: dict[str, list[dict[str, Any]]] = {group: [] for group in TOP20_REEVAL_GROUPS}
    group_sources: dict[str, dict[str, Any]] = {}

    baseline_limits = {
        "prompt_only": 20,
        "sft_only_onepattern": 20,
        "sft_only_fourpattern": 20,
        "brute_constrained_random": 20,
    }
    for group, limit in baseline_limits.items():
        picked = _select_top_successes(
            baseline_by_setting.get(group, []),
            group=group,
            limit=limit,
            horizon=1,
        )
        for rank, (line_index, record, acc1) in enumerate(picked, start=1):
            group_candidates[group].append(
                _candidate_from_selected_baseline(
                    record,
                    group=group,
                    rank=rank,
                    acc1=acc1,
                    source_jsonl=baseline_path,
                    source_line_index=line_index,
                )
            )
        group_sources[group] = {
            "source_jsonl": str(baseline_path),
            "source_window": "all",
            "available_successes": len(_select_top_successes(
                baseline_by_setting.get(group, []),
                group=group,
                limit=10_000,
                horizon=1,
            )),
            "selected": len(group_candidates[group]),
        }

    rl_specs = [
        ("rl_onepattern_last100", "onepattern_rl_a7_g10b", rl_one_path),
        ("rl_fourpattern_last100", "fourpattern_rl_a3_g10b", rl_four_path),
    ]
    for group, source_run, path in rl_specs:
        rows = list(enumerate(_read_jsonl(path), start=1))
        window_rows = rows[-100:]
        picked = _select_top_successes(window_rows, group=group, limit=20, horizon=1)
        for rank, (line_index, record, acc1) in enumerate(picked, start=1):
            group_candidates[group].append(
                _candidate_from_selected_rl(
                    record,
                    group=group,
                    source_run=source_run,
                    rank=rank,
                    acc1=acc1,
                    source_jsonl=path,
                    source_line_index=line_index,
                )
            )
        group_sources[group] = {
            "source_jsonl": str(path),
            "source_window": "last100",
            "available_successes": len(_select_top_successes(window_rows, group=group, limit=10_000, horizon=1)),
            "selected": len(group_candidates[group]),
        }

    interleaved: list[dict[str, Any]] = []
    max_group_count = max((len(items) for items in group_candidates.values()), default=0)
    for offset in range(max_group_count):
        for group in TOP20_REEVAL_GROUPS:
            items = group_candidates[group]
            if offset < len(items):
                interleaved.append(items[offset])

    for candidate in interleaved:
        _append_jsonl(queue_path, candidate)

    manifest = {
        "phase": "top20_10epoch_selection",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "selection_horizon": 1,
        "selection_rule": "formal_success candidates sorted by 1-epoch test accuracy desc, tie-broken by source JSONL line index",
        "queue_path": str(queue_path),
        "total_selected": len(interleaved),
        "groups": group_sources,
        "group_order": TOP20_REEVAL_GROUPS,
        "git": _repo_commit(),
    }
    _write_json(output_dir / "selection_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default))


def _selection_group(record: dict[str, Any], fallback_label: str | None = None) -> str:
    candidate = record.get("candidate")
    if isinstance(candidate, dict):
        group = candidate.get("selection_group") or candidate.get("setting")
        if group:
            return str(group)
    return _record_setting(record, fallback_label)


def _write_top20_reeval_table(path: Path, summary: dict[str, Any]) -> None:
    headers = [
        "Setting",
        "Reeval N",
        "Selected 1-epoch mean",
        "10-epoch mean",
        "10-epoch max",
        "10-epoch median",
        "10-epoch top-5 mean",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |",
    ]
    for group in TOP20_REEVAL_GROUPS:
        metrics = summary.get(group)
        if not metrics:
            continue
        row = [
            group,
            _fmt(metrics.get("samples")),
            _fmt(metrics.get("selection_acc1_mean")),
            _fmt(metrics.get("acc10_mean")),
            _fmt(metrics.get("acc10_max")),
            _fmt(metrics.get("acc10_median")),
            _fmt(metrics.get("acc10_top5_mean")),
        ]
        lines.append("| " + " | ".join(row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def command_summarize_top20_reeval(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in _read_jsonl(input_path):
        grouped[_selection_group(record)].append(record)

    summary: dict[str, Any] = {}
    failure_summary: dict[str, Any] = {}
    selection_manifest = {
        "phase": "top20_10epoch_reeval_summary",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "input": str(input_path),
        "horizon": 10,
        "groups": {},
        "git": _repo_commit(),
    }

    for group in TOP20_REEVAL_GROUPS:
        records = grouped.get(group, [])
        selection_acc1: list[float] = []
        acc10: list[float] = []
        failures = Counter()
        reeval_success_count = 0
        for record in records:
            candidate = record.get("candidate") if isinstance(record.get("candidate"), dict) else {}
            value = candidate.get("selection_acc1") if isinstance(candidate, dict) else None
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                selection_acc1.append(float(value))
            result = record.get("api_result") if isinstance(record.get("api_result"), dict) else {}
            if _formal_success_from_result(result):
                reeval_success_count += 1
                value10 = _horizon_acc_from_result(result, 10, fallback=True)
                if value10 is not None:
                    acc10.append(value10)
            else:
                failures[_failure_mode(record)] += 1

        summary[group] = {
            "samples": len(records),
            "reeval_success_count": reeval_success_count,
            "reeval_success_rate": (reeval_success_count / len(records)) if records else None,
            "selection_acc1_count": len(selection_acc1),
            "selection_acc1_mean": _mean(selection_acc1),
            "acc10_count": len(acc10),
            "acc10_mean": _mean(acc10),
            "acc10_max": max(acc10) if acc10 else None,
            "acc10_median": _quantile(acc10, 0.5),
            "acc10_top5_mean": _top_mean(acc10, 5),
        }
        failure_summary[group] = dict(failures.most_common())
        selection_manifest["groups"][group] = {
            "records": len(records),
            "source_lines": [
                {
                    "candidate_id": (record.get("candidate") or {}).get("candidate_id"),
                    "original_candidate_id": (record.get("candidate") or {}).get("original_candidate_id"),
                    "source_jsonl": (record.get("candidate") or {}).get("source_jsonl"),
                    "source_line_index": (record.get("candidate") or {}).get("source_line_index"),
                    "selection_rank": (record.get("candidate") or {}).get("selection_rank"),
                    "selection_acc1": (record.get("candidate") or {}).get("selection_acc1"),
                }
                for record in records
            ],
        }

    _write_json(output_dir / "top20_10epoch_summary.json", summary)
    _write_json(output_dir / "failure_modes.json", failure_summary)
    _write_json(output_dir / "selection_manifest.json", selection_manifest)
    _write_top20_reeval_table(output_dir / "top20_10epoch_table.md", summary)
    print(f"Wrote top20 10-epoch summary under {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("gen-only")
    gen.add_argument("--setting", required=True)
    gen.add_argument("--source-run", default="")
    gen.add_argument("--output-dir", required=True)
    gen.add_argument("--budget", type=int, default=100)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--append", action="store_true")
    gen.add_argument("--base-model-id", default=os.getenv("NNGPT_SFT_BASE_MODEL_ID", ""))
    gen.add_argument("--tokenizer-id", default=os.getenv("NNGPT_SFT_TOKENIZER_ID", ""))
    gen.add_argument("--adapter-path", default=os.getenv("NNGPT_BASELINE_ADAPTER_PATH", ""))
    gen.add_argument("--adapter-mode", default=os.getenv("NNGPT_SFT_INITIAL_ADAPTER_MODE", "trainable"))
    gen.add_argument("--nn-prefixes", default=os.getenv("NNGPT_SFT_RL_NN_PREFIXES", "rl-bb-test1"))
    gen.add_argument("--prompt-mode", default=os.getenv("NNGPT_SFT_RL_PROMPT_MODE", "sft_aligned"))
    gen.add_argument("--feedback-char-budget", type=int, default=int(os.getenv("NNGPT_SFT_FEEDBACK_CHAR_BUDGET", "0") or 0))
    gen.add_argument("--temperature", type=float, default=float(os.getenv("NNGPT_SFT_TEMPERATURE", "0.8") or 0.8))
    gen.add_argument("--top-p", type=float, default=float(os.getenv("NNGPT_SFT_TOP_P", "0.95") or 0.95))
    gen.add_argument("--top-k", type=int, default=int(os.getenv("NNGPT_SFT_TOP_K", "50") or 50))
    gen.add_argument("--max-new-tokens", type=int, default=int(os.getenv("NNGPT_SFT_MAX_COMPLETION_LENGTH", "1536") or 1536))
    gen.set_defaults(func=command_gen_only)

    brute = sub.add_parser("brute-gen-only")
    brute.add_argument("--setting", default="brute_constrained_random")
    brute.add_argument("--source-run", default="")
    brute.add_argument("--output-dir", required=True)
    brute.add_argument("--budget", type=int, default=100)
    brute.add_argument("--seed", type=int, default=42)
    brute.add_argument("--append", action="store_true")
    brute.add_argument("--pattern-set", default="union", choices=("basic", "diverse", "all", "union"))
    brute.add_argument("--seed-accuracy-baseline", type=float, default=0.10)
    brute.set_defaults(func=command_brute_gen_only)

    collect = sub.add_parser("collect-synth")
    collect.add_argument("--setting", required=True)
    collect.add_argument("--source-run", default="")
    collect.add_argument("--synth-dir", required=True)
    collect.add_argument("--output-dir", required=True)
    collect.add_argument("--limit", type=int)
    collect.add_argument("--seed", type=int, default=42)
    collect.add_argument("--append", action="store_true")
    collect.add_argument("--source-format", default="full_code", choices=("full_code", "xml_completion"))
    collect.add_argument("--seed-accuracy-baseline", type=float, default=0.10)
    collect.set_defaults(func=command_collect_synth)

    eval_parser = sub.add_parser("eval-only")
    eval_parser.add_argument("--candidate-file", action="append", required=True)
    eval_parser.add_argument("--output-dir", required=True)
    eval_parser.add_argument("--limit", type=int)
    eval_parser.add_argument("--seed", type=int, default=42)
    eval_parser.add_argument("--append", action="store_true")
    eval_parser.add_argument("--base-model-id", default="")
    eval_parser.add_argument("--tokenizer-id", default="")
    eval_parser.add_argument("--nn-prefixes", default="")
    eval_parser.add_argument("--prompt-mode", default="")
    eval_parser.add_argument("--feedback-char-budget", type=int)
    eval_parser.add_argument(
        "--eval-concurrency",
        type=int,
        default=int(os.getenv("NNGPT_BASELINE_EVAL_CONCURRENCY", "0") or 0),
        help="Number of candidates to dispatch per eval batch; <=0 uses worker pool size.",
    )
    eval_parser.set_defaults(func=command_eval_only)

    summarize = sub.add_parser("summarize")
    summarize.add_argument("--input", action="append", required=True, help="Either path or setting=path")
    summarize.add_argument("--output-dir", required=True)
    summarize.add_argument("--subsample", type=int)
    summarize.add_argument("--seed", type=int, default=42)
    summarize.set_defaults(func=command_summarize)

    top20 = sub.add_parser("build-top20-reeval-queue")
    top20.add_argument("--baseline-jsonl", required=True)
    top20.add_argument("--rl-onepattern-jsonl", required=True)
    top20.add_argument("--rl-fourpattern-jsonl", required=True)
    top20.add_argument("--output-dir", required=True)
    top20.set_defaults(func=command_build_top20_reeval_queue)

    top20_summary = sub.add_parser("summarize-top20-reeval")
    top20_summary.add_argument("--input", required=True)
    top20_summary.add_argument("--output-dir", required=True)
    top20_summary.set_defaults(func=command_summarize_top20_reeval)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
