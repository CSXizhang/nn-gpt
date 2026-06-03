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


def _describe_eval_split(dataset: str, split_protocol: str) -> tuple[str, str, str]:
    normalized_dataset = str(dataset or "").strip().lower().replace("_", "-")
    normalized_protocol = str(split_protocol or "").strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    if normalized_protocol in {"721", "7/2/1", "702010", "70/20/10", "trainval", "trainvaltest", "trainvaltestsplit"}:
        if normalized_dataset in {"cifar-10", "cifar10"}:
            return "cifar10-train[45k]", "cifar10-train[5k]", "cifar10-test[10k]"
        if normalized_dataset in {"cifar-100", "cifar100"}:
            return "cifar100-train[45k]", "cifar100-train[5k]", "cifar100-test[10k]"
        if normalized_dataset == "imagenette":
            return "imagenette-train[7500]", "imagenette-train[1969]", "imagenette-test[3925]"
    return "official-train", "official-test", "none"


def _attach_sft_generate_tokenizer(model, tokenizer) -> None:
    original_generate = model.generate

    def generate(*gen_args, **gen_kwargs):
        generation_config = gen_kwargs.get("generation_config")
        if generation_config is not None and getattr(generation_config, "stop_strings", None):
            gen_kwargs["tokenizer"] = tokenizer
        return original_generate(*gen_args, **gen_kwargs)

    model.generate = generate


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
    if getattr(args, "eval_split_protocol", None):
        os.environ["NNGPT_SFT_EVAL_SPLIT_PROTOCOL"] = str(args.eval_split_protocol)
    if getattr(args, "eval_split_seed", None) is not None:
        os.environ["NNGPT_SFT_EVAL_SPLIT_SEED"] = str(args.eval_split_seed)
    if getattr(args, "eval_split_role", None):
        os.environ["NNGPT_SFT_EVAL_SPLIT_ROLE"] = str(args.eval_split_role)
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
    _attach_sft_generate_tokenizer(model, tokenizer)

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
            from transformers import GenerationConfig

            eos_id = getattr(tokenizer, "eos_token_id", None)
            pad_id = getattr(tokenizer, "pad_token_id", None)
            gen_cfg = GenerationConfig(
                max_new_tokens=int(args.max_new_tokens),
                do_sample=True,
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                top_k=int(args.top_k),
                stop_strings=["</forward>"],
                eos_token_id=eos_id,
                pad_token_id=pad_id if pad_id is not None and pad_id != eos_id else None,
            )
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    generation_config=gen_cfg,
                    disable_compile=True,
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
    import ab.gpt.util.DatasetSplit as DatasetSplit
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
    split_protocol = DatasetSplit.normalize_split_protocol(os.environ.get("NNGPT_SFT_EVAL_SPLIT_PROTOCOL", "trainvaltest"))
    split_seed = int(os.environ.get("NNGPT_SFT_EVAL_SPLIT_SEED", "42") or 42)
    eval_split_role = str(os.environ.get("NNGPT_SFT_EVAL_SPLIT_ROLE", "reward_eval") or "reward_eval")
    train_set_label, reward_eval_label, heldout_test_label = _describe_eval_split("cifar-10", split_protocol)

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
            "split_protocol": split_protocol,
            "split_seed": split_seed,
            "eval_split_role": eval_split_role,
            "train_set": train_set_label,
            "reward_eval_set": reward_eval_label,
            "heldout_test_set": heldout_test_label,
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
        "--eval-split-protocol",
        default=os.getenv("NNGPT_SFT_EVAL_SPLIT_PROTOCOL", "trainvaltest"),
        help="Shared train/val/test split protocol for final/baseline eval; default: trainvaltest.",
    )
    eval_parser.add_argument(
        "--eval-split-seed",
        type=int,
        default=int(os.getenv("NNGPT_SFT_EVAL_SPLIT_SEED", "42") or 42),
        help="Seed for the shared train/val/test split.",
    )
    eval_parser.add_argument(
        "--eval-split-role",
        default=os.getenv("NNGPT_SFT_EVAL_SPLIT_ROLE", "reward_eval"),
        choices=("reward_eval", "heldout_test"),
        help="Which shared split to score on; reward_eval is validation, heldout_test is the final test split.",
    )
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
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
