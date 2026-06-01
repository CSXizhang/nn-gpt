from __future__ import annotations

import ast
import hashlib
import inspect
import re
import textwrap
import time
from typing import Any, Dict

import torch
from ab.gpt.rl_pipeline.completion import extract_completion_blocks_strict
from ab.gpt.util.ArchDiscovery import extract_graph_info
import ab.gpt.util.SFTUtil as SFTUtil


def _tunerl():
    import ab.gpt.TuneRL as TuneRL

    return TuneRL


def extract_seed_context(kwargs: Dict[str, Any], expected_count: int):
    return _tunerl().require_sample_accuracy_baselines(kwargs, expected_count)


def base_discovery_reward_fn(*args, **kwargs):
    return _tunerl().base_discovery_reward_fn(*args, **kwargs)


def prepare_entries(
    prompts,
    completions,
    *,
    seed_contexts,
    group_context: Dict[str, Any],
    precompute_eval: bool,
):
    entries = []
    for index, (prompt, completion, seed_context) in enumerate(zip(prompts, completions, seed_contexts)):
        record = {
            "prompt": prompt,
            "completion": completion,
            "seed_accuracy_baseline": seed_context,
        }
        entry = _entry_from_record(record, index=index)
        entry["rank"] = _tunerl()._distributed_rank()
        entries.append(entry)
    if precompute_eval:
        precompute_entries(entries, group_context=group_context)
    return entries


def precompute_entries(entries, *, group_context: Dict[str, Any]) -> None:
    batched_eval_entries, batched_eval_specs = _build_batched_eval_specs(
        entries,
        group_context=group_context,
    )
    if not batched_eval_specs:
        return
    TuneRL = _tunerl()
    rank = TuneRL._distributed_rank()
    local_rank = TuneRL.env_int("LOCAL_RANK", 0)
    started_at = time.time()
    print(
        "[Reward Precompute Local] start "
        f"rank={rank} "
        f"local_rank={local_rank} "
        f"reward_batch_index={group_context.get('reward_batch_index')} "
        f"entries={len(batched_eval_specs)} "
        f"wall_time={started_at:.6f}"
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    batched_eval_results = TuneRL.evaluate_reward_code_batch(batched_eval_specs)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    ended_at = time.time()
    elapsed_seconds = max(0.0, ended_at - started_at)
    print(
        "[Reward Precompute Local] end "
        f"rank={rank} "
        f"local_rank={local_rank} "
        f"reward_batch_index={group_context.get('reward_batch_index')} "
        f"entries={len(batched_eval_specs)} "
        f"elapsed_seconds={elapsed_seconds:.2f} "
        f"wall_time={ended_at:.6f}"
    )
    for entry, eval_result in zip(batched_eval_entries, batched_eval_results):
        entry["precomputed_eval_result"] = eval_result


def score_entries(
    entries,
    *,
    group_context: Dict[str, Any],
    archive_snapshot_family_counts: Dict[str, int],
):
    TuneRL = _tunerl()
    archive_snapshot_descriptor_counts = dict(TuneRL.descriptor_archive_counts)
    archive_snapshot_backbone_signature_counts = dict(TuneRL.backbone_signature_archive_counts)
    archive_snapshot_cnn_signature_counts = dict(TuneRL.cnn_signature_archive_counts)
    archive_snapshot_graph_counts = dict(TuneRL.graph_archive_counts)
    archive_snapshot_block_signature_counts = dict(TuneRL.block_signature_archive_counts)
    archive_snapshot_backbone_cnn_pair_counts = dict(TuneRL.backbone_cnn_pair_archive_counts)
    archive_snapshot_backbone_block_pair_counts = dict(TuneRL.backbone_block_pair_archive_counts)
    archive_snapshot_backbone_block_best_quality = dict(TuneRL.best_quality_acc_by_backbone_block)
    batch_graph_hashes = [
        entry["graph_info"].graph_hash if entry.get("graph_info") and entry["graph_info"].parse_ok else "incomplete"
        for entry in entries
    ]
    batch_family_hashes = [
        entry["graph_info"].family_hash if entry.get("graph_info") and entry["graph_info"].parse_ok else "incomplete"
        for entry in entries
    ]
    batch_descriptor_keys = [
        entry["graph_info"].descriptor_key if entry.get("graph_info") and entry["graph_info"].parse_ok else "incomplete"
        for entry in entries
    ]
    batch_backbone_signatures = [_entry_backbone_signature(entry) for entry in entries]
    batch_cnn_signatures = [_entry_cnn_signature(entry) for entry in entries]
    batch_block_signatures = [_entry_block_signature(entry) for entry in entries]
    batch_backbone_block_signatures = [
        TuneRL._backbone_block_pair_key(backbone_signature, block_signature)
        for backbone_signature, block_signature in zip(batch_backbone_signatures, batch_block_signatures)
    ]
    scored_results = []

    for position, entry in enumerate(entries):
        index = int(entry["local_index"])
        completion_index = int(entry.get("global_index", index))
        TuneRL.code_logger.log_to_file("=" * 50)
        try:
            res = TuneRL.reward_task_reward_fn(
                entry["completion"],
                seed_accuracy_baseline=entry["seed_accuracy_baseline"],
                precomputed_eval_result=entry.get("precomputed_eval_result"),
                graph_info=entry.get("graph_info"),
                batch_graph_hashes=batch_graph_hashes,
                batch_family_hashes=batch_family_hashes,
                batch_descriptor_keys=batch_descriptor_keys,
                batch_backbone_signatures=batch_backbone_signatures,
                batch_cnn_signatures=batch_cnn_signatures,
                batch_block_signatures=batch_block_signatures,
                batch_backbone_block_signatures=batch_backbone_block_signatures,
                prompt_goal_tags=entry.get("prompt_goal_tags"),
                prompt_target_pattern=entry.get("prompt_target_pattern", ""),
                archive_snapshot_family_counts=archive_snapshot_family_counts,
                archive_snapshot_descriptor_counts=archive_snapshot_descriptor_counts,
                archive_snapshot_backbone_signature_counts=archive_snapshot_backbone_signature_counts,
                archive_snapshot_cnn_signature_counts=archive_snapshot_cnn_signature_counts,
                archive_snapshot_graph_counts=archive_snapshot_graph_counts,
                archive_snapshot_block_signature_counts=archive_snapshot_block_signature_counts,
                archive_snapshot_backbone_cnn_pair_counts=archive_snapshot_backbone_cnn_pair_counts,
                archive_snapshot_backbone_block_pair_counts=archive_snapshot_backbone_block_pair_counts,
                archive_snapshot_backbone_block_best_quality=archive_snapshot_backbone_block_best_quality,
                group_baseline_train_acc=group_context["group_baseline_train_acc"],
                group_baseline_reward_target_acc=group_context["group_baseline_reward_target_acc"],
                reward_batch_index=group_context["reward_batch_index"],
                reward_group_id=group_context["reward_group_id"],
                group_warmup=group_context["group_warmup"],
                completion_index=completion_index,
                batch_last_item=position == (len(entries) - 1),
            )
            res = TuneRL._attach_group_context(
                res,
                seed_accuracy_baseline=entry["seed_accuracy_baseline"],
                group_context=group_context,
            )
            dispatch_parts = []
            if res.get("worker_slot") is not None:
                dispatch_parts.append(f"worker_slot={res.get('worker_slot')}")
            if res.get("assigned_gpu") is not None:
                dispatch_parts.append(f"assigned_gpu={res.get('assigned_gpu')}")
            if res.get("worker_device") is not None:
                dispatch_parts.append(f"worker_device={res.get('worker_device')}")
            if dispatch_parts:
                TuneRL.code_logger.log_to_file(
                    f"[Reward Dispatch] rank={entry['rank']} batch_index={index}, " + ", ".join(dispatch_parts)
                )
            TuneRL._log_reward_failure_trace(entry, res)
            score = float(res.get("reward", -2.0))
        except TuneRL.PersistentEvalWorkerError:
            raise
        except Exception as exc:
            TuneRL.code_logger.log_to_file(f"Reward calculation failed at rank={entry['rank']} index={index}: {exc}")
            res = TuneRL._reward_failure_result(
                error=str(exc),
                seed_accuracy_baseline=entry["seed_accuracy_baseline"],
                group_context=group_context,
            )
            score = -1.0
        scored_results.append(
            {
                **entry,
                "result": res,
                "score": score,
            }
        )

    apply_batch_elite_bonuses(scored_results, group_context)
    for item in scored_results:
        item["score"] = float(item["result"].get("reward", item.get("score", -1.0)))
    return scored_results


def entries_from_records(records):
    return [_entry_from_record(record, index=index) for index, record in enumerate(records)]


def describe_code_sections(*, block_code: str, init_code: str, forward_code: str):
    graph_info = None
    if init_code and forward_code and "self.pattern" not in forward_code:
        try:
            graph_info = extract_graph_info(
                init_code,
                forward_code,
                legacy_patterns=SFTUtil.legacy_patterns,
            )
        except Exception:
            graph_info = None
    backbone_model_names = _extract_backbone_model_names(init_code)
    backbone_signature = _build_backbone_signature(backbone_model_names)
    block_signature = _block_signature_from_code(block_code)
    return {
        "graph_info": graph_info,
        "block_code": block_code,
        "init_code": init_code,
        "forward_code": forward_code,
        "backbone_model_names": backbone_model_names,
        "backbone_signature": backbone_signature,
        "block_signature": block_signature,
        "cnn_signature": (
            str(getattr(graph_info, "cnn_signature", "") or "")
            if graph_info is not None
            else "incomplete_cnn"
        ),
        "cnn_expr": (
            str(getattr(graph_info, "cnn_expr", "") or "")
            if graph_info is not None
            else "IncompleteCNN"
        ),
    }


def apply_batch_elite_bonuses(scored_results, group_context: Dict[str, Any]) -> None:
    _tunerl()._apply_batch_elite_bonuses(scored_results, group_context)


def finalize_scored_results(scored_results) -> None:
    _tunerl()._finalize_scored_results(scored_results)


def _extract_backbone_model_names(init_code: str) -> list[str]:
    matches: dict[str, str] = {}
    patterns = (
        r"self\.(backbone_[ab])\s*=\s*TorchVision\(\s*model\s*=\s*['\"]([^'\"]+)['\"]",
        r"self\.(backbone_[ab])\s*=\s*TorchVision\(\s*['\"]([^'\"]+)['\"]",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, init_code or ""):
            matches.setdefault(match.group(1), match.group(2))
    return [matches[name] for name in ("backbone_a", "backbone_b") if name in matches]


def _build_backbone_signature(backbone_model_names: list[str] | None) -> str:
    normalized = [
        str(name).strip()
        for name in list(backbone_model_names or [])
        if str(name).strip()
    ]
    normalized.sort()
    return " + ".join(normalized) if normalized else "unknown_backbone_pair"


def _block_signature_from_code(block_code: str) -> str:
    source = textwrap.dedent(str(block_code or "")).strip()
    if not source:
        return "incomplete_block"
    try:
        tree = ast.parse(source)
        payload = ast.dump(tree, annotate_fields=True, include_attributes=False)
    except Exception:
        payload = "\n".join(line.strip() for line in source.splitlines() if line.strip())
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _record_completion(record: Dict[str, Any]) -> str:
    return str(record.get("completion") or record.get("raw_completion") or "")


def _record_api_result(record: Dict[str, Any]) -> Dict[str, Any]:
    value = record.get("api_result")
    return value if isinstance(value, dict) else {}


def _record_seed_accuracy(record: Dict[str, Any]) -> float:
    TuneRL = _tunerl()
    api_result = _record_api_result(record)
    sources = (
        record,
        api_result,
        record.get("candidate") if isinstance(record.get("candidate"), dict) else {},
    )
    for source in sources:
        value = (
            source.get("seed_accuracy_baseline")
            if source.get("seed_accuracy_baseline") is not None
            else source.get("accuracy_baseline")
            if source.get("accuracy_baseline") is not None
            else source.get("accuracy")
        )
        if value is not None:
            return TuneRL._coerce_accuracy_baseline(value, context="replay record accuracy")
    return 0.10


def _entry_from_record(record: Dict[str, Any], *, index: int) -> Dict[str, Any]:
    TuneRL = _tunerl()
    completion = _record_completion(record)
    block_code, init_code, forward_code = extract_completion_blocks_strict(completion)
    section_info = describe_code_sections(
        block_code=block_code,
        init_code=init_code,
        forward_code=forward_code,
    )
    prompt = str(record.get("prompt") or "")
    prompt_goal_tags = TuneRL.extract_prompt_goal_tags(prompt)
    prompt_target_pattern = TuneRL.extract_prompt_target_pattern(prompt)
    return {
        "rank": 0,
        "local_index": index,
        "global_index": index,
        "completion": completion,
        "prompt": prompt,
        "graph_info": section_info.get("graph_info"),
        "backbone_model_names": section_info.get("backbone_model_names"),
        "backbone_signature": section_info.get("backbone_signature"),
        "cnn_signature": section_info.get("cnn_signature"),
        "prompt_goal_tags": prompt_goal_tags,
        "prompt_target_pattern": prompt_target_pattern,
        "goal_key": TuneRL.primary_goal_key(prompt_goal_tags, prompt_target_pattern),
        "seed_accuracy_baseline": _record_seed_accuracy(record),
        "precomputed_eval_result": dict(_record_api_result(record)),
    }


def _entry_backbone_model_names(entry: Dict[str, Any]) -> list[str]:
    backbone_names = list(entry.get("backbone_model_names") or [])
    if backbone_names:
        return backbone_names
    _, init_code, _ = extract_completion_blocks_strict(str(entry.get("completion") or ""))
    return _extract_backbone_model_names(init_code)


def _entry_backbone_signature(entry: Dict[str, Any]) -> str:
    signature = str(entry.get("backbone_signature") or "").strip()
    if signature:
        return signature
    return _build_backbone_signature(_entry_backbone_model_names(entry))


def _entry_cnn_signature(entry: Dict[str, Any]) -> str:
    signature = str(entry.get("cnn_signature") or "").strip()
    if signature:
        return signature
    graph_info = entry.get("graph_info")
    if graph_info is not None:
        signature = str(getattr(graph_info, "cnn_signature", "") or "").strip()
        if signature:
            return signature
    return "incomplete_cnn"


def _entry_block_signature(entry: Dict[str, Any]) -> str:
    signature = str(entry.get("block_signature") or "").strip()
    if signature:
        return signature
    block_code, init_code, forward_code = extract_completion_blocks_strict(str(entry.get("completion") or ""))
    if not _block_contributes_to_forward(init_code, forward_code):
        return "incomplete_block"
    return _block_signature_from_code(block_code)


def _block_contributes_to_forward(init_code: str, forward_code: str) -> bool:
    init_source = str(init_code or "")
    forward_source = str(forward_code or "")
    block_tokens = ("drop_conv3x3_block", "FractalUnit", "FractalBlock")
    if not any(token in init_source or token in forward_source for token in block_tokens):
        return False
    if "drop_conv3x3_block" in forward_source:
        return True
    if any(token in init_source for token in block_tokens) and "self.features" in init_source and "self.features" in forward_source:
        return True

    referenced_attrs = set(re.findall(r"self\.([A-Za-z_][A-Za-z0-9_]*)", forward_source))
    for line in init_source.splitlines():
        if not any(token in line for token in block_tokens):
            continue
        match = re.search(r"self\.([A-Za-z_][A-Za-z0-9_]*)", line)
        if match and match.group(1) in referenced_attrs:
            return True
    return False


def _invoke_eval_cfg_builder(eval_cfg_builder, **kwargs):
    if not callable(eval_cfg_builder):
        return None
    signature = inspect.signature(eval_cfg_builder)
    supported_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
        and signature.parameters[key].kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return eval_cfg_builder(**supported_kwargs)


def _build_batched_eval_specs(entries, *, group_context: Dict[str, Any]):
    TuneRL = _tunerl()
    eval_cfg_builder = TuneRL.reward_eval_cfg_builder()
    batched_eval_entries = []
    batched_eval_specs = []

    for entry in entries:
        if entry.get("precomputed_eval_result") is not None:
            continue

        completion = str(entry.get("completion", ""))
        graph_info = entry.get("graph_info")
        block_code, init_code, forward_code = extract_completion_blocks_strict(completion)
        if not block_code or not init_code or not forward_code:
            continue
        if "self.pattern" in forward_code or graph_info is None:
            continue

        pattern_override = graph_info.suggested_pattern_name if not graph_info.has_custom_pattern_name else ""
        final_code = TuneRL.reconstruct_code(completion, pattern_name_override=pattern_override)
        if not final_code:
            continue

        formal_input_shape = _formal_reward_input_shape()
        prm = {
            "lr": 0.01,
            "batch": 64,
            "dropout": 0.3,
            "momentum": 0.9,
            "transform": TuneRL.FORMAL_REWARD_TRANSFORM,
            "epoch": 1,
        }
        spec = {
            "code": final_code,
            "in_shape": formal_input_shape,
            "out_shape": (10,),
            "prm": prm,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "seed_accuracy_baseline": entry["seed_accuracy_baseline"],
            "reward_batch_index": group_context["reward_batch_index"],
            "completion_index": int(entry.get("global_index", entry["local_index"])),
            "batch_last_item": False,
        }
        if callable(eval_cfg_builder):
            spec["cfg"] = _invoke_eval_cfg_builder(
                eval_cfg_builder,
                stage_name=str(group_context.get("current_stage_name") or TuneRL.current_stage_name),
                in_shape=formal_input_shape,
                out_shape=(10,),
                prm=spec["prm"],
                cfg=None,
                device=spec["device"],
            )

        batched_eval_entries.append(entry)
        batched_eval_specs.append(spec)

    if batched_eval_specs:
        batched_eval_specs[-1]["batch_last_item"] = True

    return batched_eval_entries, batched_eval_specs


def _formal_reward_input_shape(batch: int = 1) -> tuple[int, int, int, int]:
    transform = str(_tunerl().FORMAL_REWARD_TRANSFORM)
    match = re.search(r"(?:^|_)norm_(\d+)(?:_|$)", transform)
    resize = 128
    if match:
        try:
            parsed = int(match.group(1))
        except (TypeError, ValueError):
            parsed = 128
        if parsed > 0:
            resize = parsed
    return (int(batch), 3, resize, resize)
