#!/usr/bin/env python3
"""Compare struct1 SFT adapter generation before and after merge."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch


DEFAULT_BASE_MODEL = "/home/s471802/nn-gpt/out/llm/deepseek-ai/deepseek-coder-6.7b-instruct"
DEFAULT_TOKENIZER = (
    "/home/s471802/nn-gpt/parallel_runs/20260426_1905_main_resume_quality_diversity_std3/"
    "grpo_backbone_outputs_trainer/checkpoint-130"
)
DEFAULT_A13_ADAPTER = "/home/s471802/nn-gpt/out/nngpt/llm/epoch/A13/deepseek-ai/deepseek-coder-6.7b-instruct"
DEFAULT_MERGED_MODEL = "/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-models/merged_struct1_a13_bf16_deepseek"


def _set_default_env() -> None:
    defaults = {
        "NNGPT_SFT_BASE_MODEL_ID": DEFAULT_BASE_MODEL,
        "NNGPT_SFT_TOKENIZER_ID": DEFAULT_TOKENIZER,
        "NNGPT_SFT_RL_NN_PREFIXES": "rl-bb-struct1",
        "NNGPT_SFT_RL_PROMPT_MODE": "sft_aligned",
        "NNGPT_SFT_MAX_PROMPT_LENGTH": "3500",
        "NNGPT_SFT_MAX_COMPLETION_LENGTH": "1536",
        "NNGPT_SFT_MODE": "split",
        "NNGPT_SFT_REWARD_EXCLUDE_TRAIN_GPU": "1",
        "NNGPT_SFT_REWARD_WORKERS": "1",
        "NNGPT_RL_RESUME_STAGE": "stage2_formal_explore",
        "NNGPT_RL_FORMAL_REWARD_EPOCHS": "1",
        "NNGPT_SFT_STOP_AFTER_FORWARD_XML": "1",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def _variant_spec(name: str, args: argparse.Namespace) -> dict[str, Any]:
    if name == "merged_bf16":
        return {
            "name": name,
            "model_source": args.merged_model,
            "adapter_path": "",
            "adapter_dtype_label": "",
        }
    if name == "unmerged_fp32":
        return {
            "name": name,
            "model_source": args.base_model,
            "adapter_path": args.adapter,
            "adapter_dtype_label": "fp32",
        }
    if name == "unmerged_bf16":
        return {
            "name": name,
            "model_source": args.base_model,
            "adapter_path": args.adapter,
            "adapter_dtype_label": "bf16",
        }
    if name == "unmerged_fp16":
        return {
            "name": name,
            "model_source": args.base_model,
            "adapter_path": args.adapter,
            "adapter_dtype_label": "fp16",
        }
    raise ValueError(f"Unknown variant: {name}")


def _completion_error_label(result: dict[str, Any]) -> str:
    stage = str(result.get("error_stage") or "none")
    error_type = str(result.get("error_type") or "none")
    message = str(result.get("error") or "").split("\n", 1)[0]
    if message:
        message = message[:180]
    return f"{stage}|{error_type}|{message}"


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = [float(row["reward"]) for row in records]
    tests = [float(row["horizon1_test_acc"]) for row in records if row.get("horizon1_test_acc") is not None]
    return {
        "count": len(records),
        "reward_mean": sum(rewards) / len(rewards) if rewards else None,
        "reward_min": min(rewards) if rewards else None,
        "reward_max": max(rewards) if rewards else None,
        "positive_reward": sum(value > 0.0 for value in rewards),
        "built_ok": sum(bool(row.get("built_ok")) for row in records),
        "executable": sum(bool(row.get("executable_candidate")) for row in records),
        "formal_success": sum(bool(row.get("formal_success_candidate")) for row in records),
        "xml_exact": sum(bool(row.get("xml_tag_exact")) for row in records),
        "dual_backbone": sum(bool(row.get("dual_backbone_ok")) for row in records),
        "cpu_prevalidate_fail": sum(row.get("error_stage") == "cpu_prevalidate" for row in records),
        "preflight_forward_fail": sum(row.get("error_stage") == "preflight_forward" for row in records),
        "horizon1_test_mean": sum(tests) / len(tests) if tests else None,
        "horizon1_test_max": max(tests) if tests else None,
        "completion_tokens_mean": (
            sum(int(row["completion_tokens"]) for row in records) / len(records) if records else None
        ),
        "eos_count": sum(bool(row.get("has_eos")) for row in records),
        "top_errors": Counter(row.get("error_label") for row in records).most_common(8),
        "top_family": Counter(row.get("family_id") for row in records if row.get("family_id")).most_common(5),
        "top_backbone": Counter(row.get("backbone_signature") for row in records if row.get("backbone_signature")).most_common(5),
        "top_block": Counter(row.get("block_signature") for row in records if row.get("block_signature")).most_common(5),
    }


def _load_model(spec: dict[str, Any], precision: dict[str, Any], train_device: str):
    from ab.gpt import TuneRL, TuneRLSft
    from ab.gpt.rl_pipeline import trainer_runtime as TrainerRuntime

    model = TrainerRuntime.load_quantized_causal_lm(
        model_source=str(spec["model_source"]),
        precision=precision,
        train_device=train_device,
        use_deepspeed=False,
    )
    model = TuneRL.prepare_model_for_kbit_training(model)
    TuneRL.align_generation_head_dtype(model, precision["torch_dtype"])
    adapter_path = str(spec.get("adapter_path") or "")
    if adapter_path:
        adapter_dtype = TuneRLSft.resolve_sft_initial_adapter_dtype(str(spec["adapter_dtype_label"]), precision)
        model = TrainerRuntime.load_trainable_initial_adapter(
            model,
            enabled=True,
            adapter_path=adapter_path,
            label=spec["name"],
            adapter_dtype=adapter_dtype,
        )
        TuneRL.align_generation_head_dtype(model, precision["torch_dtype"])
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = True
    return model


def _generate_one(
    *,
    model,
    tokenizer,
    encoded: dict[str, torch.Tensor],
    prompt_len: int,
    seed: int,
    do_sample: bool,
    max_completion_length: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> tuple[str, int, bool]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_completion_length,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
        "stop_strings": ["</forward>"],
        "tokenizer": tokenizer,
    }
    if do_sample:
        kwargs.update({"temperature": temperature, "top_p": top_p, "top_k": top_k})
    with torch.no_grad():
        output = model.generate(**encoded, **kwargs)[0]
    completion_ids = output[prompt_len:]
    has_eos = bool((completion_ids == tokenizer.eos_token_id).any().item()) if tokenizer.eos_token_id is not None else False
    completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
    return completion, int(completion_ids.numel()), has_eos


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--adapter", default=DEFAULT_A13_ADAPTER)
    parser.add_argument("--merged-model", default=DEFAULT_MERGED_MODEL)
    parser.add_argument("--variants", default="unmerged_fp32,unmerged_bf16,merged_bf16")
    parser.add_argument("--num-prompts", type=int, default=3)
    parser.add_argument("--samples-per-prompt", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    _set_default_env()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    from ab.gpt import TuneRL, TuneRLSft
    from ab.gpt.rl_pipeline import trainer_runtime as TrainerRuntime

    if torch.cuda.is_available():
        TuneRLSft._configure_sft_gpu_role_env(torch.cuda.device_count())
        torch.cuda.set_device(0)

    TuneRL.apply_resume_stage_override(os.environ["NNGPT_RL_RESUME_STAGE"], log_prefix="[merge-ab probe]")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NNGPT_SFT_LOG_DIR", str(out_dir / "reward_logs"))
    Path(os.environ["NNGPT_SFT_LOG_DIR"]).mkdir(parents=True, exist_ok=True)
    TuneRL.code_logger = TuneRL.SimpleCodeLogger(os.environ["NNGPT_SFT_LOG_DIR"])

    precision = TuneRL.best_mixed_precision()
    train_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = TrainerRuntime.load_tokenizer(args.tokenizer)
    dataset = TuneRLSft.load_rl_dataset_sft(tokenizer)
    max_prompt_length = int(os.environ["NNGPT_SFT_MAX_PROMPT_LENGTH"])
    max_completion_length = int(os.environ["NNGPT_SFT_MAX_COMPLETION_LENGTH"])
    prompt_count = min(args.num_prompts, len(dataset))
    prompts: list[dict[str, Any]] = []
    for prompt_index in range(prompt_count):
        row = dataset[prompt_index]
        encoded_cpu = tokenizer(
            row["prompt"],
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_length,
        )
        prompts.append(
            {
                "prompt_index": prompt_index,
                "row": row,
                "encoded_cpu": encoded_cpu,
                "prompt_tokens": int(encoded_cpu["input_ids"].shape[1]),
            }
        )

    all_summary: dict[str, Any] = {
        "args": vars(args),
        "precision": {key: str(value) for key, value in precision.items()},
        "prompt_tokens": [prompt["prompt_tokens"] for prompt in prompts],
        "variants": {},
    }
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]

    for variant_name in variants:
        spec = _variant_spec(variant_name, args)
        print(f"[merge-ab probe] loading variant={variant_name} source={spec['model_source']}", flush=True)
        variant_start = time.time()
        TuneRL.reset_reward_runtime_state()
        TuneRL.apply_resume_stage_override(os.environ["NNGPT_RL_RESUME_STAGE"], log_prefix=f"[merge-ab probe:{variant_name}]")
        model = _load_model(spec, precision, train_device)
        records: list[dict[str, Any]] = []
        output_jsonl = out_dir / f"{variant_name}.jsonl"
        with output_jsonl.open("w", encoding="utf-8") as handle:
            for prompt in prompts:
                row = prompt["row"]
                encoded = {key: value.to(train_device) for key, value in prompt["encoded_cpu"].items()}
                prompt_len = int(prompt["prompt_tokens"])
                generations = [("greedy", 0, False)]
                generations.extend(("sample", sample_index, True) for sample_index in range(args.samples_per_prompt))
                for mode, sample_index, do_sample in generations:
                    seed = args.seed + prompt["prompt_index"] * 1000 + sample_index
                    completion, completion_tokens, has_eos = _generate_one(
                        model=model,
                        tokenizer=tokenizer,
                        encoded=encoded,
                        prompt_len=prompt_len,
                        seed=seed,
                        do_sample=do_sample,
                        max_completion_length=max_completion_length,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                    )
                    result = TuneRLSft.sft_reward_fn(
                        completion,
                        seed_accuracy_baseline=float(row["accuracy"]),
                        completion_index=len(records),
                    )
                    raw = result.get("raw_extraction") or {}
                    horizons = result.get("formal_horizon_test_acc") or {}
                    record = {
                        "variant": variant_name,
                        "prompt_index": prompt["prompt_index"],
                        "generation_mode": mode,
                        "sample_index": sample_index,
                        "prompt_tokens": prompt_len,
                        "completion_tokens": completion_tokens,
                        "has_eos": has_eos,
                        "target_pattern": row.get("target_pattern"),
                        "seed_accuracy_baseline": float(row["accuracy"]),
                        "reward": float(result.get("reward", -99.0)),
                        "built_ok": bool(result.get("built_ok")),
                        "forward_shape_ok": bool(result.get("forward_shape_ok")),
                        "backward_ok": bool(result.get("backward_ok")),
                        "loss_drop_ok": bool(result.get("loss_drop_ok")),
                        "executable_candidate": bool(result.get("executable_candidate")),
                        "discovery_candidate": bool(result.get("discovery_candidate")),
                        "formal_success_candidate": bool(result.get("formal_success_candidate")),
                        "reward_target_value": result.get("reward_target_value"),
                        "horizon1_test_acc": horizons.get("1") if isinstance(horizons, dict) else None,
                        "error_stage": result.get("error_stage"),
                        "error_type": result.get("error_type"),
                        "error": result.get("error"),
                        "error_label": _completion_error_label(result),
                        "xml_tag_exact": raw.get("xml_tag_exact"),
                        "dual_backbone_ok": raw.get("dual_backbone_ok"),
                        "candidate_line_count": raw.get("candidate_line_count"),
                        "family_id": result.get("family_id"),
                        "backbone_signature": result.get("backbone_signature"),
                        "cnn_signature": result.get("cnn_signature"),
                        "block_signature": result.get("block_signature"),
                        "completion_tail": completion[-800:],
                    }
                    records.append(record)
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    print(
                        f"[merge-ab probe] {variant_name} prompt={prompt['prompt_index']} "
                        f"{mode}:{sample_index} reward={record['reward']:.4f} "
                        f"exec={record['executable_candidate']} formal={record['formal_success_candidate']} "
                        f"err={record['error_label'][:120]}",
                        flush=True,
                    )
        summary = _summarize(records)
        summary["elapsed_seconds"] = time.time() - variant_start
        summary["jsonl"] = str(output_jsonl)
        all_summary["variants"][variant_name] = summary
        (out_dir / f"{variant_name}.summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({variant_name: summary}, indent=2, ensure_ascii=False), flush=True)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    (out_dir / "summary.json").write_text(
        json.dumps(all_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(all_summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
