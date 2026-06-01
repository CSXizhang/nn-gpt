#!/usr/bin/env python3
"""Generate A3 struct1 completions and score them without GRPO updates."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

import torch


def _set_default_env() -> None:
    os.environ.setdefault(
        "NNGPT_SFT_BASE_MODEL_ID",
        "/home/s471802/nn-gpt/out/llm/deepseek-ai/deepseek-coder-6.7b-instruct",
    )
    os.environ.setdefault(
        "NNGPT_SFT_TOKENIZER_ID",
        "/home/s471802/nn-gpt/parallel_runs/20260426_1905_main_resume_quality_diversity_std3/"
        "grpo_backbone_outputs_trainer/checkpoint-130",
    )
    os.environ.setdefault(
        "NNGPT_SFT_INIT_ADAPTER",
        "/home/s471802/nn-gpt/out/nngpt/llm/epoch/A3/deepseek-ai/deepseek-coder-6.7b-instruct",
    )
    os.environ.setdefault("NNGPT_SFT_LOAD_INITIAL_ADAPTER", "1")
    os.environ.setdefault("NNGPT_SFT_INITIAL_ADAPTER_MODE", "trainable")
    os.environ.setdefault("NNGPT_SFT_INITIAL_ADAPTER_DTYPE", "fp32")
    os.environ.setdefault("NNGPT_SFT_RL_NN_PREFIXES", "rl-bb-struct1")
    os.environ.setdefault("NNGPT_SFT_RL_PROMPT_MODE", "sft_aligned")
    os.environ.setdefault("NNGPT_SFT_MAX_PROMPT_LENGTH", "3500")
    os.environ.setdefault("NNGPT_SFT_MAX_COMPLETION_LENGTH", "1536")
    os.environ.setdefault("NNGPT_RL_RESUME_STAGE", "stage1_structure_explore")
    os.environ.setdefault("NNGPT_SFT_DATASET_LIMIT", "64")
    os.environ.setdefault("NNGPT_SFT_MODE", "split")
    os.environ.setdefault("NNGPT_SFT_REWARD_EXCLUDE_TRAIN_GPU", "1")


def _error_label(result: dict) -> str:
    error_type = result.get("error_type")
    if error_type:
        return str(error_type)
    error = str(result.get("error") or "")
    if not error:
        return "none"
    return error.split(":", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-prompts", type=int, default=4)
    parser.add_argument("--samples-per-prompt", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", required=True)
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

    model_source, tokenizer_source, _ = TuneRLSft.configure_sft_runtime()
    TuneRL.apply_resume_stage_override(os.environ["NNGPT_RL_RESUME_STAGE"], log_prefix="[A3 probe]")
    log_dir = Path(os.environ.get("NNGPT_SFT_LOG_DIR", "/tmp/nngpt_a3_direct_probe_log"))
    log_dir.mkdir(parents=True, exist_ok=True)
    TuneRL.code_logger = TuneRL.SimpleCodeLogger(str(log_dir))

    print("[A3 probe] loading tokenizer", flush=True)
    tokenizer = TrainerRuntime.load_tokenizer(tokenizer_source)
    print("[A3 probe] loading dataset", flush=True)
    dataset = TuneRL.load_reward_dataset(tokenizer)
    precision = TuneRL.best_mixed_precision()
    train_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[A3 probe] loading model on {train_device}", flush=True)
    model = TrainerRuntime.load_quantized_causal_lm(
        model_source=model_source,
        precision=precision,
        train_device=train_device,
        use_deepspeed=False,
    )
    model = TuneRL.prepare_model_for_kbit_training(model)
    TuneRL.align_generation_head_dtype(model, precision["torch_dtype"])
    print("[A3 probe] loading A3 adapter", flush=True)
    model = TrainerRuntime.load_trainable_initial_adapter(
        model,
        enabled=True,
        adapter_path=TuneRLSft.resolve_sft_init_adapter(),
        label="SFT",
        adapter_dtype=TuneRLSft.resolve_sft_initial_adapter_dtype("fp32", precision),
    )
    TuneRL.align_generation_head_dtype(model, precision["torch_dtype"])
    model.eval()
    print("[A3 probe] model ready", flush=True)

    max_prompt_length = int(os.environ["NNGPT_SFT_MAX_PROMPT_LENGTH"])
    max_completion_length = int(os.environ["NNGPT_SFT_MAX_COMPLETION_LENGTH"])
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with output_path.open("w", encoding="utf-8") as handle:
        for prompt_index in range(min(args.num_prompts, len(dataset))):
            row = dataset[prompt_index]
            prompt = row["prompt"]
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_prompt_length,
            )
            encoded = {key: value.to(train_device) for key, value in encoded.items()}
            prompt_len = int(encoded["input_ids"].shape[1])
            print(
                f"[A3 probe] generating prompt={prompt_index} prompt_tokens={prompt_len} "
                f"samples={args.samples_per_prompt}",
                flush=True,
            )
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=max_completion_length,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    top_k=50,
                    num_return_sequences=args.samples_per_prompt,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            for sample_index, output_ids in enumerate(generated):
                completion_ids = output_ids[prompt_len:]
                completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
                print(f"[A3 probe] scoring prompt={prompt_index} sample={sample_index}", flush=True)
                result = TuneRL.reward_task_reward_fn(
                    completion,
                    seed_accuracy_baseline=float(row["accuracy"]),
                    completion_index=len(rows),
                )
                record = {
                    "prompt_index": prompt_index,
                    "sample_index": sample_index,
                    "prompt_tokens": prompt_len,
                    "target_pattern": row.get("target_pattern"),
                    "reward": float(result.get("reward", -99.0)),
                    "built_ok": bool(result.get("built_ok")),
                    "forward_ok": bool(result.get("forward_ok")),
                    "forward_shape_ok": bool(result.get("forward_shape_ok")),
                    "backward_ok": bool(result.get("backward_ok")),
                    "loss_drop_ok": bool(result.get("loss_drop_ok")),
                    "reward_target_value": result.get("reward_target_value"),
                    "error_type": result.get("error_type"),
                    "error": result.get("error"),
                    "has_infer_call": "self.infer_dimensions_dynamically" in completion,
                    "has_torchvision": "TorchVision" in completion,
                    "has_backbone_literal": "Backbone(" in completion,
                    "has_model_literal": "Model(" in completion,
                    "completion": completion,
                }
                rows.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    f"[A3 probe] scored prompt={prompt_index} sample={sample_index} "
                    f"reward={record['reward']:.3f} built={record['built_ok']} "
                    f"infer={record['has_infer_call']} error={record['error']}",
                    flush=True,
                )

    rewards = [float(row["reward"]) for row in rows]
    summary = {
        "count": len(rows),
        "positive": sum(value > 0 for value in rewards),
        "reward_min": min(rewards) if rewards else None,
        "reward_max": max(rewards) if rewards else None,
        "reward_mean": sum(rewards) / len(rewards) if rewards else None,
        "built_ok": sum(row["built_ok"] for row in rows),
        "forward_ok": sum(row["forward_ok"] for row in rows),
        "forward_shape_ok": sum(row["forward_shape_ok"] for row in rows),
        "has_infer_call": sum(row["has_infer_call"] for row in rows),
        "has_torchvision": sum(row["has_torchvision"] for row in rows),
        "has_backbone_literal": sum(row["has_backbone_literal"] for row in rows),
        "has_model_literal": sum(row["has_model_literal"] for row in rows),
        "error_labels": Counter(_error_label(row) for row in rows).most_common(),
        "output": str(output_path),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
