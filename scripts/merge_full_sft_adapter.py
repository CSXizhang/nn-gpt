#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _quant_config_has_4bit(config_dict: dict) -> bool:
    quant = config_dict.get("quantization_config")
    if not isinstance(quant, dict):
        return False
    return bool(quant.get("load_in_4bit") or quant.get("_load_in_4bit"))


def _load_config_dict(model_id: str) -> dict:
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    return cfg.to_dict()


def _dtype(label: str) -> torch.dtype:
    if label == "bf16":
        return torch.bfloat16
    if label == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {label}")


def merge_adapter(
    *,
    base_model: str,
    adapter: Path,
    output: Path,
    dtype: torch.dtype,
    overwrite: bool,
) -> None:
    if not adapter.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter}")
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output}")
        shutil.rmtree(output)

    base_config = _load_config_dict(base_model)
    if _quant_config_has_4bit(base_config):
        raise RuntimeError(
            "Refusing to merge into a 4-bit base model. Use the original full/bf16 base "
            "or a local base whose config has no quantization_config."
        )

    print(f"[merge-full-sft] base={base_model}")
    print(f"[merge-full-sft] adapter={adapter}")
    print(f"[merge-full-sft] output={output}")
    print(f"[merge-full-sft] dtype={dtype}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    model = model.merge_and_unload()

    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)

    saved_config = json.loads((output / "config.json").read_text())
    if _quant_config_has_4bit(saved_config):
        raise RuntimeError(f"Merged output still contains 4-bit quantization_config: {output / 'config.json'}")
    print("[merge-full-sft] complete")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge a full/bf16 SFT LoRA adapter into a non-quantized base model.")
    parser.add_argument("--base-model", default="deepseek-ai/deepseek-coder-6.7b-instruct")
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    merge_adapter(
        base_model=args.base_model,
        adapter=args.adapter.expanduser(),
        output=args.output.expanduser(),
        dtype=_dtype(args.dtype),
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
