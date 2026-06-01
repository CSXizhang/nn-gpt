"""Lightweight dependency stubs for local reward replay tests.

These stubs are opt-in and only intended for replaying rewards from
precomputed eval payloads on machines without the training stack installed.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path


def install() -> None:
    if getattr(install, "_installed", False):
        return

    root = Path(__file__).resolve().parents[1]

    torch_mod = types.ModuleType("torch")
    torch_mod.Tensor = object
    torch_mod.dtype = object
    torch_mod.float16 = "float16"
    torch_mod.bfloat16 = "bfloat16"
    torch_mod.float32 = "float32"
    torch_mod.device = lambda name: name
    torch_mod.is_floating_point = lambda _value: True

    def _decorator_passthrough(fn=None):
        if fn is None:
            return lambda inner: inner
        return fn

    torch_mod.no_grad = _decorator_passthrough
    torch_mod.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        is_bf16_supported=lambda: False,
        device_count=lambda: 0,
        empty_cache=lambda: None,
        set_device=lambda *_args, **_kwargs: None,
        current_device=lambda: 0,
        memory_allocated=lambda *_args, **_kwargs: 0,
        memory_reserved=lambda *_args, **_kwargs: 0,
        reset_peak_memory_stats=lambda *_args, **_kwargs: None,
    )
    torch_mod.distributed = types.SimpleNamespace(
        is_available=lambda: False,
        is_initialized=lambda: False,
    )
    nn_mod = types.ModuleType("torch.nn")
    nn_mod.Module = object
    nn_mod.Identity = lambda *args, **kwargs: object()
    functional_mod = types.ModuleType("torch.nn.functional")
    nn_mod.functional = functional_mod
    torch_mod.nn = nn_mod
    utils_mod = types.ModuleType("torch.utils")
    data_mod = types.ModuleType("torch.utils.data")
    data_mod.Dataset = object
    data_mod.DataLoader = object
    data_mod.Subset = object
    checkpoint_mod = types.ModuleType("torch.utils.checkpoint")
    utils_mod.data = data_mod
    utils_mod.checkpoint = checkpoint_mod
    torch_mod.utils = utils_mod
    sys.modules.setdefault("torch", torch_mod)
    sys.modules.setdefault("torch.nn", nn_mod)
    sys.modules.setdefault("torch.nn.functional", functional_mod)
    sys.modules.setdefault("torch.utils", utils_mod)
    sys.modules.setdefault("torch.utils.data", data_mod)
    sys.modules.setdefault("torch.utils.checkpoint", checkpoint_mod)

    transformers_mod = types.ModuleType("transformers")
    transformers_mod.AutoTokenizer = object
    transformers_mod.AutoModelForCausalLM = object
    transformers_mod.TrainingArguments = object
    transformers_mod.BitsAndBytesConfig = lambda *args, **kwargs: dict(kwargs)
    sys.modules.setdefault("transformers", transformers_mod)

    peft_mod = types.ModuleType("peft")
    peft_mod.LoraConfig = lambda *args, **kwargs: dict(kwargs)
    peft_mod.get_peft_model = lambda model, _config: model
    peft_mod.prepare_model_for_kbit_training = lambda model, *args, **kwargs: model
    peft_mod.PeftModel = types.SimpleNamespace(from_pretrained=lambda model, *_args, **_kwargs: model)
    sys.modules.setdefault("peft", peft_mod)

    trl_mod = types.ModuleType("trl")
    trl_trainer_mod = types.ModuleType("trl.trainer")
    trl_grpo_trainer_mod = types.ModuleType("trl.trainer.grpo_trainer")
    trl_grpo_trainer_mod.GRPOTrainer = object
    trl_grpo_config_mod = types.ModuleType("trl.trainer.grpo_config")
    trl_grpo_config_mod.GRPOConfig = object
    sys.modules.setdefault("trl", trl_mod)
    sys.modules.setdefault("trl.trainer", trl_trainer_mod)
    sys.modules.setdefault("trl.trainer.grpo_trainer", trl_grpo_trainer_mod)
    sys.modules.setdefault("trl.trainer.grpo_config", trl_grpo_config_mod)

    datasets_mod = types.ModuleType("datasets")
    datasets_mod.Dataset = object
    sys.modules.setdefault("datasets", datasets_mod)

    reward_mod = types.ModuleType("ab.gpt.util.Reward")

    class EvalConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    reward_mod.EvalConfig = EvalConfig
    reward_mod.FORMAL_MULTI_HORIZON_REWARD_TARGET_METRIC = "formal_multi_horizon_acc"
    reward_mod.PersistentEvalWorkerError = RuntimeError
    reward_mod.evaluate_code_and_reward = lambda *args, **kwargs: None
    reward_mod.evaluate_code_and_reward_batch = lambda *args, **kwargs: []
    reward_mod.get_distributed_runtime_info = lambda: {
        "distributed": False,
        "world_size": 1,
        "rank": 0,
        "raw_local_rank": 0,
        "local_rank": 0,
        "visible_gpu_count": 0,
        "visible_gpu_tokens": [],
        "train_gpu": None,
        "train_gpu_token": None,
    }
    reward_mod.get_eval_worker_diagnostics = lambda: None
    reward_mod.prewarm_eval_workers = lambda *args, **kwargs: None
    reward_mod.shutdown_eval_worker = lambda: None
    sys.modules["ab.gpt.util.Reward"] = reward_mod

    sftutil_mod = types.ModuleType("ab.gpt.util.SFTUtil")
    sftutil_mod.legacy_patterns = []
    sftutil_mod.available_backbones = ["resnet18", "mobilenet_v2", "efficientnet_b0"]
    sftutil_mod.open_discovery_goal_profiles = []
    sftutil_mod.open_discovery_prompt_template = ""
    sftutil_mod.open_discovery_rl_prompt_template = ""
    sftutil_mod.open_discovery_skeleton_code = _skeleton_code()
    sftutil_mod.skeleton_code = _skeleton_code()
    sftutil_mod.goal_profile_target_pattern = lambda profile: str(profile.get("target_pattern", ""))
    sftutil_mod.goal_tag_parser_cues = lambda tags: ", ".join(tags or [])
    sftutil_mod.extract_target_pattern_from_code = lambda _code: ""
    sftutil_mod.format_backbone_prompt = lambda *args, **kwargs: ""
    sys.modules["ab.gpt.util.SFTUtil"] = sftutil_mod

    util_mod = types.ModuleType("ab.gpt.util.Util")
    util_mod.extract_str = (
        lambda text, start, end: text.split(start, 1)[1].split(end, 1)[0]
        if start in text and end in text
        else ""
    )
    sys.modules["ab.gpt.util.Util"] = util_mod

    const_mod = types.ModuleType("ab.gpt.util.Const")
    const_mod.conf_dir = root / "ab" / "gpt" / "conf"
    const_mod.conf_train_dir = lambda *args, **kwargs: ""
    const_mod.conf_test_dir = lambda *args, **kwargs: ""
    const_mod.epoch_dir = lambda *args, **kwargs: ""
    const_mod.new_nn_file = "new_nn.py"
    const_mod.new_out_file = "new_out.txt"
    const_mod.synth_dir = lambda path: Path(path)
    sys.modules["ab.gpt.util.Const"] = const_mod

    ab_nn_mod = types.ModuleType("ab.nn")
    ab_nn_mod.__path__ = [str(root / "ab" / "nn")]
    ab_nn_api_mod = types.ModuleType("ab.nn.api")
    ab_nn_api_mod.data = lambda *args, **kwargs: None
    ab_nn_util_mod = types.ModuleType("ab.nn.util")
    ab_nn_util_mod.__path__ = [str(root / "ab" / "nn" / "util")]
    ab_nn_util_util_mod = types.ModuleType("ab.nn.util.Util")
    ab_nn_util_util_mod.create_file = lambda path, name, content: Path(path, name).write_text(
        str(content),
        encoding="utf-8",
    )
    sys.modules.setdefault("ab.nn", ab_nn_mod)
    sys.modules["ab.nn.api"] = ab_nn_api_mod
    sys.modules.setdefault("ab.nn.util", ab_nn_util_mod)
    sys.modules["ab.nn.util.Util"] = ab_nn_util_util_mod
    ab_nn_mod.api = ab_nn_api_mod
    ab_nn_util_mod.Util = ab_nn_util_util_mod

    install._installed = True


def _skeleton_code() -> str:
    return """
import torch
from torch import nn

def adaptive_pool_flatten(x):
    return x

def drop_conv3x3_block(in_channels, out_channels, stride=1, padding=1, bias=False, dropout_prob=0.0):
    return nn.Identity()

class TorchVision(nn.Module):
    def __init__(self, model=None, in_channels=3):
        super().__init__()

    def forward(self, x):
        return x

class FractalBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()

    def infer_dimensions_dynamically(self, n_classes):
        return None

    def forward(self, x: torch.Tensor, is_probing: bool = False) -> torch.Tensor:
        return x
""".strip()


if os.getenv("NNGPT_REWARD_REPLAY_LIGHTWEIGHT", "").strip().lower() in {"1", "true", "yes", "on"}:
    install()
