import ast
import csv
from datetime import timedelta
import hashlib
import inspect
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
import warnings


_RL_FILTERED_LOG_PATTERNS = (
    "Skipping import of cpp extensions due to incompatible torch version",
    "github.com/pytorch/ao/issues/2919",
)


class _RLFilteredStream:
    def __init__(self, wrapped) -> None:
        self._wrapped = wrapped
        self._buffer = ""

    def write(self, text):
        text = str(text)
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._write_line(line + "\n")
        return len(text)

    def flush(self):
        if self._buffer:
            self._write_line(self._buffer)
            self._buffer = ""
        return self._wrapped.flush()

    def _write_line(self, text: str) -> None:
        normalized = " ".join(text.split())
        if all(pattern in normalized for pattern in _RL_FILTERED_LOG_PATTERNS):
            return
        self._wrapped.write(text)

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


def _install_rl_runtime_noise_filters() -> None:
    if getattr(_install_rl_runtime_noise_filters, "_installed", False):
        return
    warnings.filterwarnings(
        "ignore",
        message=r".*Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers\. Use `HF_HOME` instead\..*",
        category=FutureWarning,
    )
    sys.stdout = _RLFilteredStream(sys.stdout)
    sys.stderr = _RLFilteredStream(sys.stderr)
    _install_rl_runtime_noise_filters._installed = True


_install_rl_runtime_noise_filters()

import torch
from peft import prepare_model_for_kbit_training
from trl.trainer.grpo_trainer import GRPOTrainer
from trl.trainer.grpo_config import GRPOConfig
from ab.gpt.util.generation_dtype import align_generation_head_dtype
import ab.gpt.rl_pipeline.trainer_runtime as TrainerRuntime
import ab.gpt.rl_pipeline.stage_state as StageState
import ab.gpt.rl_pipeline.reward_payload as RewardPayload
from ab.gpt.rl_pipeline.reward_task import RewardTask
import ab.gpt.util.SFTUtil as SFTUtil
from ab.gpt.util.ArchDiscovery import (
    extract_graph_info,
    normalize_pattern_name,
)
from ab.gpt.util.Const import conf_train_dir, conf_test_dir, epoch_dir, new_nn_file, synth_dir, new_out_file
from ab.nn.util.Util import create_file
from ab.gpt.util.Reward import (
    EvalConfig,
    FORMAL_MULTI_HORIZON_REWARD_TARGET_METRIC,
    PersistentEvalWorkerError,
    evaluate_code_and_reward,
    evaluate_code_and_reward_batch,
    get_distributed_runtime_info,
    get_eval_worker_diagnostics,
    prewarm_eval_workers,
    shutdown_eval_worker,
)
import ab.nn.api as api

import textwrap
import shutil
import json
from pathlib import Path
from dataclasses import dataclass, asdict

from ab.gpt.util.simple_logger import SimpleCodeLogger
import ab.gpt.util.training_runtime as TrainingRuntime
from typing import Tuple, Any, List, Dict, Optional, Set
from collections import Counter, deque

train_reference_stats: Dict[str, int] = {}


# ===== Configuration Options =====
base_model = "ABrain/NNGPT-Backbone-deepseek-coder-6.7b-instruct" # 使用新的 Backbone 模型
tokenizer_source = base_model
LOAD_EXISTING_MODEL = False  # Model is already merged
SAVED_MODEL_PATH = "rl_backbone_model" 
active_reward_task: Optional[RewardTask] = None
B_index = 0
GROUP_BATCH_SIZE = 20
GROUP_IMPROVEMENT_DELTA = 0.003
BEST_GROUP_REFRESH_DELTA = 0.0015
GOAL_REFRESH_DELTA = 0.0015
NON_IMPROVING_REWARD_CAP = 0.04
FORMAL_REWARD_TRANSFORM = "norm_128_flip"


def register_reward_task(task: Optional[RewardTask]) -> None:
    global active_reward_task
    active_reward_task = task
    from ab.gpt.rl_pipeline import backbone_reward_runtime as BackboneRewardRuntime

    BackboneRewardRuntime.configure_runtime_services(TuneRLBackboneRuntimeServices())


def current_reward_task() -> Optional[RewardTask]:
    return active_reward_task


def _reward_task_callable(name: str, default):
    task = current_reward_task()
    method = getattr(task, name, None) if task is not None else None
    return method if callable(method) else default


class TuneRLBackboneRuntimeServices:
    @property
    def code_logger(self):
        return code_logger

    @property
    def current_stage_name(self) -> str:
        return current_stage_name

    @property
    def prev_closed_group_mean_reward_target_acc(self) -> Optional[float]:
        return prev_closed_group_mean_reward_target_acc

    @property
    def best_closed_group_mean_reward_target_acc(self) -> Optional[float]:
        return best_closed_group_mean_reward_target_acc

    @property
    def best_closed_group_mean_train_acc(self) -> Optional[float]:
        return best_closed_group_mean_train_acc

    @property
    def best_closed_group_mean_test_acc(self) -> Optional[float]:
        return best_closed_group_mean_test_acc

    @property
    def best_reward_target_by_goal(self) -> Dict[str, float]:
        return best_reward_target_by_goal

    @property
    def archive_index(self) -> int:
        return B_index

    @property
    def persistent_eval_worker_error(self):
        return PersistentEvalWorkerError

    def set_archive_index(self, value: int) -> None:
        global B_index
        B_index = int(value)

    def current_generation_total(self) -> int:
        return StageState.current_generation_total(sys.modules[__name__])

    def record_generation_event(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return StageState.record_generation_event(sys.modules[__name__], payload)

    def close_reward_group_if_needed(self) -> Optional[Dict[str, Any]]:
        return StageState.close_reward_group_if_needed(sys.modules[__name__])

    def evaluate_reward_code(self, *args, **kwargs):
        return evaluate_reward_code(*args, **kwargs)

    def evaluate_reward_code_batch(self, specs):
        return evaluate_reward_code_batch(specs)

    def reward_eval_cfg_builder(self):
        return reward_eval_cfg_builder()

    def reward_run_epoch_dir(self, *args):
        return reward_run_epoch_dir(*args)

    def record_current_group_trainable_sample(self, goal_key: str, res: Dict[str, Any], graph_info) -> None:
        _record_current_group_trainable_sample(goal_key, res, graph_info)

    def training_context_guidance(self, summary: Dict[str, Any]) -> str:
        return _training_context_guidance(summary)

    def summarize_stage_training_context(self, stage_name: str, *, window_size: int = 50) -> Dict[str, Any]:
        return summarize_stage_training_context(stage_name, window_size=window_size)

    def update_current_group_metrics(self, results: List[Dict[str, Any]]) -> None:
        update_current_group_metrics(results)

    def extract_seed_context(self, kwargs: Dict[str, Any], expected_count: int):
        return require_sample_accuracy_baselines(kwargs, expected_count)

    def extract_completion_blocks(self, completion: str) -> Tuple[str, str, str]:
        return _reward_task_callable(
            "extract_completion_blocks",
            _backbone_reward_runtime().extract_completion_blocks,
        )(completion)

    def group_context_fields(self) -> Dict[str, Any]:
        return reward_task_group_context_fields()

    def bootstrap_trainset_reference_library(self, data) -> None:
        bootstrap_trainset_reference_library(data)

    def get_prompt_feedback_state(self) -> Dict[str, Any]:
        return get_prompt_feedback_state()

    def distributed_rank(self) -> int:
        return _distributed_rank()

    def env_int(self, name: str, default: int) -> int:
        return env_int(name, default)

    def reward_task_reward_fn(self, *args, **kwargs):
        return reward_task_reward_fn(*args, **kwargs)

    def attach_group_context(self, res: Dict[str, Any], *, seed_accuracy_baseline: float, group_context: Dict[str, Any]) -> Dict[str, Any]:
        return _attach_group_context(res, seed_accuracy_baseline=seed_accuracy_baseline, group_context=group_context)

    def log_reward_failure_trace(self, entry: Dict[str, Any], res: Dict[str, Any]) -> None:
        _log_reward_failure_trace(entry, res)

    def reward_failure_result(self, *, error: str, seed_accuracy_baseline: float, group_context: Dict[str, Any]) -> Dict[str, Any]:
        return _reward_failure_result(
            error=error,
            seed_accuracy_baseline=seed_accuracy_baseline,
            group_context=group_context,
        )


def reward_model_source() -> str:
    task = current_reward_task()
    value = getattr(task, "model_source", None) if task is not None else None
    return str(value or base_model)


def reward_tokenizer_source() -> str:
    task = current_reward_task()
    value = getattr(task, "tokenizer_source", None) if task is not None else None
    return str(value or tokenizer_source or reward_model_source())


def reward_load_existing_model() -> bool:
    task = current_reward_task()
    value = getattr(task, "load_existing_model", None) if task is not None else None
    return bool(LOAD_EXISTING_MODEL if value is None else value)


def reward_saved_model_path() -> str:
    task = current_reward_task()
    value = getattr(task, "saved_model_path", None) if task is not None else None
    return str(value or SAVED_MODEL_PATH)


def _formal_reward_resize(default: int = 128) -> int:
    match = re.search(r"(?:^|_)norm_(\d+)(?:_|$)", FORMAL_REWARD_TRANSFORM)
    if not match:
        return int(default)
    try:
        value = int(match.group(1))
    except (TypeError, ValueError):
        return int(default)
    return value if value > 0 else int(default)


def _formal_reward_input_shape(batch: int = 1) -> Tuple[int, int, int, int]:
    resize = _formal_reward_resize()
    return (int(batch), 3, resize, resize)


BACKBONE_BASELINE_MIN_ARCHIVE_SAMPLES = 3
SAVE_DUPLICATE_BACKBONE_CNN_DELTA = 0.002
BATCH_ELITE_SOFT_BONUSES = (0.02, 0.015, 0.01, 0.005, 0.0)
BATCH_ELITE_IMPROVING_BONUSES = (0.04, 0.03, 0.02, 0.01, 0.0)
STRUCTURE_MACRO_BONUS = 0.04
STRUCTURE_MULTI_STAGE_BONUS = 0.03
STRUCTURE_MOTIF_BONUS = 0.02
STRUCTURE_BATCH_DIVERSITY_BONUS = 0.03
STRUCTURE_NON_DOMINANT_FAMILY_BONUS = 0.02
STRUCTURE_ARCHIVE_RARITY_STRONG_BONUS = 0.03
STRUCTURE_ARCHIVE_RARITY_MEDIUM_BONUS = 0.02
STRUCTURE_ARCHIVE_RARITY_LIGHT_BONUS = 0.01
REPEAT_FAMILY_PENALTY = -0.05
PLAIN_FUSE_PENALTY = -0.10
PLAIN_DUAL_BACKBONE_FUSE_PENALTY = -0.28
TARGET_STRUCTURE_DEAD_BLOCK_PENALTY = -0.80
TARGET_STRUCTURE_DUAL_BACKBONE_PENALTY = -0.60
TARGET_STRUCTURE_PATH_PENALTY = -0.05
TARGET_STRUCTURE_PARSE_PENALTY = -0.30
TARGET_STRUCTURE_PENALTY_FLOOR = -1.00
TARGET_STRUCTURE_MATCH_BONUS = 0.20
NO_PROGRESS_PENALTY = -0.06
GOAL_REFRESH_BONUS = 0.08
GOAL_MATCH_REWARD_SCALE = 0.12
TRAINSET_NOVEL_FAMILY_BONUS = 0.04
TRAINSET_NOVEL_GRAPH_BONUS = 0.02
GENERALIZATION_GAP_TOLERANCE = 0.02
GENERALIZATION_PENALTY_SCALE = 2.0
GENERALIZATION_PENALTY_CAP = -0.20
REWARD_TARGET_METRIC = "frozen_test_acc"
FEEDBACK_GRAPH_EXPR_MAX_CHARS = 160
FEEDBACK_SUMMARY_MAX_CHARS = 240
FEEDBACK_SUMMARY_LIMIT = 2
RL_DEEPSPEED_DEFAULT_CONFIG = str(Path(__file__).resolve().parent / "conf" / "DeepSpeedSftGrpo.json")
STAGE1_STRUCTURE_EXPLORE = "stage1_structure_explore"
STAGE2_FORMAL_EXPLORE = "stage2_formal_explore"
STAGE3_FORMAL_OPTIMIZE = "stage3_formal_optimize"
RL_STAGE_ORDER = (
    STAGE1_STRUCTURE_EXPLORE,
    STAGE2_FORMAL_EXPLORE,
    STAGE3_FORMAL_OPTIMIZE,
)
RL_STAGE_TO_INDEX = {
    stage_name: index
    for index, stage_name in enumerate(RL_STAGE_ORDER, start=1)
}
STAGE_REFERENCE_MIN_GROUPS = {
    STAGE1_STRUCTURE_EXPLORE: 4,
    STAGE2_FORMAL_EXPLORE: 5,
    STAGE3_FORMAL_OPTIMIZE: 0,
}
STAGE1_GATE_WINDOW_GENERATIONS = 1600
STAGE2_GATE_WINDOW_GENERATIONS = 4000
RECOVERY_GATE_WINDOW_GENERATIONS = 2000
STAGE1_EXECUTABLE_STABLE_WINDOW_GENERATIONS = 320
STAGE1_EXECUTABLE_STABLE_MIN_GROUPS = 6
STAGE1_EXECUTABLE_STABLE_MIN_RATE = 0.95
STAGE1_TRAINABLE_STABLE_MIN_RATE = 0.30
STAGE1_PROMOTION_MIN_GROUPS = 20
STAGE1_GATE_EXECUTABLE_MIN = 96
STAGE1_GATE_TRAINABLE_MIN = 480
STAGE1_GATE_DISCOVERY_MIN = 8
STAGE1_GATE_UNIQUE_DISCOVERY_FAMILIES_MIN = 6
STAGE1_FORCE_PROMOTION_EXECUTABLE_MIN = 800
STAGE1_FORCE_PROMOTION_TRAINABLE_MIN = 480
STAGE1_FORCE_PROMOTION_DISCOVERY_MIN = 8
STAGE1_FORCE_PROMOTION_UNIQUE_DISCOVERY_FAMILIES_MIN = 6
STAGE2_GATE_MIN_REWARD_TARGET = 0.90
STAGE2_GATE_MIN_TARGET_COUNT = 16
STAGE2_GATE_MIN_UNIQUE_TARGET_FAMILIES = 6
STAGE2_GATE_IMPROVING_GROUPS_REQUIRED = 2
STAGE2_GATE_MAX_DOMINANT_DESCRIPTOR_SHARE = 0.50
STAGE_RECOVERY_DOMINANT_SHARE_THRESHOLD = 0.55
STAGE_RECOVERY_NEW_DISCOVERY_FAMILIES_MAX = 1
STAGE_RECOVERY_RELEASE_GENERATIONS = 2000
STAGE_RECOVERY_RELEASE_DISCOVERY_FAMILIES = 4
MAX_STAGE_SAMPLE_HISTORY = 24000
MAX_STAGE_GROUP_HISTORY = 512
TRAINING_CONTEXT_WINDOW = 50
TRAINING_CONTEXT_MIN_POINTS = 8
STATIC_STAGE_REWARD_TARGET_METRIC = "stage1_static_score"
FORMAL_STAGE_REWARD_TARGET_METRIC = FORMAL_MULTI_HORIZON_REWARD_TARGET_METRIC
FORMAL_SUCCESS_SIGNAL_BONUS = 0.08
STAGE1_EXECUTABLE_BONUS = 0.10
STAGE1_DISCOVERY_FAMILY_BONUS = 0.42
STAGE1_DISCOVERY_GRAPH_BONUS = 0.20
STAGE1_STATIC_BASE_SCORE = 0.04
STAGE1_GOAL_MATCH_SCALE = 0.10
STAGE1_DISCOVERY_MIN_GOAL_HIT_RATE = 1.0 / 3.0
STAGE1_ZERO_GOAL_HIT_PENALTY = 0.0
STAGE1_LOW_GOAL_HIT_PENALTY = 0.0
STAGE1_STRUCTURE_GROUP_SCALE = 1.45
STAGE1_STRUCTURE_ARCHIVE_SCALE = 1.85
STAGE1_NON_DISCOVERY_EXECUTABLE_PENALTY = 0.0
STAGE1_ARCHIVE_REPEAT_STEP_PENALTY = -0.05
STAGE1_ARCHIVE_REPEAT_MAX_PENALTY = -0.30
STAGE1_BATCH_REPEAT_STEP_PENALTY = -0.05
STAGE1_BATCH_REPEAT_MAX_PENALTY = -0.24
STAGE1_DOMINANT_FAMILY_PENALTY = -0.08
STAGE1_PLAIN_PARALLEL_PENALTY = -0.10
STAGE1_PLAIN_PARALLEL_WARMUP_PENALTY = -0.03
STAGE1_DESCRIPTOR_BATCH_UNIQUE_BONUS = 0.12
STAGE1_GRAPH_BATCH_UNIQUE_BONUS = 0.05
STAGE1_DESCRIPTOR_ARCHIVE_NOVEL_BONUS = 0.03
STAGE1_DESCRIPTOR_BATCH_REPEAT_STEP_PENALTY = -0.04
STAGE1_DESCRIPTOR_BATCH_REPEAT_MAX_PENALTY = -0.12
STAGE1_DESCRIPTOR_ARCHIVE_REPEAT_STEP_PENALTY = -0.02
STAGE1_DESCRIPTOR_ARCHIVE_REPEAT_MAX_PENALTY = -0.08
STAGE1_GRAPH_BATCH_REPEAT_STEP_PENALTY = -0.06
STAGE1_GRAPH_BATCH_REPEAT_MAX_PENALTY = -0.18
STAGE1_ZERO_GOAL_HIT_REWARD_CAP = 1.0
STAGE1_LOW_GOAL_HIT_REWARD_CAP = 1.0
STAGE1_PLAIN_PARALLEL_REWARD_CAP = 1.0
STAGE1_OFF_TARGET_PLAIN_PARALLEL_REWARD_CAP = 1.0
STAGE1_REPEATED_BLOCK_REWARD_CAP = 0.24
STAGE23_DESCRIPTOR_BATCH_UNIQUE_BONUS = 0.03
STAGE23_DESCRIPTOR_ARCHIVE_NOVEL_BONUS = 0.02
STAGE23_NON_DOMINANT_DESCRIPTOR_BONUS = 0.06
STAGE23_DESCRIPTOR_BATCH_REPEAT_STEP_PENALTY = -0.015
STAGE23_DESCRIPTOR_BATCH_REPEAT_MAX_PENALTY = -0.05
STAGE23_DESCRIPTOR_ARCHIVE_REPEAT_STEP_PENALTY = -0.006
STAGE23_DESCRIPTOR_ARCHIVE_REPEAT_MAX_PENALTY = -0.03
STAGE23_GLOBAL_DESCRIPTOR_ARCHIVE_NOVEL_BONUS = 0.08
STAGE23_GLOBAL_DESCRIPTOR_ARCHIVE_REPEAT_MAX_PENALTY = -0.025
STAGE23_GLOBAL_DESCRIPTOR_ARCHIVE_REPEAT_WINDOW = 32
STAGE23_DOMINANT_DESCRIPTOR_SOFT_SHARE = 0.45
STAGE23_DOMINANT_DESCRIPTOR_STRONG_SHARE = 0.60
STAGE23_DOMINANT_DESCRIPTOR_REPEAT_PENALTY = -0.03
STAGE23_DOMINANT_DESCRIPTOR_REPEAT_STRONG_PENALTY = -0.05
STAGE23_CNN_BATCH_UNIQUE_BONUS = 0.07
STAGE23_CNN_ARCHIVE_NOVEL_BONUS = 0.05
STAGE23_CNN_BATCH_REPEAT_STEP_PENALTY = -0.02
STAGE23_CNN_BATCH_REPEAT_MAX_PENALTY = -0.07
STAGE23_CNN_ARCHIVE_REPEAT_STEP_PENALTY = -0.008
STAGE23_CNN_ARCHIVE_REPEAT_MAX_PENALTY = -0.04
STAGE23_GLOBAL_CNN_ARCHIVE_NOVEL_BONUS = 0.12
STAGE23_GLOBAL_CNN_ARCHIVE_REPEAT_MAX_PENALTY = -0.06
STAGE23_GLOBAL_CNN_ARCHIVE_REPEAT_WINDOW = 32
STAGE23_BLOCK_BATCH_UNIQUE_BONUS = 0.06
STAGE23_BLOCK_BATCH_REPEAT_STEP_PENALTY = -0.015
STAGE23_BLOCK_BATCH_REPEAT_MAX_PENALTY = -0.05
STAGE23_BLOCK_ARCHIVE_NOVEL_BONUS = 0.08
STAGE23_BLOCK_ARCHIVE_REPEAT_MAX_PENALTY = -0.08
STAGE23_BLOCK_ARCHIVE_REPEAT_WINDOW = 16
STAGE23_REPEATED_BLOCK_REWARD_CAP = 2.0
STAGE23_POSITIVE_NOVELTY_ACC_THRESHOLD = 0.90
STAGE23_EARLY_LOCAL_COMPETITION_GENERATIONS = 240
STAGE23_EARLY_CELL_REPEAT_REWARD_CAP = 0.0
STAGE23_NEW_CELL_BONUS = 0.04
STAGE23_CELL_IMPROVEMENT_DELTA = 0.003
STAGE23_CELL_IMPROVEMENT_BONUS = 0.08
STAGE23_DUPLICATE_LOW_ACC_THRESHOLD = 0.92
STAGE23_DUPLICATE_LOW_ACC_REWARD_CAP = 0.03
STAGE23_HIGH_ACC_BONUS_THRESHOLD = 0.92
STAGE23_HIGH_ACC_BONUS = 0.08
STAGE23_HIGH_ACC_STRONG_THRESHOLD = 0.925
STAGE23_HIGH_ACC_STRONG_BONUS = 0.12
STAGE23_HIGH_ACC_ELITE_THRESHOLD = 0.93
STAGE23_HIGH_ACC_ELITE_BONUS = 0.18
STAGE23_NON_DOMINANT_CNN_BONUS = 0.08
STAGE23_DOMINANT_CNN_SOFT_SHARE = 0.45
STAGE23_DOMINANT_CNN_STRONG_SHARE = 0.65
STAGE23_DOMINANT_CNN_REPEAT_PENALTY = -0.04
STAGE23_DOMINANT_CNN_REPEAT_STRONG_PENALTY = -0.06
STAGE23_GLOBAL_CNN_REPEAT_PENALTY = -0.08
STAGE23_GLOBAL_CNN_REPEAT_STRONG_PENALTY = -0.12
STAGE23_DEAD_BLOCK_PENALTY = -0.04
STAGE23_STRUCTURE_ARCHIVE_RARITY_CAP = 0.03
STAGE2_DENSE_SCALE = 0.50
STAGE2_PREV_GROUP_SCALE = 0.20
STAGE2_BEST_GROUP_SCALE = 0.20
STAGE2_GLOBAL_BASELINE_BLEND = 0.20
STAGE2_BACKBONE_PREV_GROUP_SCALE = 0.25
STAGE2_BACKBONE_BEST_GROUP_SCALE = 0.25
STAGE2_GOAL_BEST_SCALE = 0.70
STAGE2_GOAL_MATCH_SCALE = 0.85
STAGE2_STRUCTURE_SCALE = 1.40
STAGE2_REPEAT_FAMILY_SCALE = 1.10
STAGE2_PLAIN_FUSE_SCALE = 1.10
STAGE2_NO_PROGRESS_SCALE = 0.50
STAGE2_NON_IMPROVING_CAP = 2.0
STAGE2_DESCRIPTOR_NON_IMPROVING_CAP = 2.0
STAGE3_DENSE_SCALE = 0.70
STAGE3_PREV_GROUP_SCALE = 1.10
STAGE3_BEST_GROUP_SCALE = 1.10
STAGE3_GLOBAL_BASELINE_BLEND = 0.25
STAGE3_BACKBONE_PREV_GROUP_SCALE = 1.20
STAGE3_BACKBONE_BEST_GROUP_SCALE = 1.15
STAGE3_GOAL_BEST_SCALE = 1.00
STAGE3_GOAL_MATCH_SCALE = 1.00
STAGE3_STRUCTURE_SCALE = 0.85
STAGE3_REPEAT_FAMILY_SCALE = 1.00
STAGE3_PLAIN_FUSE_SCALE = 1.00
STAGE3_NO_PROGRESS_SCALE = 1.15
STAGE3_NON_IMPROVING_CAP = 2.0
STAGE3_DESCRIPTOR_NON_IMPROVING_CAP = 2.0
RL_STAGE_KL_COEF = 0.005
reward_batch_index = 0
current_group_id = 0
current_group_reward_target_sum = 0.0
current_group_reward_target_count = 0
current_group_frozen_train_acc_sum = 0.0
current_group_frozen_train_acc_count = 0
current_group_frozen_test_acc_sum = 0.0
current_group_frozen_test_acc_count = 0
current_group_unfrozen_train_acc_sum = 0.0
current_group_unfrozen_train_acc_count = 0
current_group_unfrozen_test_acc_sum = 0.0
current_group_unfrozen_test_acc_count = 0
prev_closed_group_mean_reward_target_acc: Optional[float] = None
best_closed_group_mean_reward_target_acc: Optional[float] = None
prev_closed_group_train_acc_mean: Optional[float] = None
best_closed_group_mean_train_acc: Optional[float] = None
prev_closed_group_mean_test_acc: Optional[float] = None
best_closed_group_mean_test_acc: Optional[float] = None
best_closed_group_id: Optional[int] = None
best_reward_target_by_goal: Dict[str, float] = {}
prev_group_feedback: List["GroupFeedbackSummary"] = []
best_group_feedback: List["GroupFeedbackSummary"] = []
current_group_top_feedback: List["GroupFeedbackSummary"] = []
current_group_goal_best_candidates: Dict[str, float] = {}
current_stage_name = STAGE1_STRUCTURE_EXPLORE
stage_closed_group_counts = Counter()
stage_best_group_mean_reward_target: Dict[str, float] = {}
stage_entry_generation_totals: Dict[str, int] = {}
stage_entry_reward_batches: Dict[str, int] = {}
generation_history: List[Dict[str, Any]] = []
closed_group_history: List[Dict[str, Any]] = []
stage_event_history: List[Dict[str, Any]] = []
recovery_active = False
recovery_start_generation_total = 0
recovery_start_discovery_family_count = 0
# ==================================


class NullCodeLogger:
    def log_to_file(self, message: str) -> None:
        return

    def log_generation(self, prompt: str, completion: str, reward: float, api_result: Any = None) -> None:
        return

    def save_log(self) -> None:
        return


code_logger: Any = NullCodeLogger()
active_rl_model: Any = None
active_rl_tokenizer: Any = None
_registered_signal_handlers: Dict[int, Any] = {}
_signal_checkpoint_in_progress = False


def clear_extraction_meta_cache() -> None:
    return


def clear_reward_extraction_meta_cache() -> None:
    _reward_task_callable("clear_extraction_meta_cache", clear_extraction_meta_cache)()

SHALLOW_COLLAPSE_FAMILIES = {
    "ParallelTriple_Shallow",
    "DualBackboneFuse_Shallow",
    "TripleBackboneFuse_Shallow",
}


@dataclass
class GroupFeedbackSummary:
    goal_key: str
    pattern_name: str
    graph_expr_short: str
    reward_target_value: float
    frozen_train_acc: float
    frozen_test_acc: float
    unfrozen_train_acc: Optional[float]
    unfrozen_test_acc: Optional[float]
    backbone_model_names: List[str]
    stats_short: str
    summary: str
    family_hash: str
    signature: str
    reward_group_id: int
    backbone_signature: str = ""
    cnn_signature: str = ""
    cnn_expr_short: str = ""


def _current_generation_total() -> int:
    return StageState.current_generation_total(sys.modules[__name__])


def capture_reward_runtime_state() -> Dict[str, Any]:
    state = StageState.capture_reward_runtime_state(
        globals(),
        max_stage_sample_history=MAX_STAGE_SAMPLE_HISTORY,
        max_stage_group_history=MAX_STAGE_GROUP_HISTORY,
        feedback_summary_payload=_feedback_summary_payload,
        current_group_top_feedback_payload=_current_group_top_feedback_payload,
    )
    state.update(reward_task_capture_runtime_state())
    return state


def restore_reward_runtime_state(state: Optional[Dict[str, Any]]) -> None:
    StageState.restore_reward_runtime_state(
        globals(),
        state,
        max_stage_sample_history=MAX_STAGE_SAMPLE_HISTORY,
        max_stage_group_history=MAX_STAGE_GROUP_HISTORY,
        stage1_structure_explore=STAGE1_STRUCTURE_EXPLORE,
        feedback_summary_cls=GroupFeedbackSummary,
    )
    reward_task_restore_runtime_state(state)

def _distributed_initialized() -> bool:
    return bool(torch.distributed.is_available() and torch.distributed.is_initialized())


_OBJECT_SYNC_GROUP = None
_OBJECT_SYNC_GROUP_WORLD_SIZE = None
_OBJECT_SYNC_GROUP_DISABLED = False


def _distributed_world_size() -> int:
    if _distributed_initialized():
        return int(torch.distributed.get_world_size())
    return max(1, env_int("WORLD_SIZE", 1))


def _distributed_rank() -> int:
    if _distributed_initialized():
        return int(torch.distributed.get_rank())
    return env_int("RANK", 0)


def is_main_process() -> bool:
    return _distributed_rank() == 0


def _object_sync_timeout_seconds() -> int:
    return max(600, env_int("NNGPT_RL_OBJECT_SYNC_TIMEOUT_SECONDS", 3600))


def _default_process_group_backend() -> str:
    if not _distributed_initialized():
        return ""
    try:
        backend = torch.distributed.get_backend()
    except Exception:
        return ""
    backend_text = str(backend).lower()
    if backend_text.startswith("backend."):
        backend_text = backend_text.split(".", 1)[1]
    return backend_text


def _get_object_sync_group():
    global _OBJECT_SYNC_GROUP
    global _OBJECT_SYNC_GROUP_WORLD_SIZE
    global _OBJECT_SYNC_GROUP_DISABLED

    if not _distributed_initialized() or _distributed_world_size() <= 1:
        return None
    if _default_process_group_backend() == "gloo":
        return None
    current_world_size = _distributed_world_size()
    if _OBJECT_SYNC_GROUP is not None and _OBJECT_SYNC_GROUP_WORLD_SIZE == current_world_size:
        return _OBJECT_SYNC_GROUP
    if _OBJECT_SYNC_GROUP_DISABLED:
        return None

    timeout_seconds = _object_sync_timeout_seconds()
    try:
        _OBJECT_SYNC_GROUP = torch.distributed.new_group(
            backend="gloo",
            timeout=timedelta(seconds=timeout_seconds),
        )
        _OBJECT_SYNC_GROUP_WORLD_SIZE = current_world_size
        print(
            "[Reward Sync Group] initialized "
            f"rank={_distributed_rank()} "
            f"world_size={current_world_size} "
            f"backend=gloo "
            f"timeout_seconds={timeout_seconds}"
        )
    except Exception as exc:
        _OBJECT_SYNC_GROUP = None
        _OBJECT_SYNC_GROUP_WORLD_SIZE = None
        _OBJECT_SYNC_GROUP_DISABLED = True
        print(
            "[Reward Sync Group] fallback "
            f"rank={_distributed_rank()} "
            f"backend={_default_process_group_backend() or 'unknown'} "
            f"error={type(exc).__name__}: {exc}"
        )
    return _OBJECT_SYNC_GROUP


def _all_gather_object(payload: Any) -> List[Any]:
    if not _distributed_initialized() or _distributed_world_size() <= 1:
        return [payload]
    gathered: List[Any] = [None] * _distributed_world_size()
    torch.distributed.all_gather_object(gathered, payload, group=_get_object_sync_group())
    return gathered


def _broadcast_object(payload: Any, *, src: int = 0) -> Any:
    if not _distributed_initialized() or _distributed_world_size() <= 1:
        return payload
    objects = [payload if _distributed_rank() == src else None]
    torch.distributed.broadcast_object_list(objects, src=src, group=_get_object_sync_group())
    return objects[0]


def has_structural_motif(graph_info) -> bool:
    return _backbone_reward_runtime().has_structural_motif(graph_info)


def is_multi_stage_architecture(graph_info) -> bool:
    return _backbone_reward_runtime().is_multi_stage_architecture(graph_info)


def passes_macro_structure_gate(graph_info) -> bool:
    return _backbone_reward_runtime().passes_macro_structure_gate(graph_info)


def is_shallow_one_shot_fuse(graph_info) -> bool:
    return _backbone_reward_runtime().is_shallow_one_shot_fuse(graph_info)


def family_save_cap(graph_info) -> int:
    return 4


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


REWARD_VARIANT_ENV = "NNGPT_RL_REWARD_VARIANT"
REWARD_VARIANT_FULL = "full"
REWARD_VARIANT_NO_STRUCTURAL_NOVELTY = "no_structural_novelty"
REWARD_VARIANT_STRONG_REPEAT_PENALTY = "strong_repeat_penalty"
REWARD_VARIANTS = {
    REWARD_VARIANT_FULL,
    REWARD_VARIANT_NO_STRUCTURAL_NOVELTY,
    REWARD_VARIANT_STRONG_REPEAT_PENALTY,
}


def resolve_reward_variant() -> str:
    raw_value = os.getenv(REWARD_VARIANT_ENV, REWARD_VARIANT_FULL).strip().lower().replace("-", "_")
    if raw_value in {"", "default", "baseline"}:
        return REWARD_VARIANT_FULL
    if raw_value not in REWARD_VARIANTS:
        raise ValueError(
            f"Invalid {REWARD_VARIANT_ENV}={raw_value!r}; "
            f"expected one of {sorted(REWARD_VARIANTS)}"
        )
    return raw_value


def _reward_variant_is_no_structural_novelty() -> bool:
    return resolve_reward_variant() == REWARD_VARIANT_NO_STRUCTURAL_NOVELTY


def _reward_variant_is_strong_repeat_penalty() -> bool:
    return resolve_reward_variant() == REWARD_VARIANT_STRONG_REPEAT_PENALTY


def _without_positive_bonus(value: float) -> float:
    return min(float(value or 0.0), 0.0)


def _remove_positive_structural_novelty_components(
    components: Dict[str, float],
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    adjusted: Dict[str, float] = {}
    removed_positive: Dict[str, float] = {}
    original: Dict[str, float] = {}
    for key, value in components.items():
        original_value = float(value or 0.0)
        adjusted_value = _without_positive_bonus(original_value)
        original[key] = original_value
        adjusted[key] = adjusted_value
        if adjusted_value != original_value:
            removed_positive[key] = original_value - adjusted_value
    return adjusted, {
        "original_components": original,
        "removed_positive_bonus": removed_positive,
    }


def _stage1_only_enabled() -> bool:
    return env_flag("NNGPT_RL_STAGE1_ONLY", False)


def resolve_generation_plan(
    runtime: Dict[str, Any],
    *,
    env_name: str,
    default: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
) -> Dict[str, int]:
    world_size = max(1, int(runtime.get("world_size", 1)))
    requested_global_num_generations = max(1, env_int(env_name, default))
    effective_train_batch_size = max(
        1,
        int(world_size) * max(1, int(per_device_train_batch_size)) * max(1, int(gradient_accumulation_steps)),
    )
    valid_generation_values = [
        value
        for value in range(2, effective_train_batch_size + 1)
        if effective_train_batch_size % value == 0
    ]
    if not valid_generation_values:
        raise ValueError(
            f"{env_name} cannot be resolved because effective_train_batch_size={effective_train_batch_size} "
            "does not permit GRPO's minimum 2 generations per prompt. Increase gradient accumulation or batch size."
        )

    if requested_global_num_generations in valid_generation_values:
        resolved_global_num_generations = requested_global_num_generations
    else:
        lower_or_equal = [value for value in valid_generation_values if value <= requested_global_num_generations]
        resolved_global_num_generations = (
            max(lower_or_equal)
            if lower_or_equal
            else min(valid_generation_values)
        )
    return {
        "world_size": world_size,
        "per_device_train_batch_size": int(per_device_train_batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "effective_train_batch_size": int(effective_train_batch_size),
        "requested_global_num_generations": requested_global_num_generations,
        "global_num_generations": int(resolved_global_num_generations),
        "effective_global_num_generations": int(resolved_global_num_generations),
        "global_num_generations_adapted": int(resolved_global_num_generations != requested_global_num_generations),
        "valid_generation_values": list(valid_generation_values),
    }


def resolve_rl_runtime_settings(runtime: Dict[str, Any]) -> Dict[str, int]:
    grad_accum = env_int("NNGPT_RL_GRAD_ACCUM", 16)
    fixed_num_generations = 8
    os.environ["NNGPT_RL_NUM_GENERATIONS"] = str(fixed_num_generations)
    generation_plan = resolve_generation_plan(
        runtime,
        env_name="NNGPT_RL_NUM_GENERATIONS",
        default=fixed_num_generations,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=grad_accum,
    )
    return {
        "dataset_limit": env_int("NNGPT_RL_DATASET_LIMIT", 500),
        "grad_accum": grad_accum,
        "max_completion_length": env_int("NNGPT_RL_MAX_COMPLETION_LENGTH", 1024),
        "effective_train_batch_size": generation_plan["effective_train_batch_size"],
        "requested_global_num_generations": generation_plan["requested_global_num_generations"],
        "global_num_generations": generation_plan["global_num_generations"],
        "effective_global_num_generations": generation_plan["effective_global_num_generations"],
        "global_num_generations_adapted": generation_plan["global_num_generations_adapted"],
        "valid_generation_values": generation_plan["valid_generation_values"],
    }


def _resolve_rl_deepspeed_enabled(runtime: Dict[str, Any]) -> bool:
    raw = os.getenv("NNGPT_RL_USE_DEEPSPEED")
    if raw is None or raw == "":
        return int(runtime.get("world_size", 1)) > 1
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_rl_deepspeed_config_path() -> str:
    config_path = Path(os.getenv("NNGPT_RL_DEEPSPEED_CONFIG", RL_DEEPSPEED_DEFAULT_CONFIG)).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"RL DeepSpeed config not found: {config_path}")
    return str(config_path)


def _maybe_init_hf_deepspeed_config(config_path: str) -> Any:
    last_error: Optional[Exception] = None
    for module_name in ("transformers.integrations", "transformers.deepspeed"):
        try:
            module = __import__(module_name, fromlist=["HfDeepSpeedConfig"])
            config_cls = getattr(module, "HfDeepSpeedConfig", None)
            if config_cls is not None:
                return config_cls(config_path)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(
        "DeepSpeed ZeRO-3 requested for RL GRPO, but HfDeepSpeedConfig could not be imported"
    ) from last_error


def _build_rl_grpo_config(
    *,
    precision: Dict[str, Any],
    use_deepspeed: bool,
    deepspeed_config_path: Optional[str],
    runtime_settings: Dict[str, int],
) -> Any:
    config_signature = inspect.signature(GRPOConfig.__init__)
    config_kwargs: Dict[str, Any] = {
        "temperature": env_float("NNGPT_RL_TEMPERATURE", 1.0),
        "learning_rate": env_float("NNGPT_RL_LR", 5e-5),
        "max_completion_length": runtime_settings["max_completion_length"],
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": runtime_settings["grad_accum"],
        "lr_scheduler_type": "cosine",
        "num_train_epochs": env_int("NNGPT_RL_NUM_EPOCHS", 5),
        "remove_unused_columns": False,
        "logging_steps": 1,
        "output_dir": os.getenv("NNGPT_RL_TRAINER_OUT", "./grpo_backbone_outputs"),
        "eval_strategy": "no",
        "bf16": precision["bf16"],
        "fp16": precision["fp16"],
        "gradient_checkpointing": True,
        "num_generations": runtime_settings["global_num_generations"],
    }
    if "gradient_checkpointing_kwargs" in config_signature.parameters:
        config_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    explicit_kl_coef = env_float("NNGPT_RL_KL_COEF", RL_STAGE_KL_COEF)
    if "beta" in config_signature.parameters:
        config_kwargs["beta"] = explicit_kl_coef
    elif "kl_coef" in config_signature.parameters:
        config_kwargs["kl_coef"] = explicit_kl_coef
    else:
        raise RuntimeError("Installed GRPOConfig does not expose `beta` or `kl_coef`; cannot set explicit KL control")
    if use_deepspeed:
        if "deepspeed" not in config_signature.parameters:
            raise RuntimeError("Installed GRPOConfig does not support the `deepspeed` argument")
        config_kwargs["deepspeed"] = deepspeed_config_path
        if "ds3_gather_for_generation" in config_signature.parameters:
            config_kwargs["ds3_gather_for_generation"] = False
    return GRPOConfig(**config_kwargs)


def best_mixed_precision() -> Dict[str, Any]:
    bf16_requested = os.getenv("NNGPT_RL_USE_BF16", "").strip().lower() in {"1", "true", "yes", "on"}
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_bf16 = bool(bf16_requested and bf16_ok)
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float16
    return {
        "bf16": use_bf16,
        "fp16": not use_bf16,
        "torch_dtype": torch_dtype,
        "label": "bf16" if use_bf16 else "fp16",
    }


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _optional_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _result_reward_target_value(res: Dict[str, Any]) -> Optional[float]:
    reward_target_value = _optional_float(res.get("reward_target_value"))
    if reward_target_value is not None:
        return reward_target_value
    return _optional_float(res.get("frozen_test_acc", res.get("val_metric")))


def _increment_optional_metric(sum_name: str, count_name: str, value: Optional[float]) -> None:
    if value is None:
        return
    globals()[sum_name] += float(value)
    globals()[count_name] += 1


def _mean_from_accumulator(sum_value: float, count_value: int) -> Optional[float]:
    if count_value <= 0:
        return None
    return float(sum_value) / float(count_value)


def _truncate_text(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."


def _feedback_stats_short(open_discovery: Dict[str, Any]) -> str:
    structure_progress = float(open_discovery.get("r_structure_group", 0.0) or 0.0) + float(
        open_discovery.get("r_structure_archive", 0.0) or 0.0
    )
    return (
        f"depth:{int(open_discovery.get('depth', 0))} "
        f"merges:{int(open_discovery.get('merges', 0))} "
        f"stem:{int(open_discovery.get('stem_calls', 0))} "
        f"project:{int(open_discovery.get('project_calls', 0))} "
        f"fuse:{int(open_discovery.get('fuse_calls', 0))} "
        f"struct:{structure_progress:.2f}"
    )


def _group_feedback_paths() -> Tuple[Path, Path, Path]:
    log_dir = Path(reward_run_log_dir())
    log_dir.mkdir(parents=True, exist_ok=True)
    return (
        log_dir / "group_progress.jsonl",
        log_dir / "group_feedback_samples.jsonl",
        log_dir / "best_group_feedback.json",
    )


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _reward_runtime_hooks() -> TrainingRuntime.RuntimeStateHooks:
    # Reward bookkeeping is pipeline-owned, but the save/restore contract is shared.
    return TrainingRuntime.RuntimeStateHooks(
        capture=capture_reward_runtime_state,
        restore=restore_reward_runtime_state,
        reset=reset_reward_runtime_state,
    )


def _current_group_top_feedback_payload() -> List[Dict[str, Any]]:
    return [asdict(item) for item in current_group_top_feedback[:FEEDBACK_SUMMARY_LIMIT]]


def _feedback_summary_payload(items: List[GroupFeedbackSummary]) -> List[Dict[str, Any]]:
    return [asdict(item) for item in items[:FEEDBACK_SUMMARY_LIMIT]]


def _build_group_feedback_summary(
    *,
    goal_key: str,
    res: Dict[str, Any],
    graph_info,
    reward_group_id: int,
) -> GroupFeedbackSummary:
    graph_expr_short = _truncate_text(str(res.get("graph_expr") or ""), FEEDBACK_GRAPH_EXPR_MAX_CHARS)
    pattern_name = str(res.get("pattern_name") or res.get("suggested_pattern_name") or "unknown")
    reward_target_value = float(_result_reward_target_value(res) or 0.0)
    frozen_train_acc = float(_optional_float(res.get("frozen_train_acc", res.get("train_acc"))) or 0.0)
    frozen_test_acc = float(_optional_float(res.get("frozen_test_acc", res.get("test_acc", res.get("val_metric")))) or 0.0)
    unfrozen_train_acc = _optional_float(res.get("unfrozen_train_acc"))
    unfrozen_test_acc = _optional_float(res.get("unfrozen_test_acc"))
    backbone_names = list(res.get("backbone_model_names") or [])
    backbone_signature = str(res.get("backbone_signature") or _backbone_reward_runtime().build_backbone_signature(backbone_names))
    cnn_signature = str(res.get("cnn_signature") or getattr(graph_info, "cnn_signature", "") or "")
    cnn_expr_short = _truncate_text(str(res.get("cnn_expr") or getattr(graph_info, "cnn_expr", "") or ""), 96)
    open_discovery = dict(res.get("open_discovery") or {})
    stats_short = _feedback_stats_short(open_discovery)
    summary = (
        f"pattern={pattern_name}; "
        f"target={reward_target_value:.4f}; "
        f"frozen_train={frozen_train_acc:.4f}; "
        f"frozen_test={frozen_test_acc:.4f}; "
        f"backbones=[{', '.join(backbone_names)}]; "
        f"backbone_bucket={backbone_signature}; "
        f"cnn={cnn_expr_short or cnn_signature or 'n/a'}; "
        f"graph={graph_expr_short}; "
        f"stats={stats_short}"
    )
    summary = _truncate_text(summary, FEEDBACK_SUMMARY_MAX_CHARS)
    return GroupFeedbackSummary(
        goal_key=goal_key,
        pattern_name=pattern_name,
        graph_expr_short=graph_expr_short,
        reward_target_value=reward_target_value,
        frozen_train_acc=frozen_train_acc,
        frozen_test_acc=frozen_test_acc,
        unfrozen_train_acc=unfrozen_train_acc,
        unfrozen_test_acc=unfrozen_test_acc,
        backbone_model_names=backbone_names,
        stats_short=stats_short,
        summary=summary,
        family_hash=str(getattr(graph_info, "family_hash", "") or res.get("family_hash") or ""),
        signature=str(res.get("signature") or ""),
        reward_group_id=reward_group_id,
        backbone_signature=backbone_signature,
        cnn_signature=cnn_signature,
        cnn_expr_short=cnn_expr_short,
    )


def _update_top_feedback(summary: GroupFeedbackSummary) -> None:
    current_group_top_feedback.append(summary)
    current_group_top_feedback.sort(key=lambda item: item.reward_target_value, reverse=True)
    del current_group_top_feedback[FEEDBACK_SUMMARY_LIMIT:]


def _record_current_group_trainable_sample(goal_key: str, res: Dict[str, Any], graph_info) -> None:
    reward_target_value = _result_reward_target_value(res)
    if reward_target_value is None:
        return
    current_best = current_group_goal_best_candidates.get(goal_key)
    if current_best is None or float(reward_target_value) > current_best:
        current_group_goal_best_candidates[goal_key] = float(reward_target_value)
    summary = _build_group_feedback_summary(
        goal_key=goal_key,
        res=res,
        graph_info=graph_info,
        reward_group_id=current_group_id,
    )
    _update_top_feedback(summary)


def _reset_current_group_feedback_state() -> None:
    current_group_top_feedback.clear()
    current_group_goal_best_candidates.clear()


def get_prompt_feedback_state() -> Dict[str, Any]:
    training_context = summarize_stage_training_context(current_stage_name)
    return {
        "prev_closed_group_mean_reward_target_acc": prev_closed_group_mean_reward_target_acc,
        "best_closed_group_mean_reward_target_acc": best_closed_group_mean_reward_target_acc,
        "prev_closed_group_mean_train_acc": prev_closed_group_train_acc_mean,
        "best_closed_group_mean_train_acc": best_closed_group_mean_train_acc,
        "prev_closed_group_mean_test_acc": prev_closed_group_mean_test_acc,
        "best_closed_group_mean_test_acc": best_closed_group_mean_test_acc,
        "best_closed_group_id": best_closed_group_id,
        **reward_task_group_context_fields(),
        "prev_group_feedback": _feedback_summary_payload(prev_group_feedback),
        "best_group_feedback": _feedback_summary_payload(best_group_feedback),
        "training_context": training_context,
    }


def _format_optional_metric(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _format_optional_signed_metric(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):+.4f}"


def _format_target_metric(base_value: Optional[float], delta: float) -> str:
    if base_value is None:
        return "n/a"
    return f"{float(base_value) + float(delta):.4f}"


def render_prompt_feedback_text(*, feedback_char_budget: int = 1200) -> str:
    return _reward_task_callable(
        "render_prompt_feedback_text",
        _backbone_reward_runtime().render_prompt_feedback_text,
    )(feedback_char_budget=feedback_char_budget)

def reset_reward_runtime_state() -> None:
    StageState.reset_reward_runtime_state(
        globals(),
        stage1_structure_explore=STAGE1_STRUCTURE_EXPLORE,
        reset_current_group_feedback_state=_reset_current_group_feedback_state,
    )
    reward_task_reset_runtime_state()

def current_reward_group_context() -> Dict[str, Any]:
    return StageState.current_reward_group_context(sys.modules[__name__])


def default_reward_replay_group_context() -> Dict[str, Any]:
    return {
        "group_baseline_train_acc": None,
        "group_baseline_reward_target_acc": None,
        "group_baseline_test_acc": None,
        "best_closed_group_mean_train_acc": None,
        "best_closed_group_mean_reward_target_acc": None,
        "best_closed_group_mean_test_acc": None,
        "best_closed_group_id": None,
        **reward_task_group_context_fields(),
        "reward_batch_index": 0,
        "reward_group_id": 0,
        "group_warmup": False,
        "current_stage_name": current_stage_name,
        "current_stage_index": RL_STAGE_TO_INDEX.get(current_stage_name, 0),
        "generation_total": 0,
        "stage_group_count": 0,
        "recovery_active": False,
    }


def _read_process_rss_gib() -> Optional[float]:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return float(parts[1]) / (1024.0 * 1024.0)
                    break
    except OSError:
        return None
    return None


def _cuda_memory_gib() -> Tuple[Optional[float], Optional[float]]:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    try:
        allocated = torch.cuda.memory_allocated() / float(1024 ** 3)
        reserved = torch.cuda.memory_reserved() / float(1024 ** 3)
        return allocated, reserved
    except RuntimeError:
        return None, None


def _format_mem_value(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _visible_cuda_device_tokens() -> List[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return []
    raw = raw.strip()
    if raw in {"", "-1"}:
        return []
    return [token.strip() for token in raw.split(",") if token.strip()]


def _resolved_train_gpu_index() -> Optional[int]:
    if not torch.cuda.is_available():
        return None
    visible_gpu_count = int(torch.cuda.device_count())
    if visible_gpu_count <= 0:
        return None
    if _distributed_world_size() > 1:
        raw_local_rank = env_int("LOCAL_RANK", 0)
        if visible_gpu_count == 1:
            return 0
        if 0 <= raw_local_rank < visible_gpu_count:
            return raw_local_rank
    try:
        current_device = int(torch.cuda.current_device())
        if 0 <= current_device < visible_gpu_count:
            return current_device
    except Exception:
        pass
    raw_local_rank = env_int("LOCAL_RANK", 0)
    if visible_gpu_count == 1:
        return 0
    if 0 <= raw_local_rank < visible_gpu_count:
        return raw_local_rank
    return 0


def _visible_cuda_memory_snapshots(*, include_all_visible_gpus: bool) -> List[Dict[str, Any]]:
    if not torch.cuda.is_available():
        return []
    visible_gpu_count = int(torch.cuda.device_count())
    if visible_gpu_count <= 0:
        return []
    device_tokens = _visible_cuda_device_tokens()
    train_gpu_index = _resolved_train_gpu_index()
    if include_all_visible_gpus:
        device_indices = list(range(visible_gpu_count))
    elif train_gpu_index is not None:
        device_indices = [int(train_gpu_index)]
    else:
        device_indices = [0]

    snapshots: List[Dict[str, Any]] = []
    for device_index in device_indices:
        total_gib = None
        free_gib = None
        used_gib = None
        allocated_gib = None
        reserved_gib = None
        device_name = ""
        try:
            props = torch.cuda.get_device_properties(device_index)
            total_gib = float(props.total_memory) / float(1024 ** 3)
            device_name = str(getattr(props, "name", "") or "")
        except Exception:
            pass
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
            free_gib = float(free_bytes) / float(1024 ** 3)
            used_gib = float(total_bytes - free_bytes) / float(1024 ** 3)
            if total_gib is None:
                total_gib = float(total_bytes) / float(1024 ** 3)
        except Exception:
            pass
        try:
            allocated_gib = float(torch.cuda.memory_allocated(device_index)) / float(1024 ** 3)
        except Exception:
            allocated_gib = None
        try:
            reserved_gib = float(torch.cuda.memory_reserved(device_index)) / float(1024 ** 3)
        except Exception:
            reserved_gib = None

        other_used_gib = None
        if used_gib is not None and allocated_gib is not None:
            other_used_gib = max(0.0, float(used_gib) - float(allocated_gib))

        device_token = (
            device_tokens[device_index]
            if 0 <= device_index < len(device_tokens)
            else str(int(device_index))
        )
        snapshots.append(
            {
                "device_index": int(device_index),
                "device_token": str(device_token),
                "device_name": device_name,
                "total_gib": total_gib,
                "free_gib": free_gib,
                "used_gib": used_gib,
                "allocated_gib": allocated_gib,
                "reserved_gib": reserved_gib,
                "other_used_gib": other_used_gib,
                "is_train_gpu": bool(train_gpu_index is not None and int(train_gpu_index) == int(device_index)),
            }
        )
    return snapshots


def _current_cuda_allocator_snapshot() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {}
    try:
        current_device = int(torch.cuda.current_device())
    except Exception:
        current_device = _resolved_train_gpu_index()
    if current_device is None:
        return {}
    try:
        stats = torch.cuda.memory_stats(current_device)
    except Exception:
        return {}
    return {
        "current_device": int(current_device),
        "active_gib": float(stats.get("active_bytes.all.current", 0.0)) / float(1024 ** 3),
        "reserved_gib": float(stats.get("reserved_bytes.all.current", 0.0)) / float(1024 ** 3),
        "inactive_split_gib": float(stats.get("inactive_split_bytes.all.current", 0.0)) / float(1024 ** 3),
        "num_ooms": int(stats.get("num_ooms", 0)),
        "num_alloc_retries": int(stats.get("num_alloc_retries", 0)),
    }


def _query_nvidia_smi_csv(query_kind: str, columns: List[str]) -> List[List[str]]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return []
    command = [
        executable,
        f"--query-{query_kind}={','.join(columns)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return []
    if completed.returncode != 0 or not completed.stdout.strip():
        return []
    reader = csv.reader(completed.stdout.splitlines())
    return [[cell.strip() for cell in row] for row in reader if row]


def _log_nvidia_smi_snapshot(stage: str) -> None:
    gpu_rows = _query_nvidia_smi_csv(
        "gpu",
        ["index", "uuid", "name", "memory.total", "memory.used", "memory.free"],
    )
    if not gpu_rows:
        print(f"[OOM nvidia-smi] stage={stage} unavailable=True")
        return

    visible_tokens = set(_visible_cuda_device_tokens())
    filter_visible = bool(visible_tokens)
    gpu_rows_by_uuid: Dict[str, Dict[str, str]] = {}
    for row in gpu_rows:
        if len(row) < 6:
            continue
        gpu_index, gpu_uuid, gpu_name, total_mib, used_mib, free_mib = row[:6]
        if filter_visible and gpu_index not in visible_tokens and gpu_uuid not in visible_tokens:
            continue
        gpu_rows_by_uuid[gpu_uuid] = {
            "index": gpu_index,
            "uuid": gpu_uuid,
            "name": gpu_name,
            "total_mib": total_mib,
            "used_mib": used_mib,
            "free_mib": free_mib,
        }
        print(
            "[OOM nvidia-smi GPU] "
            f"stage={stage} "
            f"gpu={gpu_index} "
            f"name={gpu_name!r} "
            f"used_mib={used_mib} "
            f"free_mib={free_mib} "
            f"total_mib={total_mib}"
        )

    process_rows = _query_nvidia_smi_csv(
        "compute-apps",
        ["gpu_uuid", "pid", "process_name", "used_memory"],
    )
    process_snapshots: List[Dict[str, Any]] = []
    for row in process_rows:
        if len(row) < 4:
            continue
        gpu_uuid, pid, process_name, used_mib = row[:4]
        gpu_info = gpu_rows_by_uuid.get(gpu_uuid)
        if gpu_info is None:
            continue
        try:
            used_mib_value = int(float(used_mib))
        except (TypeError, ValueError):
            used_mib_value = 0
        process_snapshots.append(
            {
                "gpu": gpu_info["index"],
                "pid": pid,
                "process_name": process_name,
                "used_mib": used_mib_value,
            }
        )
    process_snapshots.sort(key=lambda item: int(item["used_mib"]), reverse=True)
    for snapshot in process_snapshots[:24]:
        print(
            "[OOM nvidia-smi Proc] "
            f"stage={stage} "
            f"gpu={snapshot['gpu']} "
            f"pid={snapshot['pid']} "
            f"used_mib={snapshot['used_mib']} "
            f"process={snapshot['process_name']!r}"
        )


def is_cuda_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    normalized = " ".join(str(exc).split()).lower()
    return "out of memory" in normalized and "cuda" in normalized


def log_cuda_oom_diagnostics(
    stage: str,
    exc: BaseException,
    *,
    group_context: Optional[Dict[str, Any]] = None,
) -> None:
    print(f"[OOM] stage={stage} error={type(exc).__name__}: {exc}")
    log_memory_snapshot(stage, group_context=group_context, include_all_visible_gpus=True)
    allocator_snapshot = _current_cuda_allocator_snapshot()
    if allocator_snapshot:
        print(
            "[OOM Allocator] "
            f"stage={stage} "
            f"current_device={allocator_snapshot['current_device']} "
            f"active_gib={_format_mem_value(allocator_snapshot['active_gib'])} "
            f"reserved_gib={_format_mem_value(allocator_snapshot['reserved_gib'])} "
            f"inactive_split_gib={_format_mem_value(allocator_snapshot['inactive_split_gib'])} "
            f"num_ooms={allocator_snapshot['num_ooms']} "
            f"num_alloc_retries={allocator_snapshot['num_alloc_retries']}"
        )
    _log_nvidia_smi_snapshot(stage)


class _CudaMemoryMonitor:
    def __init__(self, stage_prefix: str) -> None:
        self._stage_prefix = str(stage_prefix)
        self._enabled = bool(torch.cuda.is_available()) and env_int("NNGPT_CUDA_MEMORY_MONITOR", 1) > 0
        self._interval_seconds = max(1.0, env_float("NNGPT_CUDA_MEMORY_MONITOR_INTERVAL_SECONDS", 30.0))
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> Optional["_CudaMemoryMonitor"]:
        if not self._enabled:
            return None
        print(
            "[Memory Monitor] "
            f"stage_prefix={self._stage_prefix} "
            f"interval_seconds={self._interval_seconds:.1f}"
        )
        self._thread = threading.Thread(
            target=self._run,
            name=f"nngpt-cuda-memory-monitor-{self._stage_prefix}",
            daemon=True,
        )
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_seconds):
            try:
                log_memory_snapshot(f"{self._stage_prefix}:tick")
            except Exception as exc:
                print(
                    "[Memory Monitor] "
                    f"stage_prefix={self._stage_prefix} "
                    f"error={type(exc).__name__}: {exc}"
                )

    def close(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=max(2.0, self._interval_seconds + 1.0))
        self._thread = None


def start_cuda_memory_monitor(stage_prefix: str) -> Optional[_CudaMemoryMonitor]:
    monitor = _CudaMemoryMonitor(stage_prefix)
    return monitor.start()


def log_memory_snapshot(
    stage: str,
    *,
    group_context: Optional[Dict[str, Any]] = None,
    include_all_visible_gpus: Optional[bool] = None,
) -> None:
    effective_group_context = group_context or current_reward_group_context()
    cuda_allocated_gib, cuda_reserved_gib = _cuda_memory_gib()
    worker_info = get_eval_worker_diagnostics()
    worker_pid = worker_info.get("worker_pids", [worker_info.get("pid")]) if worker_info else None
    rank = _distributed_rank()
    local_rank = env_int("LOCAL_RANK", 0)
    world_size = _distributed_world_size()
    train_gpu = _resolved_train_gpu_index()
    if include_all_visible_gpus is None:
        # In single-process SFT/RL runs, touching every visible GPU here creates
        # extra CUDA contexts on reward GPUs. Default to the training GPU only
        # unless a caller explicitly asks for a full visible-device snapshot.
        include_all_visible_gpus = bool(world_size > 1 and is_main_process())
    visible_cuda_snapshots = _visible_cuda_memory_snapshots(
        include_all_visible_gpus=bool(include_all_visible_gpus)
    )
    print(
        "[Memory] "
        f"stage={stage} "
        f"pid={os.getpid()} "
        f"rank={rank} "
        f"local_rank={local_rank} "
        f"world_size={world_size} "
        f"train_gpu={train_gpu} "
        f"reward_batch_index={effective_group_context.get('reward_batch_index')} "
        f"reward_group_id={effective_group_context.get('reward_group_id')} "
        f"rss_gib={_format_mem_value(_read_process_rss_gib())} "
        f"cuda_allocated_gib={_format_mem_value(cuda_allocated_gib)} "
        f"cuda_reserved_gib={_format_mem_value(cuda_reserved_gib)} "
        f"worker_pid={worker_pid}"
    )
    for snapshot in visible_cuda_snapshots:
        train_gpu_marker = "*" if snapshot.get("is_train_gpu") else ""
        print(
            "[Memory GPU] "
            f"stage={stage} "
            f"gpu={snapshot['device_index']}{train_gpu_marker} "
            f"token={snapshot['device_token']} "
            f"name={snapshot['device_name']!r} "
            f"free_gib={_format_mem_value(snapshot['free_gib'])} "
            f"used_gib={_format_mem_value(snapshot['used_gib'])} "
            f"total_gib={_format_mem_value(snapshot['total_gib'])} "
            f"proc_allocated_gib={_format_mem_value(snapshot['allocated_gib'])} "
            f"proc_reserved_gib={_format_mem_value(snapshot['reserved_gib'])} "
            f"other_used_gib={_format_mem_value(snapshot['other_used_gib'])}"
        )
def update_current_group_metrics(results: List[Dict[str, Any]]) -> None:
    reward_task_update_group_metrics(results)
    for res in results:
        reward_target_value = _result_reward_target_value(res)
        _increment_optional_metric(
            "current_group_reward_target_sum",
            "current_group_reward_target_count",
            reward_target_value,
        )
        _increment_optional_metric(
            "current_group_frozen_train_acc_sum",
            "current_group_frozen_train_acc_count",
            _optional_float(res.get("frozen_train_acc", res.get("train_acc"))),
        )
        _increment_optional_metric(
            "current_group_frozen_test_acc_sum",
            "current_group_frozen_test_acc_count",
            _optional_float(res.get("frozen_test_acc", res.get("test_acc", res.get("val_metric")))),
        )
        _increment_optional_metric(
            "current_group_unfrozen_train_acc_sum",
            "current_group_unfrozen_train_acc_count",
            _optional_float(res.get("unfrozen_train_acc")),
        )
        _increment_optional_metric(
            "current_group_unfrozen_test_acc_sum",
            "current_group_unfrozen_test_acc_count",
            _optional_float(res.get("unfrozen_test_acc")),
        )


def _reset_stage_comparison_state() -> None:
    global prev_closed_group_mean_reward_target_acc
    global best_closed_group_mean_reward_target_acc
    global prev_closed_group_train_acc_mean
    global best_closed_group_mean_train_acc
    global prev_closed_group_mean_test_acc
    global best_closed_group_mean_test_acc
    global best_closed_group_id

    prev_closed_group_mean_reward_target_acc = None
    best_closed_group_mean_reward_target_acc = None
    prev_closed_group_train_acc_mean = None
    best_closed_group_mean_train_acc = None
    prev_closed_group_mean_test_acc = None
    best_closed_group_mean_test_acc = None
    best_closed_group_id = None
    reward_task_reset_stage_comparison_state()
    best_reward_target_by_goal.clear()
    prev_group_feedback.clear()
    best_group_feedback.clear()
    _reset_current_group_feedback_state()


def _stage_checkpoint_root() -> Path:
    return StageState.stage_checkpoint_root(sys.modules[__name__])


def _stage_checkpoint_dir(stage_name: str) -> Path:
    return StageState.stage_checkpoint_dir(sys.modules[__name__], stage_name)


def _stage_group_snapshot_payload(current_group_payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return StageState.stage_group_snapshot_payload(sys.modules[__name__], current_group_payload)


def _save_stage_plot_snapshot(output_path: Path) -> None:
    return StageState.save_stage_plot_snapshot(sys.modules[__name__], output_path)


def _save_stage_checkpoint(
    event: str,
    *,
    stage_name: Optional[str] = None,
    group_progress_payload: Optional[Dict[str, Any]] = None,
    reason: Optional[str] = None,
    save_plot_snapshot: bool = True,
) -> Optional[Path]:
    return StageState.save_stage_checkpoint(
        sys.modules[__name__],
        event,
        stage_name=stage_name,
        group_progress_payload=group_progress_payload,
        reason=reason,
        save_plot_snapshot=save_plot_snapshot,
    )


def _handle_checkpoint_signal(signum: int, _frame) -> None:
    global _signal_checkpoint_in_progress

    if _signal_checkpoint_in_progress:
        raise SystemExit(128 + int(signum))

    _signal_checkpoint_in_progress = True
    signal_name = signal.Signals(signum).name.lower()
    try:
        _save_stage_checkpoint(
            "signal",
            stage_name=current_stage_name,
            reason=f"signal_{signal_name}",
            save_plot_snapshot=False,
        )
        try:
            code_logger.save_log()
        except Exception as exc:
            code_logger.log_to_file(f"[Signal Save] save_log failed: {type(exc).__name__}: {exc}")
    finally:
        signal.signal(signum, signal.SIG_DFL)
        _signal_checkpoint_in_progress = False

    raise SystemExit(128 + int(signum))


def register_stage_checkpoint_signal_handlers() -> None:
    if not is_main_process():
        return
    for signum in (signal.SIGTERM, signal.SIGINT):
        if signum in _registered_signal_handlers:
            continue
        _registered_signal_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, _handle_checkpoint_signal)


def _stage1_gate_ready() -> bool:
    recent_generations = _recent_stage_generation_window(STAGE1_STRUCTURE_EXPLORE, STAGE1_GATE_WINDOW_GENERATIONS)
    current_entry_group_count = len(_recent_stage_group_window(STAGE1_STRUCTURE_EXPLORE, MAX_STAGE_GROUP_HISTORY))
    if len(recent_generations) < STAGE1_GATE_WINDOW_GENERATIONS:
        return False
    if current_entry_group_count < STAGE1_PROMOTION_MIN_GROUPS:
        return False
    executable_count = sum(1 for item in recent_generations if bool(item.get("executable_candidate")))
    trainable_count = sum(
        1
        for item in recent_generations
        if bool(item.get("trained_step_ok") or item.get("backward_ok"))
    )
    discovery_rows = [item for item in recent_generations if bool(item.get("discovery_candidate"))]
    unique_discovery_families = len(_family_hash_set(discovery_rows, key="family_hash"))
    return bool(
        executable_count >= STAGE1_GATE_EXECUTABLE_MIN
        and trainable_count >= STAGE1_GATE_TRAINABLE_MIN
        and len(discovery_rows) >= STAGE1_GATE_DISCOVERY_MIN
        and unique_discovery_families >= STAGE1_GATE_UNIQUE_DISCOVERY_FAMILIES_MIN
    )


def _stage1_trainable_stable_ready() -> Optional[Dict[str, Any]]:
    recent_generations = _recent_stage_generation_window(
        STAGE1_STRUCTURE_EXPLORE,
        STAGE1_EXECUTABLE_STABLE_WINDOW_GENERATIONS,
    )
    current_entry_group_count = len(_recent_stage_group_window(STAGE1_STRUCTURE_EXPLORE, MAX_STAGE_GROUP_HISTORY))
    if current_entry_group_count < STAGE1_EXECUTABLE_STABLE_MIN_GROUPS:
        return None
    if len(recent_generations) < STAGE1_EXECUTABLE_STABLE_WINDOW_GENERATIONS:
        return None
    recent_executable_count = sum(1 for item in recent_generations if bool(item.get("executable_candidate")))
    recent_executable_rate = recent_executable_count / float(len(recent_generations))
    recent_trainable_count = sum(
        1
        for item in recent_generations
        if bool(item.get("trained_step_ok") or item.get("backward_ok"))
    )
    recent_trainable_rate = recent_trainable_count / float(len(recent_generations))
    if recent_executable_rate < STAGE1_EXECUTABLE_STABLE_MIN_RATE:
        return None
    if recent_trainable_rate < STAGE1_TRAINABLE_STABLE_MIN_RATE:
        return None
    return {
        "stage_group_count": current_entry_group_count,
        "recent_generation_count": len(recent_generations),
        "recent_executable_count": recent_executable_count,
        "recent_executable_rate": recent_executable_rate,
        "recent_trainable_count": recent_trainable_count,
        "recent_trainable_rate": recent_trainable_rate,
    }


def _stage1_force_promotion_ready() -> Optional[Dict[str, int]]:
    recent_generations = _recent_stage_generation_window(STAGE1_STRUCTURE_EXPLORE, STAGE1_GATE_WINDOW_GENERATIONS)
    current_entry_group_count = len(_recent_stage_group_window(STAGE1_STRUCTURE_EXPLORE, MAX_STAGE_GROUP_HISTORY))
    if len(recent_generations) < STAGE1_GATE_WINDOW_GENERATIONS:
        return None
    if current_entry_group_count < STAGE1_PROMOTION_MIN_GROUPS:
        return None
    recent_executable_count = sum(1 for item in recent_generations if bool(item.get("executable_candidate")))
    recent_trainable_count = sum(
        1
        for item in recent_generations
        if bool(item.get("trained_step_ok") or item.get("backward_ok"))
    )
    discovery_rows = [item for item in recent_generations if bool(item.get("discovery_candidate"))]
    recent_discovery_count = len(discovery_rows)
    recent_unique_discovery_families = len(_family_hash_set(discovery_rows, key="family_hash"))
    if recent_executable_count < STAGE1_FORCE_PROMOTION_EXECUTABLE_MIN:
        return None
    if recent_trainable_count < STAGE1_FORCE_PROMOTION_TRAINABLE_MIN:
        return None
    if recent_discovery_count < STAGE1_FORCE_PROMOTION_DISCOVERY_MIN:
        return None
    if recent_unique_discovery_families < STAGE1_FORCE_PROMOTION_UNIQUE_DISCOVERY_FAMILIES_MIN:
        return None
    return {
        "stage_group_count": current_entry_group_count,
        "recent_generation_count": len(recent_generations),
        "recent_executable_count": recent_executable_count,
        "recent_trainable_count": recent_trainable_count,
        "recent_discovery_count": recent_discovery_count,
        "recent_unique_discovery_families": recent_unique_discovery_families,
    }


def _stage2_gate_ready() -> bool:
    recent_generations = _recent_stage_generation_window(STAGE2_FORMAL_EXPLORE, STAGE2_GATE_WINDOW_GENERATIONS)
    recent_groups = _recent_stage_group_window(STAGE2_FORMAL_EXPLORE, 5)
    recent_improvement_groups = _recent_stage_group_window(STAGE2_FORMAL_EXPLORE, 4)
    current_entry_group_count = len(_recent_stage_group_window(STAGE2_FORMAL_EXPLORE, MAX_STAGE_GROUP_HISTORY))
    if len(recent_generations) < STAGE2_GATE_WINDOW_GENERATIONS:
        return False
    if current_entry_group_count < STAGE_REFERENCE_MIN_GROUPS[STAGE2_FORMAL_EXPLORE]:
        return False
    formal_rows = [item for item in recent_generations if bool(item.get("formal_success_candidate"))]
    qualified_rows = _stage2_target_qualified_rows(recent_generations)
    unique_target_families = len(_family_hash_set(qualified_rows, key="family_hash"))
    mean_dominant_share = _mean_dominant_share(recent_groups)
    mean_dominant_descriptor_share = _mean_dominant_descriptor_share(recent_groups)
    improving_groups = _count_group_improvements(recent_improvement_groups)
    return bool(
        len(qualified_rows) >= STAGE2_GATE_MIN_TARGET_COUNT
        and unique_target_families >= STAGE2_GATE_MIN_UNIQUE_TARGET_FAMILIES
        and improving_groups >= STAGE2_GATE_IMPROVING_GROUPS_REQUIRED
        and mean_dominant_share is not None
        and mean_dominant_share <= 0.45
        and mean_dominant_descriptor_share is not None
        and mean_dominant_descriptor_share <= STAGE2_GATE_MAX_DOMINANT_DESCRIPTOR_SHARE
    )


def _stage_gate_snapshot() -> Dict[str, Any]:
    stage_name = str(current_stage_name)
    recent_generations = _recent_stage_generation_window(
        stage_name,
        STAGE1_GATE_WINDOW_GENERATIONS if stage_name == STAGE1_STRUCTURE_EXPLORE else STAGE2_GATE_WINDOW_GENERATIONS,
    )
    recent_groups = _recent_stage_group_window(stage_name, 5)
    discovery_rows = [item for item in recent_generations if bool(item.get("discovery_candidate"))]
    formal_rows = [item for item in recent_generations if bool(item.get("formal_success_candidate"))]
    qualified_target_rows = _stage2_target_qualified_rows(recent_generations)
    return {
        "stage_name": stage_name,
        "stage_index": RL_STAGE_TO_INDEX.get(stage_name, 0),
        "recent_generation_count": len(recent_generations),
        "recent_executable_count": sum(1 for item in recent_generations if bool(item.get("executable_candidate"))),
        "recent_trainable_count": sum(
            1
            for item in recent_generations
            if bool(item.get("trained_step_ok") or item.get("backward_ok"))
        ),
        "recent_discovery_count": len(discovery_rows),
        "recent_unique_discovery_families": len(_family_hash_set(discovery_rows, key="family_hash")),
        "recent_formal_success_count": len(formal_rows),
        "recent_unique_formal_families": len(_family_hash_set(formal_rows, key="family_hash")),
        "recent_target_qualified_count": len(qualified_target_rows),
        "recent_unique_target_families": len(_family_hash_set(qualified_target_rows, key="family_hash")),
        "recent_unique_backbone_signatures": len(_family_hash_set(recent_generations, key="backbone_signature")),
        "recent_unique_cnn_signatures": len(_family_hash_set(recent_generations, key="cnn_signature")),
        "recent_unique_backbone_cnn_pairs": len(_family_hash_set(recent_generations, key="backbone_cnn_pair_key")),
        "recent_mean_dominant_family_share": _mean_dominant_share(recent_groups),
        "recent_mean_dominant_descriptor_share": _mean_dominant_descriptor_share(recent_groups),
        "recent_mean_dominant_backbone_share": _mean_dominant_backbone_share(recent_groups),
        "recent_mean_dominant_cnn_share": _mean_dominant_cnn_share(recent_groups),
        "recent_mean_dominant_backbone_cnn_share": _mean_dominant_backbone_cnn_share(recent_groups),
        "recent_improving_groups": _count_group_improvements(_recent_stage_group_window(stage_name, 4)),
        "recovery_active": bool(recovery_active),
    }


def _transition_to_stage(
    new_stage_name: str,
    *,
    event: str,
    reason: str,
    group_progress_payload: Optional[Dict[str, Any]] = None,
) -> None:
    return StageState.transition_to_stage(
        sys.modules[__name__],
        new_stage_name,
        event=event,
        reason=reason,
        group_progress_payload=group_progress_payload,
    )


def _maybe_update_stage_best_checkpoint(group_progress_payload: Dict[str, Any]) -> None:
    return StageState.maybe_update_stage_best_checkpoint(sys.modules[__name__], group_progress_payload)


def _evaluate_stage_transitions(group_progress_payload: Dict[str, Any]) -> None:
    return StageState.evaluate_stage_transitions(sys.modules[__name__], group_progress_payload)


def close_reward_group_if_needed() -> Optional[Dict[str, Any]]:
    return StageState.close_reward_group_if_needed(sys.modules[__name__])


def _coerce_accuracy_baseline(value: Any, *, context: str) -> float:
    if value is None:
        raise ValueError(f"{context}: missing required sample accuracy baseline")
    if isinstance(value, bool):
        raise ValueError(f"{context}: accuracy baseline must be numeric, got bool")
    try:
        baseline = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: accuracy baseline must be numeric, got {value!r}") from exc
    if baseline != baseline or baseline in {float("inf"), float("-inf")}:
        raise ValueError(f"{context}: accuracy baseline must be finite, got {value!r}")
    return baseline


def require_sample_accuracy_baselines(kwargs: Dict[str, Any], expected_count: int) -> List[float]:
    if "accuracy" not in kwargs:
        raise ValueError("compute_reward requires kwargs['accuracy'] for every sample")
    raw_values = kwargs["accuracy"]
    if len(raw_values) != expected_count:
        raise ValueError(
            f"compute_reward expected {expected_count} accuracy baselines, got {len(raw_values)}"
        )
    return [
        _coerce_accuracy_baseline(value, context=f"completion[{idx}]")
        for idx, value in enumerate(raw_values)
    ]


def run_epoch_dir(*args):
    root_override = os.getenv("NNGPT_RL_EPOCH_ROOT")
    if root_override:
        e_dir = Path(root_override)
        for d in args:
            e_dir = e_dir / f"A{d}"
        return e_dir
    return epoch_dir(*args)


def run_log_dir() -> str:
    return os.getenv("NNGPT_RL_LOG_DIR", "rl_output")


def run_model_out() -> str:
    return os.getenv("NNGPT_RL_MODEL_OUT", SAVED_MODEL_PATH)


def reward_run_epoch_dir(*args):
    return _reward_task_callable("run_epoch_dir", run_epoch_dir)(*args)


def reward_run_log_dir() -> str:
    return str(_reward_task_callable("run_log_dir", run_log_dir)())


def reward_run_model_out() -> str:
    return str(_reward_task_callable("run_model_out", run_model_out)())


def _resolve_resume_checkpoint_dir() -> Optional[Path]:
    explicit_dir = os.getenv("NNGPT_RL_RESUME_CHECKPOINT_DIR", "").strip()
    resume_stage = os.getenv("NNGPT_RL_RESUME_STAGE", "").strip()
    if explicit_dir:
        return Path(explicit_dir).expanduser().resolve()
    if resume_stage:
        return _stage_checkpoint_dir(resume_stage)
    return None


def apply_resume_stage_override(requested_stage: str, *, log_prefix: str) -> bool:
    global current_stage_name

    requested_stage = str(requested_stage or "").strip()
    if not requested_stage:
        return False
    current_state_stage = str(current_stage_name)
    if current_state_stage == requested_stage:
        return False
    print(
        f"{log_prefix} Resume stage override "
        f"checkpoint_stage={current_state_stage} requested_stage={requested_stage}"
    )
    current_stage_name = requested_stage
    return True


def _load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _current_stage_index() -> int:
    return StageState.current_stage_index(sys.modules[__name__])


def _history_trim_in_place(items: List[Dict[str, Any]], *, limit: int) -> None:
    return StageState.history_trim_in_place(sys.modules[__name__], items, limit=limit)


def _append_stage_event(payload: Dict[str, Any]) -> Dict[str, Any]:
    return StageState.append_stage_event(sys.modules[__name__], payload)


def _record_generation_event(payload: Dict[str, Any]) -> Dict[str, Any]:
    return StageState.record_generation_event(sys.modules[__name__], payload)


def _record_closed_group_event(payload: Dict[str, Any]) -> Dict[str, Any]:
    return StageState.record_closed_group_event(sys.modules[__name__], payload)


def _recent_stage_generation_window(stage_name: str, max_items: int) -> List[Dict[str, Any]]:
    return StageState.recent_stage_generation_window(sys.modules[__name__], stage_name, max_items)


def _recent_stage_group_window(stage_name: str, max_items: int) -> List[Dict[str, Any]]:
    return StageState.recent_stage_group_window(sys.modules[__name__], stage_name, max_items)


def _family_hash_set(items: List[Dict[str, Any]], *, key: str) -> Set[str]:
    return {
        str(item.get(key))
        for item in items
        if item.get(key)
    }


def _mean_dominant_share(items: List[Dict[str, Any]]) -> Optional[float]:
    shares = [
        float(item.get("dominant_family_share"))
        for item in items
        if item.get("dominant_family_share") is not None
    ]
    if not shares:
        return None
    return float(sum(shares) / len(shares))


def _mean_dominant_descriptor_share(items: List[Dict[str, Any]]) -> Optional[float]:
    shares = [
        float(item.get("dominant_descriptor_share"))
        for item in items
        if item.get("dominant_descriptor_share") is not None
    ]
    if not shares:
        return None
    return float(sum(shares) / len(shares))


def _mean_dominant_backbone_share(items: List[Dict[str, Any]]) -> Optional[float]:
    shares = [
        float(item.get("dominant_backbone_share"))
        for item in items
        if item.get("dominant_backbone_share") is not None
    ]
    if not shares:
        return None
    return float(sum(shares) / len(shares))


def _mean_dominant_cnn_share(items: List[Dict[str, Any]]) -> Optional[float]:
    shares = [
        float(item.get("dominant_cnn_share"))
        for item in items
        if item.get("dominant_cnn_share") is not None
    ]
    if not shares:
        return None
    return float(sum(shares) / len(shares))


def _mean_dominant_backbone_cnn_share(items: List[Dict[str, Any]]) -> Optional[float]:
    shares = [
        float(item.get("dominant_backbone_cnn_share"))
        for item in items
        if item.get("dominant_backbone_cnn_share") is not None
    ]
    if not shares:
        return None
    return float(sum(shares) / len(shares))


def _count_group_improvements(items: List[Dict[str, Any]]) -> int:
    count = 0
    for item in items:
        improvement_vs_prev = item.get("improvement_vs_prev")
        if improvement_vs_prev is None:
            continue
        if float(improvement_vs_prev) >= GROUP_IMPROVEMENT_DELTA:
            count += 1
    return count


def _training_context_metric_from_event(item: Dict[str, Any]) -> Tuple[Optional[str], Optional[float]]:
    best_epoch_loss = _optional_float(item.get("best_epoch_loss"))
    if best_epoch_loss is not None:
        return "best_epoch_loss", best_epoch_loss
    loss_end = _optional_float(item.get("loss_end"))
    if loss_end is not None:
        return "loss_end", loss_end
    metric_name = str(item.get("training_context_metric_name") or "").strip()
    metric_value = _optional_float(item.get("training_context_metric_value"))
    if metric_name and metric_value is not None:
        return metric_name, metric_value
    return None, None


def _recent_stage_trainable_metric_window(stage_name: str, max_items: int) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for item in _recent_stage_generation_window(stage_name, max_items):
        metric_name, metric_value = _training_context_metric_from_event(item)
        if metric_value is None:
            continue
        if not bool(item.get("backward_ok") or item.get("trained_step_ok") or item.get("loss_drop_ok")):
            continue
        epochs_completed = max(1, int(item.get("epochs_completed", 0) or 1))
        records.append(
            {
                "generation_total": int(item.get("generation_total", 0) or 0),
                "metric_name": metric_name or "best_epoch_loss",
                "metric_value": float(metric_value),
                "epochs_completed": epochs_completed,
            }
        )
    return records


def _series_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def _series_variance(values: List[float]) -> Optional[float]:
    if not values:
        return None
    mean_value = _series_mean(values)
    if mean_value is None:
        return None
    return float(sum((float(value) - mean_value) ** 2 for value in values) / len(values))


def _series_slope(values: List[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    n = len(values)
    x_mean = float(n - 1) / 2.0
    y_mean = _series_mean(values)
    if y_mean is None:
        return None
    numerator = 0.0
    denominator = 0.0
    for index, value in enumerate(values):
        x_delta = float(index) - x_mean
        numerator += x_delta * (float(value) - y_mean)
        denominator += x_delta * x_delta
    if denominator <= 0.0:
        return None
    return float(numerator / denominator)


def _epochs_since_last_best(records: List[Dict[str, Any]]) -> Optional[int]:
    if not records:
        return None
    best_value = None
    total_epochs = 0
    last_best_epoch = 0
    for item in records:
        epochs_completed = max(1, int(item.get("epochs_completed", 0) or 1))
        total_epochs += epochs_completed
        metric_value = float(item["metric_value"])
        if best_value is None or metric_value < best_value - 1e-8:
            best_value = metric_value
            last_best_epoch = total_epochs
    return max(0, total_epochs - last_best_epoch)


def _history_exploration_pressure_from_summary(summary: Dict[str, Any]) -> float:
    if not summary.get("has_recent_window"):
        return 0.0
    pressure = 0.0
    delta_avg_loss = _optional_float(summary.get("delta_avg_loss"))
    if delta_avg_loss is not None:
        if delta_avg_loss >= 0.0:
            pressure += 0.35
        elif delta_avg_loss >= -0.01:
            pressure += 0.20
    loss_slope_recent = _optional_float(summary.get("loss_slope_recent"))
    if loss_slope_recent is not None:
        if loss_slope_recent >= 0.0:
            pressure += 0.25
        elif loss_slope_recent >= -5e-4:
            pressure += 0.12
    pressure += 0.30 * float(summary.get("plateau_score") or 0.0)
    pressure += 0.18 * float(summary.get("oscillation_score") or 0.0)
    epochs_since_last_improvement = summary.get("epochs_since_last_improvement")
    recent_window_epochs = max(1, int(summary.get("recent_window_epochs", 0) or 1))
    if epochs_since_last_improvement is not None:
        pressure += 0.25 * min(1.0, float(epochs_since_last_improvement) / float(recent_window_epochs))
    return _clip(pressure, 0.0, 1.0)


def summarize_stage_training_context(
    stage_name: str,
    *,
    window_size: int = TRAINING_CONTEXT_WINDOW,
) -> Dict[str, Any]:
    effective_window = max(1, int(window_size))
    records = _recent_stage_trainable_metric_window(stage_name, max_items=max(effective_window * 4, effective_window))
    recent_records = records[-effective_window:]
    prev_records = records[-(effective_window * 2):-effective_window] if len(records) > effective_window else []
    recent_values = [float(item["metric_value"]) for item in recent_records]
    prev_values = [float(item["metric_value"]) for item in prev_records]
    recent_window_epochs = sum(max(1, int(item.get("epochs_completed", 0) or 1)) for item in recent_records)
    prev_window_epochs = sum(max(1, int(item.get("epochs_completed", 0) or 1)) for item in prev_records)
    recent_best_loss = min(recent_values) if recent_values else None
    prev_best_loss = min(prev_values) if prev_values else None
    recent_avg_loss = _series_mean(recent_values)
    prev_avg_loss = _series_mean(prev_values)
    delta_best_loss = (
        float(recent_best_loss - prev_best_loss)
        if recent_best_loss is not None and prev_best_loss is not None
        else None
    )
    delta_avg_loss = (
        float(recent_avg_loss - prev_avg_loss)
        if recent_avg_loss is not None and prev_avg_loss is not None
        else None
    )
    improvement_rate = (
        float((prev_avg_loss - recent_avg_loss) / float(max(1, recent_window_epochs)))
        if recent_avg_loss is not None and prev_avg_loss is not None
        else None
    )
    loss_slope_recent = _series_slope(recent_values)
    loss_variance_recent = _series_variance(recent_values)
    epochs_since_last_improvement = _epochs_since_last_best(records)
    recent_range = (max(recent_values) - min(recent_values)) if len(recent_values) >= 2 else 0.0
    recent_scale = max(1e-6, abs(recent_avg_loss if recent_avg_loss is not None else (recent_best_loss or 1.0)))
    normalized_slope = 0.0
    if loss_slope_recent is not None:
        normalized_slope = min(1.0, abs(float(loss_slope_recent)) * float(max(1, len(recent_values))) / recent_scale)
    normalized_range = min(1.0, float(recent_range) / recent_scale)
    plateau_score = _clip(1.0 - min(1.0, 0.65 * normalized_slope + 0.35 * normalized_range), 0.0, 1.0)
    diffs = [recent_values[index + 1] - recent_values[index] for index in range(max(0, len(recent_values) - 1))]
    nontrivial_diffs = [float(value) for value in diffs if abs(float(value)) > 1e-8]
    diff_signs = [1 if value > 0.0 else -1 for value in nontrivial_diffs]
    oscillation_score = 0.0
    if len(diff_signs) >= 2:
        sign_changes = sum(1 for left, right in zip(diff_signs, diff_signs[1:]) if left != right)
        oscillation_score = float(sign_changes) / float(len(diff_signs) - 1)
    monotonic_improving = bool(nontrivial_diffs) and all(value < 0.0 for value in nontrivial_diffs)
    metric_name = recent_records[-1]["metric_name"] if recent_records else "best_epoch_loss"
    summary = {
        "stage_name": str(stage_name),
        "metric_name": metric_name,
        "sample_count": len(records),
        "recent_window_size": len(recent_records),
        "compare_window_size": len(prev_records),
        "recent_window_epochs": recent_window_epochs,
        "compare_window_epochs": prev_window_epochs,
        "has_recent_window": len(recent_records) >= max(1, TRAINING_CONTEXT_MIN_POINTS),
        "has_compare_window": len(prev_records) >= max(1, TRAINING_CONTEXT_MIN_POINTS),
        "recent_best_loss": recent_best_loss,
        "prev_best_loss": prev_best_loss,
        "delta_best_loss": delta_best_loss,
        "recent_avg_loss": recent_avg_loss,
        "prev_avg_loss": prev_avg_loss,
        "delta_avg_loss": delta_avg_loss,
        "improvement_rate": improvement_rate,
        "loss_slope_recent": loss_slope_recent,
        "loss_variance_recent": loss_variance_recent,
        "epochs_since_last_improvement": epochs_since_last_improvement,
        "plateau_score": plateau_score,
        "oscillation_score": oscillation_score,
        "monotonic_improving": monotonic_improving,
    }
    summary["exploration_pressure"] = _history_exploration_pressure_from_summary(summary)
    return summary


def _training_context_guidance(summary: Dict[str, Any]) -> str:
    if not summary.get("has_recent_window"):
        return "no train-loss window yet"
    pressure = float(summary.get("exploration_pressure") or 0.0)
    if pressure >= 0.60:
        return "loss has plateaued or oscillated; favor structurally new candidates and avoid dominant templates"
    if bool(summary.get("monotonic_improving")) and pressure <= 0.25:
        return "loss is still improving; keep local mutations and avoid collapsing to one family"
    return "improvement is slowing; bias toward descriptor and family novelty over shallow repeats"


def _stage_reward_target_metric(stage_name: str) -> str:
    if str(stage_name) == STAGE1_STRUCTURE_EXPLORE:
        return STATIC_STAGE_REWARD_TARGET_METRIC
    return FORMAL_STAGE_REWARD_TARGET_METRIC


def _stage_uses_formal_eval(stage_name: str) -> bool:
    return str(stage_name) in {STAGE2_FORMAL_EXPLORE, STAGE3_FORMAL_OPTIMIZE}


def _stage_uses_static_only(stage_name: str) -> bool:
    return str(stage_name) == STAGE1_STRUCTURE_EXPLORE


def _iter_text_candidates(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        out: List[str] = []
        for item in value.values():
            out.extend(_iter_text_candidates(item))
        return out
    if isinstance(value, (list, tuple, set)):
        out: List[str] = []
        for item in value:
            out.extend(_iter_text_candidates(item))
        return out
    return []


def _score_seed_source_candidate(field_name: str, text: str) -> int:
    lowered_field = (field_name or "").lower()
    lowered_text = text.lower()
    score = 0
    if "<init>" in lowered_text and "<forward>" in lowered_text:
        score += 100
    if "class net" in lowered_text and "def forward" in lowered_text:
        score += 80
    if "def __init__" in lowered_text and "def forward" in lowered_text:
        score += 60
    if any(token in lowered_field for token in ("completion", "response", "output", "assistant", "xml")):
        score += 25
    if any(token in lowered_field for token in ("code", "nn", "model", "content", "text")):
        score += 10
    if len(text) > 200:
        score += 5
    return score


def _extract_method_from_module_text(source_text: str, class_name: str, method_name: str) -> str:
    try:
        tree = ast.parse(source_text)
    except Exception:
        return ""

    lines = source_text.splitlines()
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    if item.end_lineno is None:
                        return ""
                    snippet = "\n".join(lines[item.lineno - 1:item.end_lineno])
                    return textwrap.dedent(snippet).strip()
    return ""


def _extract_seed_init_forward_from_text(text: str) -> Tuple[str, str]:
    candidate = clean_block(text)
    if not candidate:
        return "", ""

    _, init_code, forward_code = extract_reward_completion_blocks(candidate)
    if init_code and forward_code:
        return init_code, forward_code

    stripped = candidate.replace("```python", "").replace("```", "").strip()
    init_code = _extract_method_from_module_text(stripped, "Net", "__init__")
    forward_code = _extract_method_from_module_text(stripped, "Net", "forward")
    return init_code, forward_code


def _extract_seed_candidates_from_row(row: Any) -> List[str]:
    row_dict = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    ranked: List[Tuple[int, str]] = []
    seen: Set[str] = set()

    for key, value in row_dict.items():
        for text in _iter_text_candidates(value):
            stripped = text.strip()
            if not stripped or stripped in seen:
                continue
            score = _score_seed_source_candidate(str(key), stripped)
            if score <= 0:
                continue
            ranked.append((score, stripped))
            seen.add(stripped)

    ranked.sort(key=lambda item: (-item[0], -len(item[1])))
    return [text for _, text in ranked]


def bootstrap_trainset_reference_library(data) -> None:
    reward_runtime = _backbone_reward_runtime()
    reward_runtime.train_graph_hashes.clear()
    reward_runtime.train_family_hashes.clear()
    reward_runtime.train_descriptor_keys.clear()

    stats = {
        "rows_seen": 0,
        "rows_parsed": 0,
        "rows_skipped": 0,
        "candidate_texts": 0,
    }

    for _, row in data.iterrows():
        stats["rows_seen"] += 1
        candidates = _extract_seed_candidates_from_row(row)
        stats["candidate_texts"] += len(candidates)
        parsed_ok = False

        for candidate in candidates:
            init_code, forward_code = _extract_seed_init_forward_from_text(candidate)
            if not init_code or not forward_code:
                continue
            graph_info = extract_graph_info(
                init_code,
                forward_code,
                legacy_patterns=SFTUtil.legacy_patterns,
            )
            if not graph_info.parse_ok:
                continue
            reward_runtime.train_graph_hashes.add(graph_info.graph_hash)
            reward_runtime.train_family_hashes.add(graph_info.family_hash)
            reward_runtime.train_descriptor_keys.add(graph_info.descriptor_key)
            parsed_ok = True
            break

        if parsed_ok:
            stats["rows_parsed"] += 1
        else:
            stats["rows_skipped"] += 1

    train_reference_stats.clear()
    train_reference_stats.update(stats)
    print(
        "[Trainset Reference] "
        f"rows={stats['rows_seen']}, parsed={stats['rows_parsed']}, skipped={stats['rows_skipped']}, "
        f"graph_hashes={len(reward_runtime.train_graph_hashes)}, family_hashes={len(reward_runtime.train_family_hashes)}, "
        f"descriptor_keys={len(reward_runtime.train_descriptor_keys)}"
    )


def extract_prompt_goal_tags(prompt_text: str) -> List[str]:
    if not prompt_text:
        return []
    match = re.search(
        r"(?:^|\n)\s*(?:-\s*)?(?:(?:Discovery|Optimization)\s+)?Target Tags:\s*([A-Za-z0-9_, \-]+)",
        prompt_text,
        flags=re.IGNORECASE,
    )
    if not match:
        return []
    return [tag.strip() for tag in match.group(1).split(",") if tag.strip()]


def extract_prompt_target_pattern(prompt_text: str) -> str:
    if not prompt_text:
        return ""
    match = re.search(
        r"(?:^|\n)\s*(?:-\s*)?Target pattern:\s*`?([A-Za-z0-9_-]+)`?",
        prompt_text,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    return match.group(1).strip()


def _compact_graph_expr(graph_expr: str) -> str:
    return re.sub(r"\s+", "", str(graph_expr or ""))


def _graph_has_block_before_backbone(graph_expr: str) -> bool:
    compact = _compact_graph_expr(graph_expr)
    return bool(
        re.search(
            r"Backbone\[[^\]]+\]\(_feature_to_input_image\((?:Sequential\[Block\]|Block|Fractal)",
            compact,
        )
    )


def _graph_has_backbone_before_block(graph_expr: str) -> bool:
    compact = _compact_graph_expr(graph_expr)
    return bool(
        re.search(
            r"(?:Sequential\[Block\]|Block|Fractal)\(_feature_to_input_image\(Backbone\[[^\]]+\]",
            compact,
        )
    )


def build_actual_structure_signature(
    graph_info,
    *,
    block_contributes_to_forward: bool,
    block_signature: str = "",
) -> str:
    if graph_info is None or not getattr(graph_info, "parse_ok", False):
        return "incomplete|block_unknown|bb0|d0|incomplete"
    block_state = "block_live" if block_contributes_to_forward else "block_dead"
    block_key = str(block_signature or "").strip() if block_contributes_to_forward else "incomplete_block"
    if not block_key:
        block_key = "unknown_block"
    return "|".join(
        [
            str(getattr(graph_info, "family_id", "") or "UnknownFamily"),
            str(getattr(graph_info, "descriptor_key", "") or "unknown_descriptor"),
            f"bb{int(getattr(graph_info, 'backbone_calls', 0) or 0)}",
            block_state,
            str(getattr(graph_info, "family_hash", "") or "unknown_family_hash")[:12],
            str(getattr(graph_info, "cnn_signature", "") or "unknown_cnn")[:12],
            str(block_key)[:12],
        ]
    )


def detect_target_structure(
    *,
    prompt_target_pattern: str,
    graph_info,
    block_contributes_to_forward: bool,
    block_signature: str = "",
) -> Dict[str, Any]:
    prompt_target_pattern = str(prompt_target_pattern or "").strip()
    normalized_target = normalize_pattern_name(prompt_target_pattern) if prompt_target_pattern else ""
    declared_pattern = str(getattr(graph_info, "pattern_name", "") or "") if graph_info is not None else ""
    actual_pattern = str(getattr(graph_info, "suggested_pattern_name", "") or "") if graph_info is not None else ""
    actual_signature = build_actual_structure_signature(
        graph_info,
        block_contributes_to_forward=bool(block_contributes_to_forward),
        block_signature=block_signature,
    )
    result = {
        "prompt_target_pattern": prompt_target_pattern,
        "normalized_prompt_target_pattern": normalized_target,
        "declared_pattern": declared_pattern,
        "actual_pattern": actual_pattern,
        "declared_pattern_matches_prompt": (
            bool(normalized_target)
            and normalize_pattern_name(declared_pattern) == normalized_target
        ),
        "actual_structure_signature": actual_signature,
        "target_structure_key": f"{normalized_target or 'open'}::{actual_signature}",
        "target_structure_match": True,
        "target_structure_mismatch_reasons": [],
        "actual_backbone_calls": int(getattr(graph_info, "backbone_calls", 0) or 0) if graph_info is not None else 0,
        "actual_block_live": bool(block_contributes_to_forward),
        "actual_block_before_backbone": False,
        "actual_backbone_before_block": False,
    }
    if not normalized_target:
        return result
    if graph_info is None or not getattr(graph_info, "parse_ok", False):
        result["target_structure_match"] = False
        result["target_structure_mismatch_reasons"] = ["graph_parse_failed"]
        return result

    graph_expr = str(getattr(graph_info, "graph_expr", "") or "")
    block_before_backbone = _graph_has_block_before_backbone(graph_expr)
    backbone_before_block = _graph_has_backbone_before_block(graph_expr)
    result["actual_block_before_backbone"] = block_before_backbone
    result["actual_backbone_before_block"] = backbone_before_block

    reasons: List[str] = []
    target_needs_block = "Fractal" in normalized_target
    target_needs_dual_backbone = (
        "DualBackbone" in normalized_target
        or "_to_B" in normalized_target
        or "_plus_B" in normalized_target
        or normalized_target.endswith("_plus_A")
    )
    if target_needs_block and not block_contributes_to_forward:
        reasons.append("target_fractal_but_block_dead")
    if target_needs_dual_backbone and int(getattr(graph_info, "backbone_calls", 0) or 0) < 2:
        reasons.append("target_dual_but_forward_uses_less_than_two_backbones")
    result["target_structure_match"] = not reasons
    result["target_structure_mismatch_reasons"] = reasons
    return result


def _target_structure_penalty(reasons: List[str]) -> float:
    reason_set = set(str(reason) for reason in (reasons or []))
    penalty = 0.0
    if "graph_parse_failed" in reason_set:
        penalty += TARGET_STRUCTURE_PARSE_PENALTY
    if "target_fractal_but_block_dead" in reason_set:
        penalty += TARGET_STRUCTURE_DEAD_BLOCK_PENALTY
    if "target_dual_but_forward_uses_less_than_two_backbones" in reason_set:
        penalty += TARGET_STRUCTURE_DUAL_BACKBONE_PENALTY
    if any(
        reason in reason_set
        for reason in (
            "fractal_not_before_backbone",
            "missing_backbone_to_fractal_path",
            "missing_fractal_to_backbone_path",
            "missing_backbone_to_fractal_branch",
        )
    ):
        penalty += TARGET_STRUCTURE_PATH_PENALTY
    return max(TARGET_STRUCTURE_PENALTY_FLOOR, penalty)


def _apply_target_structure_reward_adjustment(
    pattern_detection: Dict[str, Any],
    r_structure_group: float,
    r_structure_archive: float,
) -> Tuple[float, float, float, float]:
    if bool(pattern_detection.get("target_structure_match", True)):
        return r_structure_group, r_structure_archive, 0.0, 0.0
    suppressed_positive = max(0.0, float(r_structure_group or 0.0)) + max(
        0.0,
        float(r_structure_archive or 0.0),
    )
    return (
        min(0.0, float(r_structure_group or 0.0)),
        min(0.0, float(r_structure_archive or 0.0)),
        _target_structure_penalty(
            list(pattern_detection.get("target_structure_mismatch_reasons") or [])
        ),
        suppressed_positive,
    )


def _apply_target_structure_final_clamp(
    pattern_detection: Dict[str, Any],
    reward_value: float,
    r_target_structure_penalty: float,
) -> float:
    if bool(pattern_detection.get("target_structure_match", True)):
        return reward_value
    penalty = float(r_target_structure_penalty or 0.0)
    if penalty == 0.0:
        penalty = _target_structure_penalty(
            list(pattern_detection.get("target_structure_mismatch_reasons") or [])
        )
    return min(float(reward_value), penalty, 0.0)


def _apply_target_structure_reward_gate(res: Dict[str, Any], reward_value: float) -> float:
    return _apply_target_structure_final_clamp(
        res,
        reward_value,
        float(res.get("r_target_structure_penalty", 0.0) or 0.0),
    )


def _stage23_local_competition_reward(
    total_reward: float,
    *,
    generation_total: int,
    target_ok: bool,
    has_formal_epoch: bool,
    formal_success_candidate: bool,
    quality_acc_value: Optional[float],
    cell_archive_freq: int,
    batch_same_cell_count: int,
    cell_best_quality_acc: Optional[float],
) -> float:
    if (
        not target_ok
        or not has_formal_epoch
        or not formal_success_candidate
        or quality_acc_value is None
    ):
        return float(total_reward)

    reward_value = float(total_reward)
    quality_value = float(quality_acc_value)
    cell_is_unique_new = bool(cell_archive_freq <= 0 and batch_same_cell_count <= 1)
    cell_improved = bool(
        cell_archive_freq > 0
        and cell_best_quality_acc is not None
        and quality_value >= float(cell_best_quality_acc) + STAGE23_CELL_IMPROVEMENT_DELTA
    )

    if cell_is_unique_new:
        reward_value += STAGE23_NEW_CELL_BONUS
    elif cell_improved:
        reward_value += STAGE23_CELL_IMPROVEMENT_BONUS
    elif int(generation_total or 0) < STAGE23_EARLY_LOCAL_COMPETITION_GENERATIONS:
        reward_value = min(reward_value, STAGE23_EARLY_CELL_REPEAT_REWARD_CAP)
    elif quality_value < STAGE23_DUPLICATE_LOW_ACC_THRESHOLD:
        reward_value = min(reward_value, STAGE23_DUPLICATE_LOW_ACC_REWARD_CAP)

    if quality_value >= STAGE23_HIGH_ACC_ELITE_THRESHOLD:
        reward_value += STAGE23_HIGH_ACC_ELITE_BONUS
    elif quality_value >= STAGE23_HIGH_ACC_STRONG_THRESHOLD:
        reward_value += STAGE23_HIGH_ACC_STRONG_BONUS
    elif quality_value >= STAGE23_HIGH_ACC_BONUS_THRESHOLD:
        reward_value += STAGE23_HIGH_ACC_BONUS

    return _clip(reward_value, -2.0, 2.0)


def _stage23_gate_positive_novelty_by_quality(
    quality_acc_value: Optional[float],
    components: Dict[str, float],
) -> Dict[str, float]:
    if quality_acc_value is not None and float(quality_acc_value) >= STAGE23_POSITIVE_NOVELTY_ACC_THRESHOLD:
        return dict(components)
    return {
        key: min(float(value or 0.0), 0.0)
        for key, value in components.items()
    }


def prompt_goal_satisfied(graph_info, tag: str) -> bool:
    if not graph_info or not graph_info.parse_ok:
        return False
    if tag == "stem":
        return graph_info.stem_calls > 0
    if tag == "project":
        return graph_info.project_calls > 0
    if tag == "multi_stage":
        return is_multi_stage_architecture(graph_info)
    if tag == "fractal_deep":
        return graph_info.fractal_calls >= 2 or (graph_info.fractal_calls >= 1 and graph_info.depth >= 5)
    if tag == "branch_reuse":
        return graph_info.merges >= 2 or (graph_info.project_calls > 0 and graph_info.fuse_calls >= 2)
    if tag == "single_backbone":
        return graph_info.backbone_calls == 1
    if tag == "wide_fuse":
        return graph_info.max_fan_in >= 3 and graph_info.fuse_calls >= 1
    return False


def primary_goal_key(prompt_goal_tags: List[str], prompt_target_pattern: str = "") -> str:
    tags = [str(tag).strip() for tag in (prompt_goal_tags or []) if str(tag).strip()]
    if tags:
        return "__".join(tags)
    normalized_target = normalize_pattern_name(prompt_target_pattern) if prompt_target_pattern else ""
    return normalized_target or "open"


def goal_family_save_cap(graph_info) -> int:
    return 2


def get_goal_counter(store: Dict[str, Counter], goal_key: str) -> Counter:
    if goal_key not in store:
        store[goal_key] = Counter()
    return store[goal_key]


def clean_block(text: str) -> str:
    return _backbone_reward_runtime().clean_block(text)


def extract_completion_blocks(completion: str) -> Tuple[str, str, str]:
    return _backbone_reward_runtime().extract_completion_blocks(completion)


def extract_reward_completion_blocks(completion: str) -> Tuple[str, str, str]:
    return _reward_task_callable("extract_completion_blocks", extract_completion_blocks)(completion)


def render_completion_xml(block_code: str, init_code: str, forward_code: str) -> str:
    return _backbone_reward_runtime().render_completion_xml(block_code, init_code, forward_code)


def reconstruct_code(
    completion: str,
    *,
    pattern_name_override: str = "",
) -> str:
    return _backbone_reward_runtime().reconstruct_code(
        completion,
        pattern_name_override=pattern_name_override,
    )


def _compute_build_partial_reward(res: Dict[str, Any]) -> float:
    error_str = str(res.get('error', ''))
    error_lower = error_str.lower()
    error_stage = str(res.get("error_stage") or "")
    error_context = dict(res.get("error_context") or {})
    code_trace = dict(error_context.get("code_trace") or {})
    raw_extraction = dict(res.get("raw_extraction") or {})
    build_partial = 0.0

    if error_stage == "cpu_prevalidate":
        if "must call self.infer_dimensions_dynamically" in error_str:
            return -0.12
        elif "infer_dimensions_dynamically() takes 2 positional arguments but 3 were given" in error_str:
            build_partial = -0.04
        elif "has no attribute '_input_spec'" in error_lower:
            build_partial = -0.12
        elif "has no attribute '_output_dim'" in error_lower or "has no attribute '_input_dim'" in error_lower:
            build_partial = -0.09
        elif "has no attribute 'infer_dimensions'" in error_lower:
            build_partial = -0.08
        elif "nameerror" in error_lower and any(
            token in error_str for token in ("dropout_prob", "in_channels", "features", "out_channels")
        ):
            build_partial = -0.06
        elif "keyerror" in error_lower and "out_channels" in error_lower:
            build_partial = -0.06
        elif "runtimeerror" in error_lower and "expected input" in error_lower and "to have" in error_lower:
            build_partial = -0.05
        else:
            build_partial = -0.10

        if bool(raw_extraction.get("dual_backbone_ok")):
            build_partial += 0.02
        if bool(raw_extraction.get("xml_tag_exact")):
            build_partial += 0.01
        if bool(raw_extraction.get("exact_init_signature")):
            build_partial += 0.02
        if bool(raw_extraction.get("exact_forward_signature")):
            build_partial += 0.01
        if bool(code_trace.get("assigns_input_spec")):
            build_partial += 0.03
        elif bool(code_trace.get("references_input_spec")):
            build_partial -= 0.02

        return _clip(build_partial, -0.12, 0.12)

    if 'SyntaxError' in error_str:
        build_partial = -0.3
    elif 'NameError' in error_str or 'ImportError' in error_str:
        build_partial = -0.2
    elif 'TypeError' in error_str:
        build_partial = -0.1
    elif 'RuntimeError' in error_str and 'shape' in error_str.lower():
        build_partial = 0.05
    elif error_str:
        build_partial = -0.15
    return build_partial


def _compute_warmup_dense_reward(test_acc: Optional[float]) -> Optional[float]:
    if test_acc is None:
        return None
    return max(0.05, min(0.30, 0.08 + 0.55 * float(test_acc)))


def _is_minimal_backbone_classifier_template(init_code: str) -> bool:
    significant_lines = []
    for raw_line in textwrap.dedent(init_code or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(
            (
                "def __init__",
                "super().__init__",
                "self.device",
                "self.use_amp",
                "self._input_spec",
                "self.pattern",
                "self.infer_dimensions",
            )
        ):
            continue
        significant_lines.append(line)
    assignment_lines = [line for line in significant_lines if line.startswith("self.")]
    if len(assignment_lines) > 3:
        return False
    has_backbone_a = any("self.backbone_a" in line for line in assignment_lines)
    has_backbone_b = any("self.backbone_b" in line for line in assignment_lines)
    has_classifier = any("self.classifier" in line for line in assignment_lines)
    if not (has_backbone_a and has_backbone_b and has_classifier):
        return False
    non_core_assignments = [
        line
        for line in assignment_lines
        if all(token not in line for token in ("self.backbone_a", "self.backbone_b", "self.classifier"))
    ]
    return not non_core_assignments


def _stage1_validity_scale(res: Dict[str, Any]) -> float:
    if bool(
        res.get("built_ok")
        and res.get("forward_shape_ok")
        and (
            res.get("backward_ok")
            or res.get("trained_step_ok")
            or _has_completed_formal_epoch(res)
        )
    ):
        return 1.0
    if bool(res.get("backward_ok") or res.get("trained_step_ok") or _has_completed_formal_epoch(res)):
        return 0.45
    if bool(res.get("forward_shape_ok")):
        return 0.15
    return 0.0


def _stage1_validity_reward(res: Dict[str, Any], graph_info) -> float:
    if not graph_info or not graph_info.parse_ok:
        return -0.85
    if not res.get("built_ok"):
        build_partial = float(res.get("r_build_partial", 0.0) or 0.0)
        return min(-0.35, -0.55 + build_partial)
    if not res.get("forward_ok"):
        return -0.40
    if not res.get("forward_shape_ok"):
        return -0.30
    if not _stage1_trainability_ok(res, graph_info):
        return -0.04
    return max(STAGE1_EXECUTABLE_BONUS, 0.12)


def _template_penalty(
    *,
    stage_name: str,
    shallow_one_shot: bool,
    minimal_init_template: bool,
) -> float:
    penalty = 0.0
    if shallow_one_shot:
        penalty += -0.08 if stage_name == STAGE1_STRUCTURE_EXPLORE else -0.05
    if minimal_init_template:
        penalty += -0.10 if stage_name == STAGE1_STRUCTURE_EXPLORE else -0.08
    return penalty


def _history_context_reward(
    *,
    stage_name: str,
    training_context: Dict[str, Any],
    executable_candidate: bool,
    formal_success_candidate: bool,
    discovery_candidate: bool,
    novel_vs_trainset_family: bool,
    novel_vs_trainset_graph: bool,
    dominant_family_repeat: bool,
    dominant_descriptor_repeat: bool,
    shallow_one_shot: bool,
    plain_parallel_repeat: bool,
    minimal_init_template: bool,
    batch_same_descriptor_count: int,
    validity_scale: float = 1.0,
) -> float:
    return 0.0


def _goal_tag_match_stats(graph_info, prompt_goal_tags: Optional[List[str]]) -> Tuple[int, int, float]:
    tags = list(prompt_goal_tags or [])
    if not tags:
        return 0, 0, 0.0
    hit_count = sum(1 for tag in tags if prompt_goal_satisfied(graph_info, tag))
    total_count = len(tags)
    hit_rate = float(hit_count) / float(total_count) if total_count > 0 else 0.0
    return hit_count, total_count, hit_rate


def _discovery_failure_result(
    reward: float,
    error: str,
    *,
    seed_accuracy_baseline: float,
    backbone_model_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "reward": reward,
        "built_ok": False,
        "forward_ok": False,
        "forward_shape_ok": False,
        "trained_step_ok": False,
        "backward_ok": False,
        "loss_start": None,
        "loss_end": None,
        "loss_drop": None,
        "loss_drop_ok": False,
        "best_epoch_loss": None,
        "avg_epoch_loss": None,
        "epochs_completed": 0,
        "epoch_loss_series": [],
        "training_context_metric_name": "best_epoch_loss",
        "training_context_metric_value": None,
        "test_acc": None,
        "train_acc": None,
        "frozen_train_acc": None,
        "frozen_test_acc": None,
        "unfrozen_train_acc": None,
        "unfrozen_test_acc": None,
        "frozen_eval": None,
        "unfrozen_eval": None,
        "seed_accuracy_baseline": seed_accuracy_baseline,
        "seed_train_acc_gap": None,
        "seed_train_acc_improved": False,
        "accuracy_baseline": seed_accuracy_baseline,
        "train_acc_gain": None,
        "train_acc_improved": False,
        "group_baseline_train_acc": None,
        "group_train_acc_gain": None,
        "group_train_acc_improved": False,
        "group_baseline_reward_target_acc": None,
        "group_reward_target_gain": None,
        "group_reward_target_improved": False,
        "reward_batch_index": None,
        "reward_group_id": None,
        "group_warmup": False,
        "val_metric": None,
        "latency_ms": None,
        "params_m": None,
        "timed_out": False,
        "estimated_total_seconds": None,
        "eval_limit_seconds": None,
        "warmup_dense_reward": None,
        "backbone_model_names": list(backbone_model_names or []),
        "reward_target_metric": _stage_reward_target_metric(current_stage_name),
        "reward_target_value": None,
        "best_closed_group_mean_reward_target_acc": best_closed_group_mean_reward_target_acc,
        "best_closed_group_mean_train_acc": best_closed_group_mean_train_acc,
        "best_closed_group_mean_test_acc": best_closed_group_mean_test_acc,
        "best_reward_target_for_goal": None,
        "r_dense": 0.0,
        "r_prev_group": 0.0,
        "r_best_group": 0.0,
        "r_goal_best": 0.0,
        "r_goal_match": 0.0,
        "r_trainset_novelty": 0.0,
        "r_generalization": 0.0,
        "r_structure_group": 0.0,
        "r_structure_archive": 0.0,
        "r_descriptor_diversity": 0.0,
        "r_block_diversity": 0.0,
        "r_batch_elite": 0.0,
        "r_repeat_family": 0.0,
        "r_plain_fuse_penalty": 0.0,
        "r_template_penalty": 0.0,
        "r_history_context": 0.0,
        "r_no_progress_penalty": 0.0,
        "batch_elite_rank": None,
        "batch_elite_tier": "none",
        "batch_elite_threshold_passed": False,
        "goal_tag_hit_count": 0,
        "goal_tag_total_count": 0,
        "goal_tag_hit_rate": 0.0,
        "prev_target_reward_target_acc": None,
        "best_target_reward_target_acc": None,
        "open_discovery": {
            "r_primary": 0.0,
            "r_tiebreak": 0.0,
            "r_trainset_novelty": 0.0,
            "r_dense": 0.0,
            "r_prev_group": 0.0,
            "r_best_group": 0.0,
            "r_goal_best": 0.0,
            "r_goal_match": 0.0,
            "r_generalization": 0.0,
            "r_structure_group": 0.0,
            "r_structure_archive": 0.0,
            "r_descriptor_diversity": 0.0,
            "r_block_diversity": 0.0,
            "r_batch_elite": 0.0,
            "r_repeat_family": 0.0,
            "r_plain_fuse_penalty": 0.0,
            "r_template_penalty": 0.0,
            "r_history_context": 0.0,
            "r_no_progress_penalty": 0.0,
            "batch_elite_rank": None,
            "batch_elite_tier": "none",
            "batch_elite_threshold_passed": False,
            "novel_vs_trainset_family": False,
            "novel_vs_trainset_graph": False,
            "archive_snapshot_family_freq": 0,
            "batch_same_family_count": 0,
            "reward_target_metric": _stage_reward_target_metric(current_stage_name),
            "reward_target_value": None,
            "goal_tag_hit_count": 0,
            "goal_tag_total_count": 0,
            "goal_tag_hit_rate": 0.0,
        },
        "error": error,
        "current_stage_name": current_stage_name,
        "current_stage_index": _current_stage_index(),
        "stage_uses_formal_eval": _stage_uses_formal_eval(current_stage_name),
        "stage_uses_static_only": _stage_uses_static_only(current_stage_name),
    }


def _is_trainable_candidate(res: Dict[str, Any], graph_info) -> bool:
    return _stage1_trainability_ok(res, graph_info)


def _has_completed_formal_epoch(res: Dict[str, Any]) -> bool:
    try:
        return int(res.get("epochs_completed", 0) or 0) >= 1
    except (TypeError, ValueError):
        return False


def _is_executable_candidate(res: Dict[str, Any], graph_info) -> bool:
    return bool(
        graph_info
        and graph_info.parse_ok
        and res.get("built_ok")
        and res.get("forward_shape_ok")
    )


def _stage2_target_qualified_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    qualified_rows: List[Dict[str, Any]] = []
    for item in rows:
        if not bool(item.get("executable_candidate")):
            continue
        if not _has_completed_formal_epoch(item):
            continue
        reward_target_value = _optional_float(item.get("reward_target_value"))
        if reward_target_value is None or float(reward_target_value) < STAGE2_GATE_MIN_REWARD_TARGET:
            continue
        qualified_rows.append(item)
    return qualified_rows


def _stage1_trainability_ok(res: Dict[str, Any], graph_info) -> bool:
    return bool(
        graph_info
        and graph_info.parse_ok
        and res.get("built_ok")
        and res.get("forward_shape_ok")
        and (
            res.get("backward_ok")
            or res.get("trained_step_ok")
            or _has_completed_formal_epoch(res)
        )
    )


def _apply_trainability_clamp(res: Dict[str, Any], reward_value: float, graph_info) -> float:
    parse_ok = bool(graph_info and graph_info.parse_ok)
    if not parse_ok:
        return min(reward_value, -0.30)
    if not res.get("built_ok"):
        build_partial = float(res.get("r_build_partial", 0.0))
        return min(reward_value, -0.70 + build_partial)
    if not res.get("forward_ok"):
        return min(reward_value, -0.30)
    if not res.get("forward_shape_ok"):
        return min(reward_value, -0.20)
    if not res.get("backward_ok"):
        loss_drop = _optional_float(res.get("loss_drop"))
        partial_progress = _clip(0.25 * float(loss_drop or 0.0), -0.04, 0.04)
        return min(reward_value, -0.12 + partial_progress)
    return reward_value


def _apply_stage1_trainability_clamp(res: Dict[str, Any], reward_value: float, graph_info) -> float:
    parse_ok = bool(graph_info and graph_info.parse_ok)
    if not parse_ok:
        return min(reward_value, -0.85)
    if not res.get("built_ok"):
        build_partial = float(res.get("r_build_partial", 0.0) or 0.0)
        return min(reward_value, -0.70 + build_partial)
    if not res.get("forward_ok"):
        return min(reward_value, -0.40)
    if not res.get("forward_shape_ok"):
        return min(reward_value, -0.30)
    if not _stage1_trainability_ok(res, graph_info):
        return min(reward_value, -0.04)
    return reward_value


def _apply_executability_clamp(res: Dict[str, Any], reward_value: float, graph_info) -> float:
    parse_ok = bool(graph_info and graph_info.parse_ok)
    if not parse_ok:
        return min(reward_value, -0.35)
    if not res.get("built_ok"):
        build_partial = float(res.get("r_build_partial", 0.0))
        return min(reward_value, -0.70 + build_partial)
    if not res.get("forward_ok"):
        return min(reward_value, -0.28)
    if not res.get("forward_shape_ok"):
        return min(reward_value, -0.16)
    return reward_value


def _stage_reward_profile(stage_name: str) -> Dict[str, float]:
    if stage_name == STAGE2_FORMAL_EXPLORE:
        return {
            "dense_scale": STAGE2_DENSE_SCALE,
            "prev_group_scale": STAGE2_PREV_GROUP_SCALE,
            "best_group_scale": STAGE2_BEST_GROUP_SCALE,
            "global_baseline_blend": STAGE2_GLOBAL_BASELINE_BLEND,
            "backbone_prev_group_scale": STAGE2_BACKBONE_PREV_GROUP_SCALE,
            "backbone_best_group_scale": STAGE2_BACKBONE_BEST_GROUP_SCALE,
            "goal_best_scale": STAGE2_GOAL_BEST_SCALE,
            "goal_match_scale": STAGE2_GOAL_MATCH_SCALE,
            "structure_scale": STAGE2_STRUCTURE_SCALE,
            "repeat_family_scale": STAGE2_REPEAT_FAMILY_SCALE,
            "plain_fuse_scale": STAGE2_PLAIN_FUSE_SCALE,
            "no_progress_scale": STAGE2_NO_PROGRESS_SCALE,
            "non_improving_cap": STAGE2_NON_IMPROVING_CAP,
            "descriptor_non_improving_cap": STAGE2_DESCRIPTOR_NON_IMPROVING_CAP,
        }
    return {
        "dense_scale": STAGE3_DENSE_SCALE,
        "prev_group_scale": STAGE3_PREV_GROUP_SCALE,
        "best_group_scale": STAGE3_BEST_GROUP_SCALE,
        "global_baseline_blend": STAGE3_GLOBAL_BASELINE_BLEND,
        "backbone_prev_group_scale": STAGE3_BACKBONE_PREV_GROUP_SCALE,
        "backbone_best_group_scale": STAGE3_BACKBONE_BEST_GROUP_SCALE,
        "goal_best_scale": STAGE3_GOAL_BEST_SCALE,
        "goal_match_scale": STAGE3_GOAL_MATCH_SCALE,
        "structure_scale": STAGE3_STRUCTURE_SCALE,
        "repeat_family_scale": STAGE3_REPEAT_FAMILY_SCALE,
        "plain_fuse_scale": STAGE3_PLAIN_FUSE_SCALE,
        "no_progress_scale": STAGE3_NO_PROGRESS_SCALE,
        "non_improving_cap": STAGE3_NON_IMPROVING_CAP,
        "descriptor_non_improving_cap": STAGE3_DESCRIPTOR_NON_IMPROVING_CAP,
    }


def _archive_rarity_bonus_stage1(archive_snapshot_family_freq: int) -> float:
    if archive_snapshot_family_freq <= 0:
        return STRUCTURE_ARCHIVE_RARITY_STRONG_BONUS
    if archive_snapshot_family_freq == 1:
        return STRUCTURE_ARCHIVE_RARITY_MEDIUM_BONUS
    if archive_snapshot_family_freq <= 3:
        return STRUCTURE_ARCHIVE_RARITY_LIGHT_BONUS
    return 0.0


def _archive_rarity_bonus_formal(archive_snapshot_family_freq: int) -> float:
    return min(
        STAGE23_STRUCTURE_ARCHIVE_RARITY_CAP,
        STAGE23_STRUCTURE_ARCHIVE_RARITY_CAP / math.sqrt(float(archive_snapshot_family_freq) + 1.0),
    )


def _structure_progress_components(
    graph_info,
    *,
    batch_same_family_count: int,
    archive_snapshot_family_freq: int,
    novel_vs_trainset_family: bool,
    novel_vs_trainset_graph: bool,
    shallow_one_shot: bool,
    use_formal_archive_bonus: bool = False,
) -> Tuple[float, float]:
    if not graph_info or not graph_info.parse_ok:
        return 0.0, 0.0

    r_structure_group = 0.0
    if passes_macro_structure_gate(graph_info):
        r_structure_group += STRUCTURE_MACRO_BONUS
    if is_multi_stage_architecture(graph_info):
        r_structure_group += STRUCTURE_MULTI_STAGE_BONUS
    if has_structural_motif(graph_info):
        r_structure_group += STRUCTURE_MOTIF_BONUS
    if batch_same_family_count <= 1:
        r_structure_group += STRUCTURE_BATCH_DIVERSITY_BONUS
    elif batch_same_family_count == 2:
        r_structure_group += STRUCTURE_BATCH_DIVERSITY_BONUS * 0.5
    task_context = reward_task_group_context_fields()
    current_dominant_family_hash = task_context.get("dominant_family_hash")
    current_dominant_family_share = task_context.get("dominant_family_share")
    if (
        current_dominant_family_hash
        and graph_info.family_hash != current_dominant_family_hash
        and float(current_dominant_family_share or 0.0) >= 0.20
    ):
        r_structure_group += STRUCTURE_NON_DOMINANT_FAMILY_BONUS
    if shallow_one_shot:
        r_structure_group = max(0.0, r_structure_group - 0.02)

    r_structure_archive = 0.0
    if novel_vs_trainset_family:
        r_structure_archive += TRAINSET_NOVEL_FAMILY_BONUS
    elif novel_vs_trainset_graph:
        r_structure_archive += TRAINSET_NOVEL_GRAPH_BONUS
    archive_bonus = _archive_rarity_bonus_formal if use_formal_archive_bonus else _archive_rarity_bonus_stage1
    r_structure_archive += archive_bonus(archive_snapshot_family_freq)

    return _clip(r_structure_group, 0.0, 0.14), _clip(r_structure_archive, 0.0, 0.08)


def _recompute_discovery_reward(
    res: Dict[str, Any],
    graph_info,
) -> Tuple[float, float, float]:
    stage_name = str(res.get("current_stage_name") or current_stage_name)
    r_primary = (
        float(res.get("r_dense", 0.0) or 0.0)
        + float(res.get("r_prev_group", 0.0) or 0.0)
        + float(res.get("r_best_group", 0.0) or 0.0)
        + float(res.get("r_prev_backbone_group", 0.0) or 0.0)
        + float(res.get("r_best_backbone_group", 0.0) or 0.0)
        + float(res.get("r_goal_best", 0.0) or 0.0)
        + float(res.get("r_generalization", 0.0) or 0.0)
        + float(res.get("r_structure_group", 0.0) or 0.0)
        + float(res.get("r_structure_archive", 0.0) or 0.0)
        + float(res.get("r_descriptor_diversity", 0.0) or 0.0)
        + float(res.get("r_cnn_diversity", 0.0) or 0.0)
        + float(res.get("r_block_diversity", 0.0) or 0.0)
        + float(res.get("r_batch_elite", 0.0) or 0.0)
        + float(res.get("r_repeat_family", 0.0) or 0.0)
        + float(res.get("r_plain_fuse_penalty", 0.0) or 0.0)
        + float(res.get("r_target_structure_penalty", 0.0) or 0.0)
        + float(res.get("r_template_penalty", 0.0) or 0.0)
        + float(res.get("r_history_context", 0.0) or 0.0)
        + float(res.get("r_no_progress_penalty", 0.0) or 0.0)
    )
    r_tiebreak = float(res.get("r_goal_match", 0.0) or 0.0)
    total_reward = _clip(r_primary + r_tiebreak, -2.0, 2.0)
    if stage_name == STAGE1_STRUCTURE_EXPLORE:
        total_reward = _apply_stage1_trainability_clamp(res, total_reward, graph_info)
    elif stage_name == STAGE2_FORMAL_EXPLORE:
        if _reward_variant_is_strong_repeat_penalty() and _is_strong_repeat_without_refresh(res):
            total_reward = min(total_reward, 0.0)
            res["strong_repeat_penalty_applied"] = True
        total_reward = _apply_executability_clamp(res, total_reward, graph_info)
    else:
        if _reward_variant_is_strong_repeat_penalty() and _is_strong_repeat_without_refresh(res):
            total_reward = min(total_reward, 0.0)
            res["strong_repeat_penalty_applied"] = True
        total_reward = _apply_trainability_clamp(res, total_reward, graph_info)
    total_reward = _apply_target_structure_reward_gate(res, total_reward)
    return total_reward, r_primary, r_tiebreak


def build_stage_eval_cfg(
    *,
    stage_name: Optional[str] = None,
    in_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
    out_shape: Tuple[int, ...] = (10,),
    prm: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
    cfg: Optional[EvalConfig] = None,
) -> EvalConfig:
    del cfg
    requested_stage = str(stage_name or current_stage_name)
    resolved_device = str(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if requested_stage == STAGE1_STRUCTURE_EXPLORE:
        eval_limit_seconds = env_int("NNGPT_RL_STAGE1_EVAL_LIMIT_SECONDS", 120)
        formal_epoch_limit_minutes = None
    else:
        eval_limit_seconds = env_int("NNGPT_RL_FORMAL_EVAL_LIMIT_SECONDS", 1800)
        configured_epoch_limit = env_float("NNGPT_RL_FORMAL_EPOCH_LIMIT_MINUTES", 0.0)
        formal_epoch_limit_minutes = configured_epoch_limit if configured_epoch_limit > 0.0 else None
    return EvalConfig(
        device=resolved_device,
        input_shape=tuple(in_shape),
        n_classes=int(out_shape[0]),
        train_epochs=int((prm or {}).get("epoch", 1) or 1),
        default_batch_size=int((prm or {}).get("batch", 32) or 32),
        eval_limit_seconds=eval_limit_seconds,
        reward_target_metric=_stage_reward_target_metric(requested_stage),
        formal_nn_eval=_stage_uses_formal_eval(requested_stage),
        static_only=_stage_uses_static_only(requested_stage),
        formal_task=os.getenv("NNGPT_RL_FORMAL_TASK", "img-classification"),
        formal_dataset=os.getenv("NNGPT_RL_FORMAL_DATASET", "cifar-10"),
        formal_metric=os.getenv("NNGPT_RL_FORMAL_METRIC", "acc"),
        formal_epoch_limit_minutes=formal_epoch_limit_minutes,
    )


def _invoke_eval_cfg_builder(eval_cfg_builder, **kwargs) -> EvalConfig:
    if not callable(eval_cfg_builder):
        raise TypeError("eval_cfg_builder must be callable")

    try:
        signature = inspect.signature(eval_cfg_builder)
    except (TypeError, ValueError):
        return eval_cfg_builder(**kwargs)

    parameters = signature.parameters.values()
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return eval_cfg_builder(**kwargs)

    supported_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
        and signature.parameters[key].kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return eval_cfg_builder(**supported_kwargs)


def reward_eval_cfg_builder():
    return _reward_task_callable("build_eval_cfg", build_stage_eval_cfg)


def evaluate_reward_code(*args, **kwargs):
    return _reward_task_callable("evaluate_code_and_reward", evaluate_code_and_reward)(*args, **kwargs)


def evaluate_reward_code_batch(specs):
    return _reward_task_callable("evaluate_code_and_reward_batch", evaluate_code_and_reward_batch)(specs)


def reward_task_reward_fn(*args, **kwargs):
    return _reward_task_callable("reward_fn", reward_fn)(*args, **kwargs)


def load_reward_dataset(tokenizer):
    return _reward_task_callable("load_rl_dataset", load_rl_dataset)(tokenizer)


def _backbone_reward_runtime():
    from ab.gpt.rl_pipeline import backbone_reward_runtime as BackboneRewardRuntime

    return BackboneRewardRuntime


def extract_reward_seed_context(kwargs: Dict[str, Any], expected_count: int):
    return _reward_task_callable("extract_seed_context", require_sample_accuracy_baselines)(kwargs, expected_count)


def prepare_reward_entries(
    prompts,
    completions,
    *,
    seed_contexts,
    group_context: Dict[str, Any],
    precompute_eval: bool,
) -> List[Dict[str, Any]]:
    task = current_reward_task()
    method = getattr(task, "prepare_entries", None) if task is not None else None
    if callable(method):
        return method(
            prompts,
            completions,
            seed_contexts=seed_contexts,
            group_context=group_context,
            precompute_eval=precompute_eval,
        )
    return _backbone_reward_runtime().prepare_entries(
        prompts,
        completions,
        seed_contexts=seed_contexts,
        group_context=group_context,
        precompute_eval=precompute_eval,
    )


def precompute_reward_entries(entries: List[Dict[str, Any]], *, group_context: Dict[str, Any]) -> None:
    _reward_task_callable("precompute_entries", _backbone_reward_runtime().precompute_entries)(
        entries,
        group_context=group_context,
    )


def apply_reward_batch_elite_bonuses(scored_results: List[Dict[str, Any]], group_context: Dict[str, Any]) -> None:
    _reward_task_callable("apply_batch_elite_bonuses", _backbone_reward_runtime().apply_batch_elite_bonuses)(
        scored_results,
        group_context,
    )


def finalize_reward_scored_results(scored_results: List[Dict[str, Any]]) -> None:
    _reward_task_callable("finalize_scored_results", _backbone_reward_runtime().finalize_scored_results)(scored_results)


def reward_entries_from_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _reward_task_callable("entries_from_records", _backbone_reward_runtime().entries_from_records)(records)


def describe_reward_code_sections(*, block_code: str, init_code: str, forward_code: str) -> Dict[str, Any]:
    return _reward_task_callable("describe_code_sections", _backbone_reward_runtime().describe_code_sections)(
        block_code=block_code,
        init_code=init_code,
        forward_code=forward_code,
    )


def reward_task_capture_runtime_state() -> Dict[str, Any]:
    return _reward_task_callable("capture_runtime_state", _backbone_reward_runtime().capture_runtime_state)()


def reward_task_restore_runtime_state(state: Optional[Dict[str, Any]]) -> None:
    _reward_task_callable("restore_runtime_state", _backbone_reward_runtime().restore_runtime_state)(state)


def reward_task_reset_runtime_state() -> None:
    _reward_task_callable("reset_runtime_state", _backbone_reward_runtime().reset_runtime_state)()


def reward_task_group_context_fields() -> Dict[str, Any]:
    return _reward_task_callable("group_context_fields", _backbone_reward_runtime().group_context_fields)()


def reward_task_update_group_metrics(results: List[Dict[str, Any]]) -> None:
    _reward_task_callable("update_group_metrics", _backbone_reward_runtime().update_group_metrics)(results)


def reward_task_close_group_payload() -> Dict[str, Any]:
    return _reward_task_callable("close_group_payload", _backbone_reward_runtime().close_group_payload)()


def reward_task_reset_current_group_state() -> None:
    _reward_task_callable("reset_current_group_state", _backbone_reward_runtime().reset_current_group_state)()


def reward_task_reset_stage_comparison_state() -> None:
    _reward_task_callable("reset_stage_comparison_state", _backbone_reward_runtime().reset_stage_comparison_state)()


def reward_task_archive_snapshot_family_counts() -> Dict[str, int]:
    return _reward_task_callable("archive_snapshot_family_counts", _backbone_reward_runtime().archive_snapshot_family_counts)()


def _attach_group_context(
    res: Dict[str, Any],
    *,
    seed_accuracy_baseline: float,
    group_context: Dict[str, Any],
) -> Dict[str, Any]:
    return RewardPayload.attach_group_context(
        sys.modules[__name__],
        res,
        seed_accuracy_baseline=seed_accuracy_baseline,
        group_context=group_context,
    )

def reward_fn(
    completion: str,
    *,
    seed_accuracy_baseline: float,
    precomputed_eval_result: Optional[Dict[str, Any]] = None,
    graph_info=None,
    batch_graph_hashes: List[str] = None,
    batch_family_hashes: List[str] = None,
    batch_descriptor_keys: List[str] = None,
    batch_backbone_signatures: List[str] = None,
    batch_cnn_signatures: List[str] = None,
    batch_block_signatures: List[str] = None,
    batch_backbone_block_signatures: List[str] = None,
    prompt_goal_tags: List[str] = None,
    prompt_target_pattern: str = "",
    archive_snapshot_family_counts: Optional[Dict[str, int]] = None,
    archive_snapshot_descriptor_counts: Optional[Dict[str, int]] = None,
    archive_snapshot_backbone_signature_counts: Optional[Dict[str, int]] = None,
    archive_snapshot_cnn_signature_counts: Optional[Dict[str, int]] = None,
    archive_snapshot_graph_counts: Optional[Dict[str, int]] = None,
    archive_snapshot_block_signature_counts: Optional[Dict[str, int]] = None,
    archive_snapshot_backbone_cnn_pair_counts: Optional[Dict[str, int]] = None,
    archive_snapshot_backbone_block_pair_counts: Optional[Dict[str, int]] = None,
    archive_snapshot_backbone_block_best_quality: Optional[Dict[str, float]] = None,
    group_baseline_train_acc: Optional[float] = None,
    group_baseline_reward_target_acc: Optional[float] = None,
    reward_batch_index: Optional[int] = None,
    reward_group_id: Optional[int] = None,
    group_warmup: bool = False,
    completion_index: Optional[int] = None,
    batch_last_item: bool = False,
) -> Dict[str, Any]:
    """Reward open-ended motif discovery while keeping the existing XML output ABI."""
    from ab.gpt.rl_pipeline import backbone_reward_runtime as BackboneRewardRuntime

    return BackboneRewardRuntime.base_discovery_reward_fn(
        completion,
        seed_accuracy_baseline=seed_accuracy_baseline,
        precomputed_eval_result=precomputed_eval_result,
        graph_info=graph_info,
        batch_graph_hashes=batch_graph_hashes,
        batch_family_hashes=batch_family_hashes,
        batch_descriptor_keys=batch_descriptor_keys,
        batch_backbone_signatures=batch_backbone_signatures,
        batch_cnn_signatures=batch_cnn_signatures,
        batch_block_signatures=batch_block_signatures,
        batch_backbone_block_signatures=batch_backbone_block_signatures,
        prompt_goal_tags=prompt_goal_tags,
        prompt_target_pattern=prompt_target_pattern,
        archive_snapshot_family_counts=archive_snapshot_family_counts,
        archive_snapshot_descriptor_counts=archive_snapshot_descriptor_counts,
        archive_snapshot_backbone_signature_counts=archive_snapshot_backbone_signature_counts,
        archive_snapshot_cnn_signature_counts=archive_snapshot_cnn_signature_counts,
        archive_snapshot_graph_counts=archive_snapshot_graph_counts,
        archive_snapshot_block_signature_counts=archive_snapshot_block_signature_counts,
        archive_snapshot_backbone_cnn_pair_counts=archive_snapshot_backbone_cnn_pair_counts,
        archive_snapshot_backbone_block_pair_counts=archive_snapshot_backbone_block_pair_counts,
        archive_snapshot_backbone_block_best_quality=archive_snapshot_backbone_block_best_quality,
        group_baseline_train_acc=group_baseline_train_acc,
        group_baseline_reward_target_acc=group_baseline_reward_target_acc,
        reward_batch_index=reward_batch_index,
        reward_group_id=reward_group_id,
        group_warmup=group_warmup,
        completion_index=completion_index,
        batch_last_item=batch_last_item,
    )


def _is_repeated_block_without_refresh(res: Dict[str, Any]) -> bool:
    block_signature = str(res.get("block_signature") or "")
    if not block_signature or block_signature == "incomplete_block":
        return False
    if _has_block_repeat_quality_refresh(res):
        return False
    archive_freq = int(res.get("archive_snapshot_block_freq", 0) or 0)
    batch_count = int(res.get("batch_same_block_count", 0) or 0)
    return archive_freq > 0 or batch_count > 1


def _is_repeated_graph_without_refresh(res: Dict[str, Any]) -> bool:
    if bool(res.get("repeated_graph_without_refresh")):
        return True
    graph_hash = str(res.get("graph_hash") or "")
    if not graph_hash:
        return False
    if _has_block_repeat_quality_refresh(res):
        return False
    archive_freq = int(res.get("archive_snapshot_graph_freq", 0) or 0)
    batch_count = int(res.get("batch_same_graph_count", 0) or 0)
    return archive_freq > 0 or batch_count > 1


def _is_strong_repeat_without_refresh(res: Dict[str, Any]) -> bool:
    if not _has_completed_formal_epoch(res):
        return False
    if not bool(res.get("formal_success_candidate")):
        return False
    if _has_block_repeat_quality_refresh(res):
        return False
    return bool(
        res.get("dominant_descriptor_repeat")
        or res.get("dominant_cnn_repeat")
        or _is_repeated_block_without_refresh(res)
        or _is_repeated_graph_without_refresh(res)
    )


def _has_block_repeat_quality_refresh(res: Dict[str, Any]) -> bool:
    if float(res.get("r_goal_best", 0.0) or 0.0) > 0.0:
        return True
    reward_target_value = _optional_float(res.get("reward_target_value"))
    if reward_target_value is None:
        return False
    best_targets = (
        _optional_float(res.get("backbone_best_target_reward_target_acc")),
        _optional_float(res.get("best_target_reward_target_acc")),
    )
    return any(target is not None and reward_target_value >= target for target in best_targets)

def _reward_failure_result(
    *,
    error: str,
    seed_accuracy_baseline: float,
    group_context: Dict[str, Any],
) -> Dict[str, Any]:
    return _attach_group_context(
        {
            "reward": -1.0,
            "built_ok": False,
            "forward_ok": False,
            "forward_shape_ok": False,
            "trained_step_ok": False,
            "backward_ok": False,
            "loss_start": None,
            "loss_end": None,
            "loss_drop": None,
            "loss_drop_ok": False,
            "train_acc": None,
            "val_metric": None,
            "latency_ms": None,
            "params_m": None,
            "error": error,
        },
        seed_accuracy_baseline=seed_accuracy_baseline,
        group_context=group_context,
    )


def _build_global_reward_entries(gathered_entries: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    global_entries: List[Dict[str, Any]] = []
    for global_index, entry in enumerate(
        entry
        for rank_entries in gathered_entries
        for entry in list(rank_entries or [])
    ):
        merged_entry = dict(entry)
        merged_entry["global_index"] = global_index
        global_entries.append(merged_entry)
    return global_entries


def _select_global_reward_entries_for_rank(
    entries: List[Dict[str, Any]],
    *,
    rank: int,
    world_size: int,
) -> List[Dict[str, Any]]:
    total_entries = len(entries)
    start = (total_entries * int(rank)) // max(1, int(world_size))
    end = (total_entries * (int(rank) + 1)) // max(1, int(world_size))
    return [dict(entry) for entry in entries[start:end]]


def _merge_gathered_reward_entries(
    gathered_entries: List[List[Dict[str, Any]]],
    *,
    expected_count: Optional[int] = None,
) -> List[Dict[str, Any]]:
    merged_entries = [
        dict(entry)
        for rank_entries in gathered_entries
        for entry in list(rank_entries or [])
    ]
    merged_entries.sort(key=lambda entry: int(entry.get("global_index", -1)))
    if expected_count is not None and len(merged_entries) != int(expected_count):
        raise RuntimeError(
            f"Distributed reward merge expected {expected_count} entries, but received {len(merged_entries)}"
        )
    return merged_entries


def _format_reward_trace_context(context: Optional[Dict[str, Any]]) -> str:
    if not isinstance(context, dict) or not context:
        return ""
    preferred_keys = (
        "freeze_backbones",
        "formal_eval_backend",
        "formal_eval_duration_seconds",
        "trainer_device",
        "trainer_in_shape",
        "dataset_out_shape",
        "forward_output_shape",
        "params_m",
        "batch",
        "epoch",
        "epoch_limit_minutes",
        "transform",
        "num_workers",
        "estimated_training_time_minutes",
        "reported_accuracy",
        "reported_duration_seconds",
    )
    parts = []
    for key in preferred_keys:
        if key in context and context[key] is not None:
            parts.append(f"{key}={context[key]!r}")
    code_trace = context.get("code_trace")
    if isinstance(code_trace, dict):
        for key in ("references_input_spec", "assigns_input_spec", "references_pattern_attr", "line_count"):
            if key in code_trace and code_trace[key] is not None:
                parts.append(f"code_trace.{key}={code_trace[key]!r}")
    return ", ".join(parts)


def _log_reward_failure_trace(entry: Dict[str, Any], res: Dict[str, Any]) -> None:
    graph_info = entry.get("graph_info")
    pattern_name = getattr(graph_info, "suggested_pattern_name", None) if graph_info is not None else None
    branches = [("root", res)]
    frozen_eval = res.get("frozen_eval")
    unfrozen_eval = res.get("unfrozen_eval")
    if isinstance(frozen_eval, dict):
        branches.append(("frozen", frozen_eval))
    if isinstance(unfrozen_eval, dict):
        branches.append(("unfrozen", unfrozen_eval))

    seen = set()
    for branch_name, payload in branches:
        error = payload.get("error")
        if not error:
            continue
        stage = payload.get("error_stage")
        hint = payload.get("error_hint")
        context = payload.get("error_context")
        dedupe_key = (branch_name, str(error), str(stage), str(hint), repr(context))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        trace_message = (
            f"[Reward Failure Trace] rank={entry['rank']} "
            f"batch_index={entry['local_index']} "
            f"branch={branch_name} "
            f"pattern={pattern_name!r} "
            f"stage={stage or 'unknown'} "
            f"error={error!r}"
        )
        if hint:
            trace_message += f" hint={hint!r}"
        formatted_context = _format_reward_trace_context(context)
        if formatted_context:
            trace_message += f" context=({formatted_context})"
        code_logger.log_to_file(trace_message)


def score_reward_entries(
    entries: List[Dict[str, Any]],
    *,
    group_context: Dict[str, Any],
    archive_snapshot_family_counts: Dict[str, int],
) -> List[Dict[str, Any]]:
    return _reward_task_callable("score_entries", _backbone_reward_runtime().score_entries)(
        entries,
        group_context=group_context,
        archive_snapshot_family_counts=archive_snapshot_family_counts,
    )


def print_reward_metrics() -> None:
    _backbone_reward_runtime().print_discovery_metrics()


def compute_reward(prompts, completions, **kwargs):
    clear_reward_extraction_meta_cache()
    seed_contexts = extract_reward_seed_context(kwargs, len(completions))
    group_context = current_reward_group_context()

    try:
        expected_world_size = max(1, env_int("WORLD_SIZE", 1))
        distributed_mode = _distributed_initialized() and _distributed_world_size() > 1
        if expected_world_size > 1 and not distributed_mode:
            raise RuntimeError(
                "compute_reward expected an initialized torch.distributed process group "
                f"for WORLD_SIZE={expected_world_size}, but it is not initialized"
            )

        rank = _distributed_rank()
        precompute_eval = not distributed_mode
        if not precompute_eval:
            print(
                "[Reward Precompute Local] skip "
                f"rank={rank} "
                f"reward_batch_index={group_context.get('reward_batch_index')} "
                "reason='distributed_global_sharded_gpu_eval'"
            )
        local_entries = prepare_reward_entries(
            prompts,
            completions,
            seed_contexts=seed_contexts,
            group_context=group_context,
            precompute_eval=precompute_eval,
        )
        archive_snapshot_family_counts = reward_task_archive_snapshot_family_counts()

        if not distributed_mode:
            scored_results = score_reward_entries(
                local_entries,
                group_context=group_context,
                archive_snapshot_family_counts=archive_snapshot_family_counts,
            )
            rewards = [-1.0] * len(completions)
            for item in scored_results:
                rewards[int(item["local_index"])] = float(item["score"])
            finalize_reward_scored_results(scored_results)
            print_reward_metrics()
            return rewards

        print(
            "[Reward Gather] start "
            f"rank={rank} "
            f"reward_batch_index={group_context.get('reward_batch_index')} "
            f"local_entries={len(local_entries)}"
        )
        gathered_entries = _all_gather_object(local_entries)
        total_entries = sum(len(rank_entries or []) for rank_entries in gathered_entries)
        print(
            "[Reward Gather] end "
            f"rank={rank} "
            f"reward_batch_index={group_context.get('reward_batch_index')} "
            f"gathered_ranks={len(gathered_entries)} "
            f"total_entries={total_entries}"
        )

        global_entries = _build_global_reward_entries(gathered_entries)
        assigned_entries = _select_global_reward_entries_for_rank(
            global_entries,
            rank=rank,
            world_size=len(gathered_entries),
        )
        print(
            "[Reward Shard] start "
            f"rank={rank} "
            f"reward_batch_index={group_context.get('reward_batch_index')} "
            f"global_entries={len(global_entries)} "
            f"assigned_entries={len(assigned_entries)}"
        )
        precompute_reward_entries(
            assigned_entries,
            group_context=group_context,
        )
        print(
            "[Reward Shard] end "
            f"rank={rank} "
            f"reward_batch_index={group_context.get('reward_batch_index')} "
            f"assigned_entries={len(assigned_entries)}"
        )

        gathered_precomputed_entries = _all_gather_object(assigned_entries)
        merged_precomputed_entries = _merge_gathered_reward_entries(
            gathered_precomputed_entries,
            expected_count=len(global_entries),
        )

        if is_main_process():
            print(
                "[Reward Score] start "
                f"rank={rank} "
                f"reward_batch_index={group_context.get('reward_batch_index')} "
                f"entries={len(merged_precomputed_entries)}"
            )
            scored_results = score_reward_entries(
                merged_precomputed_entries,
                group_context=group_context,
                archive_snapshot_family_counts=archive_snapshot_family_counts,
            )
            finalize_reward_scored_results(scored_results)
            print_reward_metrics()
            print(
                "[Reward Score] end "
                f"rank={rank} "
                f"reward_batch_index={group_context.get('reward_batch_index')} "
                f"entries={len(scored_results)}"
            )

            rewards_by_rank: Dict[int, List[float]] = {
                world_rank: [-1.0] * len(gathered_entries[world_rank])
                for world_rank in range(len(gathered_entries))
            }
            for item in scored_results:
                rewards_by_rank[int(item["rank"])][int(item["local_index"])] = float(item["score"])

            broadcast_payload = {
                "rewards_by_rank": rewards_by_rank,
                "reward_state": capture_reward_runtime_state(),
            }
        else:
            broadcast_payload = None

        print(
            "[Reward Broadcast] start "
            f"rank={rank} "
            f"reward_batch_index={group_context.get('reward_batch_index')}"
        )
        synced_payload = _broadcast_object(broadcast_payload, src=0)
        print(
            "[Reward Broadcast] end "
            f"rank={rank} "
            f"reward_batch_index={group_context.get('reward_batch_index')}"
        )
        restore_reward_runtime_state(synced_payload.get("reward_state"))
        return list(synced_payload["rewards_by_rank"].get(rank, [-1.0] * len(completions)))
    finally:
        clear_reward_extraction_meta_cache()

def load_rl_dataset(tokenizer):
    return _backbone_reward_runtime().load_rl_dataset(tokenizer)


class OpenDiscoveryRewardTask:
    name = "open_discovery"

    @property
    def model_source(self) -> str:
        return base_model

    @property
    def tokenizer_source(self) -> str:
        return tokenizer_source

    @property
    def load_existing_model(self) -> bool:
        return LOAD_EXISTING_MODEL

    @property
    def saved_model_path(self) -> str:
        return SAVED_MODEL_PATH

    @property
    def prompt_template(self) -> str:
        return _backbone_reward_runtime().PROMPT_TEMPLATE

    def extract_completion_blocks(self, completion: str) -> Tuple[str, str, str]:
        return _backbone_reward_runtime().extract_completion_blocks(completion)

    def clear_extraction_meta_cache(self) -> None:
        clear_extraction_meta_cache()

    def evaluate_code_and_reward(self, *args, **kwargs):
        return evaluate_code_and_reward(*args, **kwargs)

    def evaluate_code_and_reward_batch(self, specs):
        return evaluate_code_and_reward_batch(specs)

    def build_eval_cfg(self, *args, **kwargs):
        return build_stage_eval_cfg(*args, **kwargs)

    def reward_fn(self, *args, **kwargs):
        return reward_fn(*args, **kwargs)

    def load_rl_dataset(self, tokenizer):
        return _backbone_reward_runtime().load_rl_dataset(tokenizer)

    def extract_seed_context(self, kwargs: Dict[str, Any], expected_count: int):
        return require_sample_accuracy_baselines(kwargs, expected_count)

    def prepare_entries(
        self,
        prompts,
        completions,
        *,
        seed_contexts,
        group_context: Dict[str, Any],
        precompute_eval: bool,
    ) -> List[Dict[str, Any]]:
        return _backbone_reward_runtime().prepare_entries(
            prompts,
            completions,
            seed_contexts=seed_contexts,
            group_context=group_context,
            precompute_eval=precompute_eval,
        )

    def precompute_entries(self, entries: List[Dict[str, Any]], *, group_context: Dict[str, Any]) -> None:
        _backbone_reward_runtime().precompute_entries(entries, group_context=group_context)

    def score_entries(
        self,
        entries: List[Dict[str, Any]],
        *,
        group_context: Dict[str, Any],
        archive_snapshot_family_counts: Dict[str, int],
    ) -> List[Dict[str, Any]]:
        return _backbone_reward_runtime().score_entries(
            entries,
            group_context=group_context,
            archive_snapshot_family_counts=archive_snapshot_family_counts,
        )

    def entries_from_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return _backbone_reward_runtime().entries_from_records(records)

    def describe_code_sections(self, *, block_code: str, init_code: str, forward_code: str) -> Dict[str, Any]:
        return _backbone_reward_runtime().describe_code_sections(
            block_code=block_code,
            init_code=init_code,
            forward_code=forward_code,
        )

    def apply_batch_elite_bonuses(self, scored_results: List[Dict[str, Any]], group_context: Dict[str, Any]) -> None:
        _backbone_reward_runtime().apply_batch_elite_bonuses(scored_results, group_context)

    def finalize_scored_results(self, scored_results: List[Dict[str, Any]]) -> None:
        _backbone_reward_runtime().finalize_scored_results(scored_results)

    def render_prompt_feedback_text(self, *, feedback_char_budget: int = 1200) -> str:
        return _backbone_reward_runtime().render_prompt_feedback_text(feedback_char_budget=feedback_char_budget)

    def run_log_dir(self) -> str:
        return run_log_dir()

    def run_model_out(self) -> str:
        return run_model_out()

    def run_epoch_dir(self, *args):
        return run_epoch_dir(*args)


def ensure_default_reward_task() -> None:
    if current_reward_task() is None:
        register_reward_task(OpenDiscoveryRewardTask())


ensure_default_reward_task()


def main():
    global active_rl_model
    global active_rl_tokenizer
    global current_stage_name

    ensure_default_reward_task()
    torch.cuda.empty_cache()
    resume_checkpoint_dir = _resolve_resume_checkpoint_dir()
    resume_manifest = None
    restored_reward_state_path = None
    resume_stage_override = os.getenv("NNGPT_RL_RESUME_STAGE", "").strip()
    if resume_checkpoint_dir is not None:
        resume_manifest = _load_json_if_exists(resume_checkpoint_dir / "runtime_manifest.json")
        if resume_manifest is None:
            resume_manifest = _load_json_if_exists(resume_checkpoint_dir / "stage_manifest.json")
        # Restore runtime state through the shared helper before stage-specific overrides run.
        restored_reward_state_path = TrainingRuntime.restore_or_reset_runtime_state(
            resume_checkpoint_dir,
            _reward_runtime_hooks(),
            legacy_state_filenames=("reward_state.json",),
        )
        if resume_stage_override:
            apply_resume_stage_override(resume_stage_override, log_prefix="[RL]")
        print(
            "[RL] Resuming from checkpoint "
            f"dir={resume_checkpoint_dir} stage={current_stage_name} "
            f"generation_total={_current_generation_total()} reward_batch_index={reward_batch_index}"
        )
        if restored_reward_state_path is not None:
            print(f"[RL] Restored runtime state from {restored_reward_state_path}")
    else:
        TrainingRuntime.restore_or_reset_runtime_state(
            None,
            _reward_runtime_hooks(),
            legacy_state_filenames=("reward_state.json",),
        )
        if resume_stage_override:
            apply_resume_stage_override(resume_stage_override, log_prefix="[RL]")
    precision = best_mixed_precision()
    runtime = get_distributed_runtime_info()
    runtime_settings = resolve_rl_runtime_settings(runtime)
    rank = int(runtime.get("rank", 0))
    local_rank = int(runtime.get("local_rank", 0))
    raw_local_rank = int(runtime.get("raw_local_rank", 0))
    world_size = int(runtime.get("world_size", 1))
    use_deepspeed = _resolve_rl_deepspeed_enabled(runtime)
    deepspeed_config_path = _resolve_rl_deepspeed_config_path() if use_deepspeed else None
    os.environ["NNGPT_SFT_USE_DEEPSPEED"] = "1" if use_deepspeed else "0"
    if deepspeed_config_path is not None:
        os.environ["NNGPT_SFT_DEEPSPEED_CONFIG"] = deepspeed_config_path
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    train_device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    hf_deepspeed_config = _maybe_init_hf_deepspeed_config(deepspeed_config_path) if use_deepspeed else None
    model_source = reward_model_source()
    tok_source = reward_tokenizer_source()
    adapter_path = reward_saved_model_path()

    print(f"Using RL base model: {model_source}")
    if tok_source != model_source:
        print(f"Using RL tokenizer: {tok_source}")
    print(
        "[RL] Distributed Runtime: "
        f"rank={rank} local_rank={local_rank} raw_local_rank={raw_local_rank} world_size={world_size}"
    )
    print(f"[RL] DeepSpeed Enabled: {use_deepspeed}")
    if deepspeed_config_path is not None:
        print(f"[RL] DeepSpeed Config: {deepspeed_config_path}")
    print(f"[RL] Fixed training device: {train_device}")
    print(f"[RL] Mixed precision: {precision['label']} (torch_dtype={precision['torch_dtype']})")
    print(f"[RL] Current stage: {current_stage_name}")
    print(
        "[RL] Runtime limits: "
        f"dataset_limit={runtime_settings['dataset_limit']} "
        f"max_completion_length={runtime_settings['max_completion_length']} "
        f"grad_accum={runtime_settings['grad_accum']} "
        f"effective_train_batch_size={runtime_settings['effective_train_batch_size']} "
        f"requested_global_num_generations={runtime_settings['requested_global_num_generations']} "
        f"global_num_generations={runtime_settings['global_num_generations']} "
        f"effective_global_num_generations={runtime_settings['effective_global_num_generations']}"
    )
    if runtime_settings["global_num_generations_adapted"]:
        print(
            "[RL] Generation plan adapted "
            f"requested={runtime_settings['requested_global_num_generations']} "
            f"effective={runtime_settings['effective_global_num_generations']} "
            f"valid_generation_values={runtime_settings['valid_generation_values']} "
            f"world_size={world_size}"
        )
    tokenizer = TrainerRuntime.load_tokenizer(tok_source)

    # Load RL dataset (limit for training speed)
    rl_dataset = load_reward_dataset(tokenizer)
    dataset_limit = runtime_settings["dataset_limit"]
    if len(rl_dataset) > dataset_limit:
        rl_dataset = rl_dataset.select(range(dataset_limit))

    model = TrainerRuntime.load_quantized_causal_lm(
        model_source=model_source,
        precision=precision,
        train_device=train_device,
        use_deepspeed=use_deepspeed,
    )
    _ = hf_deepspeed_config

    if reward_load_existing_model() and os.path.exists(adapter_path):
        model = TrainerRuntime.maybe_merge_initial_adapter(
            model,
            enabled=True,
            adapter_path=adapter_path,
            label="extra SFT",
            load_message=f"Loading extra SFT adapter from {adapter_path}...",
        )

    model = prepare_model_for_kbit_training(model)
    align_generation_head_dtype(model, precision["torch_dtype"])

    # Apply LoRA specifically for RL phase
    peft_config = TrainerRuntime.build_lora_config(
        r=16,
        alpha=32,
        dropout=0.05,
    )
    resume_adapter_dir = (resume_checkpoint_dir / "adapter") if resume_checkpoint_dir is not None else None
    model = TrainerRuntime.attach_or_resume_lora(
        model,
        peft_config=peft_config,
        stage_adapter_dir=resume_adapter_dir,
        log_prefix="[RL]",
        missing_adapter_message=f"Missing adapter directory under resume checkpoint: {resume_adapter_dir}",
        load_message=f"[RL] Loading RL adapter from {resume_adapter_dir}..." if resume_adapter_dir is not None else None,
    )
    align_generation_head_dtype(model, precision["torch_dtype"])

    # Enable gradient checkpointing to save memory
    TrainerRuntime.enable_non_reentrant_gradient_checkpointing(
        model,
        log_prefix="[RL]",
    )

    model.print_trainable_parameters()
    active_rl_model = model
    active_rl_tokenizer = tokenizer
    stage_entry_generation_totals.setdefault(current_stage_name, _current_generation_total())
    stage_entry_reward_batches.setdefault(current_stage_name, reward_batch_index)
    if resume_checkpoint_dir is None:
        _append_stage_event(
            {
                "event": "entered",
                "reason": "initial_stage_entry",
                "previous_stage_name": None,
                "next_stage_name": current_stage_name,
            }
        )
        _save_stage_checkpoint(
            "entered",
            stage_name=current_stage_name,
            reason="initial_stage_entry",
        )

    grpo_config = _build_rl_grpo_config(
        precision=precision,
        use_deepspeed=use_deepspeed,
        deepspeed_config_path=deepspeed_config_path,
        runtime_settings=runtime_settings,
    )

    trainer = GRPOTrainer(
        model=model,
        train_dataset=rl_dataset,
        reward_funcs=compute_reward,
        args=grpo_config,
    )
    trainer_gc_patch_stats = TrainerRuntime.enforce_non_reentrant_gradient_checkpointing(trainer.model)
    print(
        "[RL] Trainer gradient checkpointing enforcement: "
        f"roots={trainer_gc_patch_stats['roots']} modules={trainer_gc_patch_stats['modules']} use_reentrant=False"
    )
    prewarm_eval_workers(timeout_seconds=60.0, require_gpu=True)
    register_stage_checkpoint_signal_handlers()

    print("Starting GRPO training for Backbone Search...")
    try:
        TrainerRuntime.train_grpo(
            trainer=trainer,
            trainer_checkpoint=None,
            log_prefix="[RL]",
        )
    except Exception as exc:
        if is_cuda_oom_error(exc):
            log_cuda_oom_diagnostics("rl/trainer.train", exc)
        raise
    finally:
        shutdown_eval_worker()

    model_out = reward_run_model_out()
    print(f"Saving model to {model_out}...")
    model.save_pretrained(model_out)
    _save_stage_checkpoint(
        "completed",
        stage_name=current_stage_name,
        reason="trainer_completed",
    )
    try:
        code_logger.save_log()
    except Exception as exc:
        code_logger.log_to_file(f"[RL] save_log failed: {type(exc).__name__}: {exc}")
    print("Model saved successfully!")

    return model

if __name__ == "__main__":
    from ab.gpt.util.simple_logger import SimpleCodeLogger
    from ab.gpt.util.Reward import evaluate_code_and_reward
    from typing import Dict

    # Ensure directories exist
    log_dir = reward_run_log_dir()
    os.makedirs(log_dir, exist_ok=True)
    code_logger = SimpleCodeLogger(log_dir)

    # 清空旧模型目录
    if _resolve_resume_checkpoint_dir() is None:
        print(f"Cleaning existing models in {reward_run_epoch_dir()}...")
        shutil.rmtree(reward_run_epoch_dir(), ignore_errors=True)
    else:
        print(f"Resuming run: keeping existing synthesized models under {reward_run_epoch_dir()}")

    main()
