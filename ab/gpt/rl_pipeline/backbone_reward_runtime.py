from __future__ import annotations

import ast
import hashlib
import inspect
import math
import os
import re
import textwrap
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from datasets import Dataset
from ab.gpt.rl_pipeline import stage_state as StageState
from ab.gpt.rl_pipeline.completion import extract_completion_blocks_strict
from ab.gpt.util.ArchDiscovery import ensure_pattern_name, extract_graph_info, normalize_pattern_name
from ab.gpt.util.Const import new_nn_file, new_out_file, synth_dir
from ab.gpt.util.Reward import FORMAL_MULTI_HORIZON_REWARD_TARGET_METRIC
import ab.gpt.util.SFTUtil as SFTUtil
from ab.gpt.util.Util import extract_str
from ab.nn.util.Util import create_file
import ab.nn.api as api


graph_archive_counts = Counter()
family_archive_counts = Counter()
family_hash_archive_counts = Counter()
descriptor_archive_counts = Counter()
backbone_signature_archive_counts = Counter()
cnn_signature_archive_counts = Counter()
block_signature_archive_counts = Counter()
backbone_cnn_pair_archive_counts = Counter()
backbone_block_pair_archive_counts = Counter()
family_metric_best: Dict[str, float] = {}
motif_name_counts = Counter()
saved_graph_counts = Counter()
saved_family_hash_counts = Counter()
saved_backbone_signature_counts = Counter()
saved_cnn_signature_counts = Counter()
saved_backbone_cnn_pair_counts = Counter()
saved_backbone_block_pair_counts = Counter()
goal_graph_archive_counts: Dict[str, Counter] = {}
goal_family_hash_archive_counts: Dict[str, Counter] = {}
saved_goal_family_hash_counts: Dict[str, Counter] = {}
train_graph_hashes: Set[str] = set()
train_family_hashes: Set[str] = set()
train_descriptor_keys: Set[str] = set()
train_reference_stats: Dict[str, int] = {}
current_group_reward_target_sum_by_backbone: Dict[str, float] = {}
current_group_reward_target_count_by_backbone = Counter()
prev_closed_group_mean_reward_target_by_backbone: Dict[str, float] = {}
best_closed_group_mean_reward_target_by_backbone: Dict[str, float] = {}
saved_best_reward_target_by_backbone_cnn: Dict[str, float] = {}
best_quality_acc_by_backbone_block: Dict[str, float] = {}
dominant_family_hash: Optional[str] = None
dominant_family_share: float = 0.0
dominant_descriptor_key: Optional[str] = None
dominant_descriptor_share: float = 0.0
dominant_backbone_signature: Optional[str] = None
dominant_backbone_share: float = 0.0
dominant_cnn_signature: Optional[str] = None
dominant_cnn_share: float = 0.0
dominant_backbone_cnn_pair: Optional[str] = None
dominant_backbone_cnn_share: float = 0.0
discovery_family_hashes_seen: Set[str] = set()
PROMPT_TEMPLATE = SFTUtil.open_discovery_prompt_template
PROMPT_BLOCK_SIGNATURE = "def drop_conv3x3_block(in_channels, out_channels, stride=1, padding=1, bias=False, dropout_prob=0.0):"
PROMPT_INIT_SIGNATURE = "def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:"
PROMPT_FORWARD_SIGNATURE = "def forward(self, x: torch.Tensor, is_probing: bool = False) -> torch.Tensor:"
GROUP_IMPROVEMENT_DELTA = 0.003
BEST_GROUP_REFRESH_DELTA = 0.0015
GOAL_REFRESH_DELTA = 0.0015
FORMAL_REWARD_TRANSFORM = "norm_128_flip"
BACKBONE_BASELINE_MIN_ARCHIVE_SAMPLES = 3
SAVE_DUPLICATE_BACKBONE_CNN_DELTA = 0.002
BATCH_ELITE_SOFT_BONUSES = (0.02, 0.015, 0.01, 0.005, 0.0)
BATCH_ELITE_IMPROVING_BONUSES = (0.04, 0.03, 0.02, 0.01, 0.0)
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
GENERALIZATION_GAP_TOLERANCE = 0.02
GENERALIZATION_PENALTY_SCALE = 2.0
GENERALIZATION_PENALTY_CAP = -0.20
FEEDBACK_SUMMARY_LIMIT = 2
FEEDBACK_GRAPH_EXPR_MAX_CHARS = 160
FEEDBACK_SUMMARY_MAX_CHARS = 240
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
STAGE1_GATE_DISCOVERY_MIN = 8
STAGE1_GATE_UNIQUE_DISCOVERY_FAMILIES_MIN = 6
STAGE1_FORCE_PROMOTION_DISCOVERY_MIN = 8
STAGE1_FORCE_PROMOTION_UNIQUE_DISCOVERY_FAMILIES_MIN = 6
STAGE2_GATE_MIN_REWARD_TARGET = 0.90
STAGE2_GATE_MIN_TARGET_COUNT = 16
STAGE2_GATE_MIN_UNIQUE_TARGET_FAMILIES = 6
STAGE2_GATE_MAX_DOMINANT_DESCRIPTOR_SHARE = 0.50
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
STATIC_STAGE_REWARD_TARGET_METRIC = "stage1_static_score"
FORMAL_STAGE_REWARD_TARGET_METRIC = FORMAL_MULTI_HORIZON_REWARD_TARGET_METRIC
REWARD_VARIANT_ENV = "NNGPT_RL_REWARD_VARIANT"
REWARD_VARIANT_FULL = "full"
REWARD_VARIANT_NO_STRUCTURAL_NOVELTY = "no_structural_novelty"
REWARD_VARIANT_STRONG_REPEAT_PENALTY = "strong_repeat_penalty"
REWARD_VARIANTS = {
    REWARD_VARIANT_FULL,
    REWARD_VARIANT_NO_STRUCTURAL_NOVELTY,
    REWARD_VARIANT_STRONG_REPEAT_PENALTY,
}
STRUCTURE_MACRO_BONUS = 0.04
STRUCTURE_MULTI_STAGE_BONUS = 0.03
STRUCTURE_MOTIF_BONUS = 0.02
STRUCTURE_BATCH_DIVERSITY_BONUS = 0.03
STRUCTURE_NON_DOMINANT_FAMILY_BONUS = 0.02
STRUCTURE_ARCHIVE_RARITY_STRONG_BONUS = 0.03
STRUCTURE_ARCHIVE_RARITY_MEDIUM_BONUS = 0.02
STRUCTURE_ARCHIVE_RARITY_LIGHT_BONUS = 0.01
STAGE23_STRUCTURE_ARCHIVE_RARITY_CAP = 0.03
TRAINSET_NOVEL_FAMILY_BONUS = 0.04
TRAINSET_NOVEL_GRAPH_BONUS = 0.02
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

_TASK_STATE_NAMES = {
    "graph_archive_counts",
    "family_archive_counts",
    "family_hash_archive_counts",
    "descriptor_archive_counts",
    "backbone_signature_archive_counts",
    "cnn_signature_archive_counts",
    "block_signature_archive_counts",
    "backbone_cnn_pair_archive_counts",
    "backbone_block_pair_archive_counts",
    "family_metric_best",
    "motif_name_counts",
    "saved_graph_counts",
    "saved_family_hash_counts",
    "saved_backbone_signature_counts",
    "saved_cnn_signature_counts",
    "saved_backbone_cnn_pair_counts",
    "saved_backbone_block_pair_counts",
    "goal_graph_archive_counts",
    "goal_family_hash_archive_counts",
    "saved_goal_family_hash_counts",
    "train_graph_hashes",
    "train_family_hashes",
    "train_descriptor_keys",
    "current_group_reward_target_sum_by_backbone",
    "current_group_reward_target_count_by_backbone",
    "prev_closed_group_mean_reward_target_by_backbone",
    "best_closed_group_mean_reward_target_by_backbone",
    "saved_best_reward_target_by_backbone_cnn",
    "best_quality_acc_by_backbone_block",
    "dominant_family_hash",
    "dominant_family_share",
    "dominant_descriptor_key",
    "dominant_descriptor_share",
    "dominant_backbone_signature",
    "dominant_backbone_share",
    "dominant_cnn_signature",
    "dominant_cnn_share",
    "dominant_backbone_cnn_pair",
    "dominant_backbone_cnn_share",
    "discovery_family_hashes_seen",
}


_runtime_services: Optional[Any] = None


def configure_runtime_services(services: Any) -> None:
    global _runtime_services
    _runtime_services = services


def _services() -> Any:
    if _runtime_services is None:
        raise RuntimeError("Backbone reward runtime services have not been configured")
    return _runtime_services


class _TuneRLCodeLoggerProxy:
    def __getattr__(self, name: str):
        return getattr(_services().code_logger, name)


code_logger = _TuneRLCodeLoggerProxy()


def _current_stage_name() -> str:
    return str(getattr(_services(), "current_stage_name", STAGE1_STRUCTURE_EXPLORE))


def _prev_closed_group_mean_reward_target_acc() -> Optional[float]:
    return getattr(_services(), "prev_closed_group_mean_reward_target_acc", None)


def _best_closed_group_mean_reward_target_acc() -> Optional[float]:
    return getattr(_services(), "best_closed_group_mean_reward_target_acc", None)


def _best_closed_group_mean_train_acc() -> Optional[float]:
    return getattr(_services(), "best_closed_group_mean_train_acc", None)


def _best_closed_group_mean_test_acc() -> Optional[float]:
    return getattr(_services(), "best_closed_group_mean_test_acc", None)


def _best_reward_target_by_goal() -> Dict[str, float]:
    return getattr(_services(), "best_reward_target_by_goal", {})


def _archive_index() -> int:
    return int(getattr(_services(), "archive_index", 0) or 0)


def _set_archive_index(value: int) -> None:
    _services().set_archive_index(int(value))


def _current_generation_total() -> int:
    return _services().current_generation_total()


def _record_generation_event(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _services().record_generation_event(payload)


def close_reward_group_if_needed() -> Optional[Dict[str, Any]]:
    return _services().close_reward_group_if_needed()


def get_goal_counter(store: Dict[str, Counter], goal_key: str) -> Counter:
    if goal_key not in store:
        store[goal_key] = Counter()
    return store[goal_key]


def evaluate_reward_code(*args, **kwargs):
    return _services().evaluate_reward_code(*args, **kwargs)


def reward_eval_cfg_builder():
    return _services().reward_eval_cfg_builder()


def reward_run_epoch_dir(*args):
    return _services().reward_run_epoch_dir(*args)


def _record_current_group_trainable_sample(goal_key: str, res: Dict[str, Any], graph_info) -> None:
    _services().record_current_group_trainable_sample(goal_key, res, graph_info)


def _training_context_guidance(summary: Dict[str, Any]) -> str:
    return _services().training_context_guidance(summary)


def summarize_stage_training_context(stage_name: str, *, window_size: int = 50) -> Dict[str, Any]:
    return _services().summarize_stage_training_context(stage_name, window_size=window_size)


def training_context_guidance(summary: Dict[str, Any]) -> str:
    if not summary.get("has_recent_window"):
        return "no train-loss window yet"
    pressure = float(summary.get("exploration_pressure") or 0.0)
    if pressure >= 0.60:
        return "loss has plateaued or oscillated; favor structurally new candidates and avoid dominant templates"
    if bool(summary.get("monotonic_improving")) and pressure <= 0.25:
        return "loss is still improving; keep local mutations and avoid collapsing to one family"
    return "improvement is slowing; bias toward descriptor and family novelty over shallow repeats"


def _stage_items_unique(items: List[Dict[str, Any]], *, key: str) -> Set[str]:
    return {
        str(item.get(key))
        for item in items
        if item.get(key)
    }


def _stage_items_mean_key(items: List[Dict[str, Any]], key: str) -> Optional[float]:
    values = [
        float(item.get(key))
        for item in items
        if item.get(key) is not None
    ]
    if not values:
        return None
    return float(sum(values) / len(values))


def _stage_target_qualified_rows(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    qualified = []
    for item in items:
        if not bool(item.get("formal_success_candidate")):
            continue
        target_value = _optional_float(item.get("reward_target_value"))
        if target_value is None:
            target_value = _optional_float(item.get("formal_reward_target_value"))
        if target_value is not None and target_value >= STAGE2_GATE_MIN_REWARD_TARGET:
            qualified.append(item)
    return qualified


def stage1_gate_ready(rl) -> bool:
    recent_generations = rl._recent_stage_generation_window(rl.STAGE1_STRUCTURE_EXPLORE, rl.STAGE1_GATE_WINDOW_GENERATIONS)
    current_entry_group_count = len(rl._recent_stage_group_window(rl.STAGE1_STRUCTURE_EXPLORE, rl.MAX_STAGE_GROUP_HISTORY))
    if len(recent_generations) < rl.STAGE1_GATE_WINDOW_GENERATIONS:
        return False
    if current_entry_group_count < rl.STAGE1_PROMOTION_MIN_GROUPS:
        return False
    executable_count = sum(1 for item in recent_generations if bool(item.get("executable_candidate")))
    trainable_count = sum(
        1
        for item in recent_generations
        if bool(item.get("trained_step_ok") or item.get("backward_ok"))
    )
    discovery_rows = [item for item in recent_generations if bool(item.get("discovery_candidate"))]
    unique_discovery_families = len(_stage_items_unique(discovery_rows, key="family_hash"))
    return bool(
        executable_count >= rl.STAGE1_GATE_EXECUTABLE_MIN
        and trainable_count >= rl.STAGE1_GATE_TRAINABLE_MIN
        and len(discovery_rows) >= STAGE1_GATE_DISCOVERY_MIN
        and unique_discovery_families >= STAGE1_GATE_UNIQUE_DISCOVERY_FAMILIES_MIN
    )


def stage1_trainable_stable_ready(rl) -> Optional[Dict[str, Any]]:
    recent_generations = rl._recent_stage_generation_window(
        rl.STAGE1_STRUCTURE_EXPLORE,
        rl.STAGE1_EXECUTABLE_STABLE_WINDOW_GENERATIONS,
    )
    current_entry_group_count = len(rl._recent_stage_group_window(rl.STAGE1_STRUCTURE_EXPLORE, rl.MAX_STAGE_GROUP_HISTORY))
    if current_entry_group_count < rl.STAGE1_EXECUTABLE_STABLE_MIN_GROUPS:
        return None
    if len(recent_generations) < rl.STAGE1_EXECUTABLE_STABLE_WINDOW_GENERATIONS:
        return None
    recent_executable_count = sum(1 for item in recent_generations if bool(item.get("executable_candidate")))
    recent_executable_rate = recent_executable_count / float(len(recent_generations))
    recent_trainable_count = sum(
        1
        for item in recent_generations
        if bool(item.get("trained_step_ok") or item.get("backward_ok"))
    )
    recent_trainable_rate = recent_trainable_count / float(len(recent_generations))
    if recent_executable_rate < rl.STAGE1_EXECUTABLE_STABLE_MIN_RATE:
        return None
    if recent_trainable_rate < rl.STAGE1_TRAINABLE_STABLE_MIN_RATE:
        return None
    return {
        "stage_group_count": current_entry_group_count,
        "recent_generation_count": len(recent_generations),
        "recent_executable_count": recent_executable_count,
        "recent_executable_rate": recent_executable_rate,
        "recent_trainable_count": recent_trainable_count,
        "recent_trainable_rate": recent_trainable_rate,
    }


def stage1_force_promotion_ready(rl) -> Optional[Dict[str, int]]:
    recent_generations = rl._recent_stage_generation_window(rl.STAGE1_STRUCTURE_EXPLORE, rl.STAGE1_GATE_WINDOW_GENERATIONS)
    current_entry_group_count = len(rl._recent_stage_group_window(rl.STAGE1_STRUCTURE_EXPLORE, rl.MAX_STAGE_GROUP_HISTORY))
    if len(recent_generations) < rl.STAGE1_GATE_WINDOW_GENERATIONS:
        return None
    if current_entry_group_count < rl.STAGE1_PROMOTION_MIN_GROUPS:
        return None
    recent_executable_count = sum(1 for item in recent_generations if bool(item.get("executable_candidate")))
    recent_trainable_count = sum(
        1
        for item in recent_generations
        if bool(item.get("trained_step_ok") or item.get("backward_ok"))
    )
    discovery_rows = [item for item in recent_generations if bool(item.get("discovery_candidate"))]
    recent_discovery_count = len(discovery_rows)
    recent_unique_discovery_families = len(_stage_items_unique(discovery_rows, key="family_hash"))
    if recent_executable_count < rl.STAGE1_FORCE_PROMOTION_EXECUTABLE_MIN:
        return None
    if recent_trainable_count < rl.STAGE1_FORCE_PROMOTION_TRAINABLE_MIN:
        return None
    if recent_discovery_count < STAGE1_FORCE_PROMOTION_DISCOVERY_MIN:
        return None
    if recent_unique_discovery_families < STAGE1_FORCE_PROMOTION_UNIQUE_DISCOVERY_FAMILIES_MIN:
        return None
    reason = (
        "stage1_forced_promotion_after_plateau: "
        f"stage_groups={current_entry_group_count}, "
        f"recent_generations={len(recent_generations)}, "
        f"recent_executable_count={recent_executable_count}, "
        f"recent_trainable_count={recent_trainable_count}, "
        f"recent_discovery_count={recent_discovery_count}, "
        f"recent_unique_discovery_families={recent_unique_discovery_families}"
    )
    return {
        "stage_group_count": current_entry_group_count,
        "recent_generation_count": len(recent_generations),
        "recent_executable_count": recent_executable_count,
        "recent_trainable_count": recent_trainable_count,
        "recent_discovery_count": recent_discovery_count,
        "recent_unique_discovery_families": recent_unique_discovery_families,
        "reason": reason,
    }


def stage2_gate_ready(rl) -> bool:
    recent_generations = rl._recent_stage_generation_window(rl.STAGE2_FORMAL_EXPLORE, rl.STAGE2_GATE_WINDOW_GENERATIONS)
    recent_groups = rl._recent_stage_group_window(rl.STAGE2_FORMAL_EXPLORE, 5)
    recent_improvement_groups = rl._recent_stage_group_window(rl.STAGE2_FORMAL_EXPLORE, 4)
    current_entry_group_count = len(rl._recent_stage_group_window(rl.STAGE2_FORMAL_EXPLORE, rl.MAX_STAGE_GROUP_HISTORY))
    if len(recent_generations) < rl.STAGE2_GATE_WINDOW_GENERATIONS:
        return False
    if current_entry_group_count < rl.STAGE_REFERENCE_MIN_GROUPS[rl.STAGE2_FORMAL_EXPLORE]:
        return False
    qualified_rows = _stage_target_qualified_rows(recent_generations)
    unique_target_families = len(_stage_items_unique(qualified_rows, key="family_hash"))
    mean_dominant_share = _stage_items_mean_key(recent_groups, "dominant_family_share")
    mean_dominant_descriptor_share = _stage_items_mean_key(recent_groups, "dominant_descriptor_share")
    improving_groups = rl._count_group_improvements(recent_improvement_groups)
    return bool(
        len(qualified_rows) >= STAGE2_GATE_MIN_TARGET_COUNT
        and unique_target_families >= STAGE2_GATE_MIN_UNIQUE_TARGET_FAMILIES
        and improving_groups >= rl.STAGE2_GATE_IMPROVING_GROUPS_REQUIRED
        and mean_dominant_share is not None
        and mean_dominant_share <= 0.45
        and mean_dominant_descriptor_share is not None
        and mean_dominant_descriptor_share <= STAGE2_GATE_MAX_DOMINANT_DESCRIPTOR_SHARE
    )


def stage_gate_snapshot(rl) -> Dict[str, Any]:
    stage_name = str(rl.current_stage_name)
    recent_generations = rl._recent_stage_generation_window(
        stage_name,
        rl.STAGE1_GATE_WINDOW_GENERATIONS if stage_name == rl.STAGE1_STRUCTURE_EXPLORE else rl.STAGE2_GATE_WINDOW_GENERATIONS,
    )
    recent_groups = rl._recent_stage_group_window(stage_name, 5)
    discovery_rows = [item for item in recent_generations if bool(item.get("discovery_candidate"))]
    formal_rows = [item for item in recent_generations if bool(item.get("formal_success_candidate"))]
    qualified_target_rows = _stage_target_qualified_rows(recent_generations)
    return {
        "stage_name": stage_name,
        "stage_index": rl.RL_STAGE_TO_INDEX.get(stage_name, 0),
        "recent_generation_count": len(recent_generations),
        "recent_executable_count": sum(1 for item in recent_generations if bool(item.get("executable_candidate"))),
        "recent_trainable_count": sum(
            1
            for item in recent_generations
            if bool(item.get("trained_step_ok") or item.get("backward_ok"))
        ),
        "recent_discovery_count": len(discovery_rows),
        "recent_unique_discovery_families": len(_stage_items_unique(discovery_rows, key="family_hash")),
        "recent_formal_success_count": len(formal_rows),
        "recent_unique_formal_families": len(_stage_items_unique(formal_rows, key="family_hash")),
        "recent_target_qualified_count": len(qualified_target_rows),
        "recent_unique_target_families": len(_stage_items_unique(qualified_target_rows, key="family_hash")),
        "recent_unique_backbone_signatures": len(_stage_items_unique(recent_generations, key="backbone_signature")),
        "recent_unique_cnn_signatures": len(_stage_items_unique(recent_generations, key="cnn_signature")),
        "recent_unique_backbone_cnn_pairs": len(_stage_items_unique(recent_generations, key="backbone_cnn_pair_key")),
        "recent_mean_dominant_family_share": _stage_items_mean_key(recent_groups, "dominant_family_share"),
        "recent_mean_dominant_descriptor_share": _stage_items_mean_key(recent_groups, "dominant_descriptor_share"),
        "recent_mean_dominant_backbone_share": _stage_items_mean_key(recent_groups, "dominant_backbone_share"),
        "recent_mean_dominant_cnn_share": _stage_items_mean_key(recent_groups, "dominant_cnn_share"),
        "recent_mean_dominant_backbone_cnn_share": _stage_items_mean_key(recent_groups, "dominant_backbone_cnn_share"),
        "recent_improving_groups": rl._count_group_improvements(rl._recent_stage_group_window(stage_name, 4)),
        "recovery_active": bool(rl.recovery_active),
    }


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


def log_reward_failure_trace(entry: Dict[str, Any], res: Dict[str, Any]) -> None:
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
        _services().code_logger.log_to_file(trace_message)


def update_current_group_metrics(results: List[Dict[str, Any]]) -> None:
    _services().update_current_group_metrics(results)


def extract_seed_context(kwargs: Dict[str, Any], expected_count: int):
    return _services().extract_seed_context(kwargs, expected_count)


def clean_block(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"^```python\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_completion_blocks(completion: str) -> Tuple[str, str, str]:
    block_code = clean_block(extract_str(completion, "<block>", "</block>"))
    init_code = clean_block(extract_str(completion, "<init>", "</init>"))
    forward_code = clean_block(extract_str(completion, "<forward>", "</forward>"))
    return block_code, init_code, forward_code


def extract_reward_completion_blocks(completion: str) -> Tuple[str, str, str]:
    return _services().extract_completion_blocks(completion)


def has_structural_motif(graph_info) -> bool:
    return bool(graph_info and (graph_info.project_calls or graph_info.stem_calls or graph_info.fractal_calls))


def is_multi_stage_architecture(graph_info) -> bool:
    return bool(graph_info and (graph_info.depth >= 5 or graph_info.merges >= 2 or graph_info.fractal_calls >= 2))


def passes_macro_structure_gate(graph_info) -> bool:
    if not graph_info or not graph_info.parse_ok or graph_info.is_plain_parallel_triple:
        return False
    if graph_info.project_calls or graph_info.stem_calls:
        return True
    return is_multi_stage_architecture(graph_info)


def is_shallow_one_shot_fuse(graph_info) -> bool:
    return bool(
        graph_info
        and graph_info.parse_ok
        and not graph_info.is_plain_parallel_triple
        and graph_info.fuse_calls >= 1
        and graph_info.merges <= 1
        and graph_info.depth <= 4
        and graph_info.project_calls == 0
        and graph_info.stem_calls == 0
        and graph_info.fractal_calls <= 1
        and graph_info.backbone_calls >= 1
    )


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


def _current_stage_index() -> int:
    return RL_STAGE_TO_INDEX.get(_current_stage_name(), 0)


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


def build_group_feedback_summary(
    *,
    goal_key: str,
    res: Dict[str, Any],
    graph_info,
    reward_group_id: int,
) -> Dict[str, Any]:
    graph_expr_short = _truncate_text(str(res.get("graph_expr") or ""), FEEDBACK_GRAPH_EXPR_MAX_CHARS)
    pattern_name = str(res.get("pattern_name") or res.get("suggested_pattern_name") or "unknown")
    reward_target_value = float(_result_reward_target_value(res) or 0.0)
    frozen_train_acc = float(_optional_float(res.get("frozen_train_acc", res.get("train_acc"))) or 0.0)
    frozen_test_acc = float(_optional_float(res.get("frozen_test_acc", res.get("test_acc", res.get("val_metric")))) or 0.0)
    unfrozen_train_acc = _optional_float(res.get("unfrozen_train_acc"))
    unfrozen_test_acc = _optional_float(res.get("unfrozen_test_acc"))
    backbone_names = list(res.get("backbone_model_names") or [])
    backbone_signature = str(res.get("backbone_signature") or build_backbone_signature(backbone_names))
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
    return {
        "goal_key": goal_key,
        "pattern_name": pattern_name,
        "graph_expr_short": graph_expr_short,
        "reward_target_value": reward_target_value,
        "frozen_train_acc": frozen_train_acc,
        "frozen_test_acc": frozen_test_acc,
        "unfrozen_train_acc": unfrozen_train_acc,
        "unfrozen_test_acc": unfrozen_test_acc,
        "backbone_model_names": backbone_names,
        "stats_short": stats_short,
        "summary": summary,
        "family_hash": str(getattr(graph_info, "family_hash", "") or res.get("family_hash") or ""),
        "signature": str(res.get("signature") or ""),
        "reward_group_id": reward_group_id,
        "backbone_signature": backbone_signature,
        "cnn_signature": cnn_signature,
        "cnn_expr_short": cnn_expr_short,
    }


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


def _has_completed_formal_epoch(res: Dict[str, Any]) -> bool:
    try:
        return int(res.get("epochs_completed", 0) or 0) >= 1
    except (TypeError, ValueError):
        return False


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


def _is_trainable_candidate(res: Dict[str, Any], graph_info) -> bool:
    return _stage1_trainability_ok(res, graph_info)


def _is_executable_candidate(res: Dict[str, Any], graph_info) -> bool:
    return bool(
        graph_info
        and graph_info.parse_ok
        and res.get("built_ok")
        and res.get("forward_shape_ok")
    )


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
        "reward_target_metric": _stage_reward_target_metric(_current_stage_name()),
        "reward_target_value": None,
        "best_closed_group_mean_reward_target_acc": _best_closed_group_mean_reward_target_acc(),
        "best_closed_group_mean_train_acc": _best_closed_group_mean_train_acc(),
        "best_closed_group_mean_test_acc": _best_closed_group_mean_test_acc(),
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
            "reward_target_metric": _stage_reward_target_metric(_current_stage_name()),
            "reward_target_value": None,
            "goal_tag_hit_count": 0,
            "goal_tag_total_count": 0,
            "goal_tag_hit_rate": 0.0,
        },
        "error": error,
        "current_stage_name": _current_stage_name(),
        "current_stage_index": _current_stage_index(),
        "stage_uses_formal_eval": _stage_uses_formal_eval(_current_stage_name()),
        "stage_uses_static_only": _stage_uses_static_only(_current_stage_name()),
    }


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


def _reward_variant_is_strong_repeat_penalty() -> bool:
    return resolve_reward_variant() == REWARD_VARIANT_STRONG_REPEAT_PENALTY


def _stage_reward_target_metric(stage_name: str) -> str:
    if str(stage_name) == STAGE1_STRUCTURE_EXPLORE:
        return STATIC_STAGE_REWARD_TARGET_METRIC
    return FORMAL_STAGE_REWARD_TARGET_METRIC


def _stage_uses_formal_eval(stage_name: str) -> bool:
    return str(stage_name) in {STAGE2_FORMAL_EXPLORE, STAGE3_FORMAL_OPTIMIZE}


def _stage_uses_static_only(stage_name: str) -> bool:
    return str(stage_name) == STAGE1_STRUCTURE_EXPLORE


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
    task_context = _services().group_context_fields()
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


def render_completion_xml(block_code: str, init_code: str, forward_code: str) -> str:
    return "\n".join(
        [
            "<block>",
            textwrap.dedent(block_code).strip(),
            "</block>",
            "<init>",
            textwrap.dedent(init_code).strip(),
            "</init>",
            "<forward>",
            textwrap.dedent(forward_code).strip(),
            "</forward>",
        ]
    )


def reconstruct_code(
    completion: str,
    *,
    pattern_name_override: str = "",
) -> str:
    block_code, init_code, forward_code = extract_reward_completion_blocks(completion)
    if not block_code or not init_code or not forward_code:
        return ""

    if pattern_name_override:
        init_code = ensure_pattern_name(init_code, pattern_name_override)

    code = SFTUtil.skeleton_code
    sig_block = "def drop_conv3x3_block(in_channels, out_channels, stride=1, padding=1, bias=False, dropout_prob=0.0):"
    code = code.replace(sig_block, textwrap.dedent(block_code))

    sig_init = "    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:"
    code = code.replace(sig_init, textwrap.indent(textwrap.dedent(init_code), "    "))

    sig_forward = "    def forward(self, x: torch.Tensor, is_probing: bool = False) -> torch.Tensor:"
    code = code.replace(sig_forward, textwrap.indent(textwrap.dedent(forward_code), "    "))
    return code


def _iter_text_candidates(value: Any) -> List[str]:
    if value is None:
        return []
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
    train_graph_hashes.clear()
    train_family_hashes.clear()
    train_descriptor_keys.clear()

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
            if graph_info and graph_info.parse_ok:
                train_graph_hashes.add(graph_info.graph_hash)
                train_family_hashes.add(graph_info.family_hash)
                train_descriptor_keys.add(graph_info.descriptor_key)
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
        f"graph_hashes={len(train_graph_hashes)}, family_hashes={len(train_family_hashes)}, "
        f"descriptor_keys={len(train_descriptor_keys)}"
    )


def load_rl_dataset(tokenizer):
    data = api.data(task="img-classification", nn_prefixes=("rl-bb-test1",))
    if data.empty:
        raise RuntimeError("No 'rl-bb-test1' data found for RL; sync the dataset prefix before training.")

    print(f"Loaded {len(data)} examples for RL")
    bootstrap_trainset_reference_library(data)

    prompts = []
    legacy_patterns = ", ".join(SFTUtil.legacy_patterns)
    goal_profiles = SFTUtil.open_discovery_goal_profiles

    for _, row in data.iterrows():
        accuracy = _coerce_accuracy_baseline(row.get("accuracy"), context="seed row accuracy")
        for profile in goal_profiles:
            target_pattern = SFTUtil.goal_profile_target_pattern(profile)
            module_hints = (
                "self.backbone_a",
                "self.backbone_b",
                *profile["module_hints"],
            )
            user_prompt = PROMPT_TEMPLATE.format(
                accuracy=accuracy,
                skeleton_code=SFTUtil.open_discovery_skeleton_code,
                available_backbones=", ".join(SFTUtil.available_backbones),
                legacy_patterns=legacy_patterns,
                goal_name=profile["name"],
                target_tags=", ".join(profile["tags"]),
                target_pattern=target_pattern,
                design_brief=profile["brief"],
                tag_realization=profile.get("realization", profile["brief"]),
                goal_tag_parser_cues=SFTUtil.goal_tag_parser_cues(profile["tags"]),
                module_hints=", ".join(module_hints),
                block_signature=PROMPT_BLOCK_SIGNATURE,
                init_signature=PROMPT_INIT_SIGNATURE,
                forward_signature=PROMPT_FORWARD_SIGNATURE,
            )

            messages = [{"role": "user", "content": user_prompt}]
            prompt_str = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )

            prompts.append({
                "prompt": prompt_str,
                "accuracy": accuracy,
                "goal_name": profile["name"],
                "target_tags": ", ".join(profile["tags"]),
            })

    rl_dataset = Dataset.from_list(prompts)
    return rl_dataset.shuffle(seed=42)


def _counter_payload(counter: Counter) -> Dict[str, int]:
    return StageState.counter_payload(counter)


def _nested_counter_payload(mapping: Dict[str, Counter]) -> Dict[str, Dict[str, int]]:
    return StageState.nested_counter_payload(mapping)


def capture_runtime_state() -> Dict[str, Any]:
    return {
        "graph_archive_counts": _counter_payload(graph_archive_counts),
        "family_archive_counts": _counter_payload(family_archive_counts),
        "family_hash_archive_counts": _counter_payload(family_hash_archive_counts),
        "descriptor_archive_counts": _counter_payload(descriptor_archive_counts),
        "backbone_signature_archive_counts": _counter_payload(backbone_signature_archive_counts),
        "cnn_signature_archive_counts": _counter_payload(cnn_signature_archive_counts),
        "block_signature_archive_counts": _counter_payload(block_signature_archive_counts),
        "backbone_cnn_pair_archive_counts": _counter_payload(backbone_cnn_pair_archive_counts),
        "backbone_block_pair_archive_counts": _counter_payload(backbone_block_pair_archive_counts),
        "family_metric_best": {str(key): float(value) for key, value in family_metric_best.items()},
        "motif_name_counts": _counter_payload(motif_name_counts),
        "saved_graph_counts": _counter_payload(saved_graph_counts),
        "saved_family_hash_counts": _counter_payload(saved_family_hash_counts),
        "saved_backbone_signature_counts": _counter_payload(saved_backbone_signature_counts),
        "saved_cnn_signature_counts": _counter_payload(saved_cnn_signature_counts),
        "saved_backbone_cnn_pair_counts": _counter_payload(saved_backbone_cnn_pair_counts),
        "saved_backbone_block_pair_counts": _counter_payload(saved_backbone_block_pair_counts),
        "goal_graph_archive_counts": _nested_counter_payload(goal_graph_archive_counts),
        "goal_family_hash_archive_counts": _nested_counter_payload(goal_family_hash_archive_counts),
        "saved_goal_family_hash_counts": _nested_counter_payload(saved_goal_family_hash_counts),
        "current_group_reward_target_sum_by_backbone": StageState.float_dict_payload(current_group_reward_target_sum_by_backbone),
        "current_group_reward_target_count_by_backbone": _counter_payload(current_group_reward_target_count_by_backbone),
        "prev_closed_group_mean_reward_target_by_backbone": StageState.float_dict_payload(prev_closed_group_mean_reward_target_by_backbone),
        "best_closed_group_mean_reward_target_by_backbone": StageState.float_dict_payload(best_closed_group_mean_reward_target_by_backbone),
        "saved_best_reward_target_by_backbone_cnn": StageState.float_dict_payload(saved_best_reward_target_by_backbone_cnn),
        "best_quality_acc_by_backbone_block": StageState.float_dict_payload(best_quality_acc_by_backbone_block),
        "dominant_family_hash": dominant_family_hash,
        "dominant_family_share": dominant_family_share,
        "dominant_descriptor_key": dominant_descriptor_key,
        "dominant_descriptor_share": dominant_descriptor_share,
        "dominant_backbone_signature": dominant_backbone_signature,
        "dominant_backbone_share": dominant_backbone_share,
        "dominant_cnn_signature": dominant_cnn_signature,
        "dominant_cnn_share": dominant_cnn_share,
        "dominant_backbone_cnn_pair": dominant_backbone_cnn_pair,
        "dominant_backbone_cnn_share": dominant_backbone_cnn_share,
        "discovery_family_hashes_seen": sorted(str(item) for item in discovery_family_hashes_seen),
    }


def restore_runtime_state(state: Optional[Dict[str, Any]]) -> None:
    if not state:
        return
    global dominant_family_hash, dominant_family_share
    global dominant_descriptor_key, dominant_descriptor_share
    global dominant_backbone_signature, dominant_backbone_share
    global dominant_cnn_signature, dominant_cnn_share
    global dominant_backbone_cnn_pair, dominant_backbone_cnn_share

    StageState.restore_counter(graph_archive_counts, state.get("graph_archive_counts"))
    StageState.restore_counter(family_archive_counts, state.get("family_archive_counts"))
    StageState.restore_counter(family_hash_archive_counts, state.get("family_hash_archive_counts"))
    StageState.restore_counter(descriptor_archive_counts, state.get("descriptor_archive_counts"))
    StageState.restore_counter(backbone_signature_archive_counts, state.get("backbone_signature_archive_counts"))
    StageState.restore_counter(cnn_signature_archive_counts, state.get("cnn_signature_archive_counts"))
    StageState.restore_counter(block_signature_archive_counts, state.get("block_signature_archive_counts"))
    StageState.restore_counter(backbone_cnn_pair_archive_counts, state.get("backbone_cnn_pair_archive_counts"))
    StageState.restore_counter(backbone_block_pair_archive_counts, state.get("backbone_block_pair_archive_counts"))
    family_metric_best.clear()
    family_metric_best.update({str(key): float(value) for key, value in (state.get("family_metric_best") or {}).items()})
    StageState.restore_counter(motif_name_counts, state.get("motif_name_counts"))
    StageState.restore_counter(saved_graph_counts, state.get("saved_graph_counts"))
    StageState.restore_counter(saved_family_hash_counts, state.get("saved_family_hash_counts"))
    StageState.restore_counter(saved_backbone_signature_counts, state.get("saved_backbone_signature_counts"))
    StageState.restore_counter(saved_cnn_signature_counts, state.get("saved_cnn_signature_counts"))
    StageState.restore_counter(saved_backbone_cnn_pair_counts, state.get("saved_backbone_cnn_pair_counts"))
    StageState.restore_counter(saved_backbone_block_pair_counts, state.get("saved_backbone_block_pair_counts"))
    StageState.restore_nested_counters(goal_graph_archive_counts, state.get("goal_graph_archive_counts"))
    StageState.restore_nested_counters(goal_family_hash_archive_counts, state.get("goal_family_hash_archive_counts"))
    StageState.restore_nested_counters(saved_goal_family_hash_counts, state.get("saved_goal_family_hash_counts"))
    StageState.restore_float_dict(current_group_reward_target_sum_by_backbone, state.get("current_group_reward_target_sum_by_backbone"))
    StageState.restore_counter(current_group_reward_target_count_by_backbone, state.get("current_group_reward_target_count_by_backbone"))
    StageState.restore_float_dict(prev_closed_group_mean_reward_target_by_backbone, state.get("prev_closed_group_mean_reward_target_by_backbone"))
    StageState.restore_float_dict(best_closed_group_mean_reward_target_by_backbone, state.get("best_closed_group_mean_reward_target_by_backbone"))
    StageState.restore_float_dict(saved_best_reward_target_by_backbone_cnn, state.get("saved_best_reward_target_by_backbone_cnn"))
    StageState.restore_float_dict(best_quality_acc_by_backbone_block, state.get("best_quality_acc_by_backbone_block"))

    dominant_family_hash = state.get("dominant_family_hash")
    dominant_family_share = float(state.get("dominant_family_share", 0.0) or 0.0)
    dominant_descriptor_key = state.get("dominant_descriptor_key")
    dominant_descriptor_share = float(state.get("dominant_descriptor_share", 0.0) or 0.0)
    dominant_backbone_signature = state.get("dominant_backbone_signature")
    dominant_backbone_share = float(state.get("dominant_backbone_share", 0.0) or 0.0)
    dominant_cnn_signature = state.get("dominant_cnn_signature")
    dominant_cnn_share = float(state.get("dominant_cnn_share", 0.0) or 0.0)
    dominant_backbone_cnn_pair = state.get("dominant_backbone_cnn_pair")
    dominant_backbone_cnn_share = float(state.get("dominant_backbone_cnn_share", 0.0) or 0.0)
    discovery_family_hashes_seen.clear()
    discovery_family_hashes_seen.update(str(item) for item in (state.get("discovery_family_hashes_seen") or []))


def reset_runtime_state() -> None:
    global dominant_family_hash, dominant_family_share
    global dominant_descriptor_key, dominant_descriptor_share
    global dominant_backbone_signature, dominant_backbone_share
    global dominant_cnn_signature, dominant_cnn_share
    global dominant_backbone_cnn_pair, dominant_backbone_cnn_share

    for item in (
        graph_archive_counts,
        family_archive_counts,
        family_hash_archive_counts,
        descriptor_archive_counts,
        backbone_signature_archive_counts,
        cnn_signature_archive_counts,
        block_signature_archive_counts,
        backbone_cnn_pair_archive_counts,
        backbone_block_pair_archive_counts,
        family_metric_best,
        motif_name_counts,
        saved_graph_counts,
        saved_family_hash_counts,
        saved_backbone_signature_counts,
        saved_cnn_signature_counts,
        saved_backbone_cnn_pair_counts,
        saved_backbone_block_pair_counts,
        goal_graph_archive_counts,
        goal_family_hash_archive_counts,
        saved_goal_family_hash_counts,
        current_group_reward_target_sum_by_backbone,
        current_group_reward_target_count_by_backbone,
        prev_closed_group_mean_reward_target_by_backbone,
        best_closed_group_mean_reward_target_by_backbone,
        saved_best_reward_target_by_backbone_cnn,
        best_quality_acc_by_backbone_block,
        discovery_family_hashes_seen,
    ):
        item.clear()

    dominant_family_hash = None
    dominant_family_share = 0.0
    dominant_descriptor_key = None
    dominant_descriptor_share = 0.0
    dominant_backbone_signature = None
    dominant_backbone_share = 0.0
    dominant_cnn_signature = None
    dominant_cnn_share = 0.0
    dominant_backbone_cnn_pair = None
    dominant_backbone_cnn_share = 0.0


def group_context_fields() -> Dict[str, Any]:
    return {
        "dominant_family_hash": dominant_family_hash,
        "dominant_family_share": dominant_family_share,
        "dominant_descriptor_key": dominant_descriptor_key,
        "dominant_descriptor_share": dominant_descriptor_share,
        "dominant_backbone_signature": dominant_backbone_signature,
        "dominant_backbone_share": dominant_backbone_share,
        "dominant_cnn_signature": dominant_cnn_signature,
        "dominant_cnn_share": dominant_cnn_share,
        "dominant_backbone_cnn_pair": dominant_backbone_cnn_pair,
        "dominant_backbone_cnn_share": dominant_backbone_cnn_share,
    }


def update_group_metrics(results: List[Dict[str, Any]]) -> None:
    for res in results:
        reward_target_value = _result_reward_target_value(res)
        backbone_signature = _result_backbone_signature(res)
        if reward_target_value is not None and backbone_signature:
            current_group_reward_target_sum_by_backbone[backbone_signature] = (
                float(current_group_reward_target_sum_by_backbone.get(backbone_signature, 0.0))
                + float(reward_target_value)
            )
            current_group_reward_target_count_by_backbone[backbone_signature] += 1


def close_group_payload() -> Dict[str, Any]:
    global dominant_family_hash, dominant_family_share
    global dominant_descriptor_key, dominant_descriptor_share
    global dominant_backbone_signature, dominant_backbone_share
    global dominant_cnn_signature, dominant_cnn_share
    global dominant_backbone_cnn_pair, dominant_backbone_cnn_share

    closed_mean_reward_target_by_backbone = {
        str(backbone_signature): float(current_group_reward_target_sum_by_backbone.get(backbone_signature, 0.0)) / float(count)
        for backbone_signature, count in current_group_reward_target_count_by_backbone.items()
        if int(count) > 0
    }
    prev_closed_group_mean_reward_target_by_backbone.clear()
    prev_closed_group_mean_reward_target_by_backbone.update(closed_mean_reward_target_by_backbone)
    for backbone_signature, backbone_mean in closed_mean_reward_target_by_backbone.items():
        best_closed_group_mean_reward_target_by_backbone[backbone_signature] = max(
            float(backbone_mean),
            float(best_closed_group_mean_reward_target_by_backbone.get(backbone_signature, float("-inf"))),
        )

    total_valid = sum(family_hash_archive_counts.values())
    if total_valid > 0:
        dominant_family_hash, dominant_count = family_hash_archive_counts.most_common(1)[0]
        dominant_family_share = dominant_count / total_valid
    else:
        dominant_family_hash = None
        dominant_family_share = 0.0
    descriptor_total = sum(descriptor_archive_counts.values())
    if descriptor_total > 0:
        dominant_descriptor_key, dominant_descriptor_count = descriptor_archive_counts.most_common(1)[0]
        dominant_descriptor_share = dominant_descriptor_count / descriptor_total
    else:
        dominant_descriptor_key = None
        dominant_descriptor_share = 0.0
    backbone_total = sum(backbone_signature_archive_counts.values())
    if backbone_total > 0:
        dominant_backbone_signature, dominant_backbone_count = backbone_signature_archive_counts.most_common(1)[0]
        dominant_backbone_share = dominant_backbone_count / backbone_total
    else:
        dominant_backbone_signature = None
        dominant_backbone_share = 0.0
    cnn_total = sum(cnn_signature_archive_counts.values())
    if cnn_total > 0:
        dominant_cnn_signature, dominant_cnn_count = cnn_signature_archive_counts.most_common(1)[0]
        dominant_cnn_share = dominant_cnn_count / cnn_total
    else:
        dominant_cnn_signature = None
        dominant_cnn_share = 0.0
    backbone_cnn_total = sum(backbone_cnn_pair_archive_counts.values())
    if backbone_cnn_total > 0:
        dominant_backbone_cnn_pair, dominant_backbone_cnn_count = backbone_cnn_pair_archive_counts.most_common(1)[0]
        dominant_backbone_cnn_share = dominant_backbone_cnn_count / backbone_cnn_total
    else:
        dominant_backbone_cnn_pair = None
        dominant_backbone_cnn_share = 0.0

    return {
        **group_context_fields(),
        "group_log_summary": (
            f"dominant_family={dominant_family_hash or 'n/a'} "
            f"({float(dominant_family_share or 0.0):.2%})"
        ),
        "closed_mean_reward_target_by_backbone": StageState.float_dict_payload(closed_mean_reward_target_by_backbone),
        "prev_closed_group_mean_reward_target_by_backbone": StageState.float_dict_payload(prev_closed_group_mean_reward_target_by_backbone),
        "best_closed_group_mean_reward_target_by_backbone": StageState.float_dict_payload(best_closed_group_mean_reward_target_by_backbone),
        "unique_descriptor_count": len(descriptor_archive_counts),
    }


def reset_current_group_state() -> None:
    current_group_reward_target_sum_by_backbone.clear()
    current_group_reward_target_count_by_backbone.clear()


def reset_stage_comparison_state() -> None:
    prev_closed_group_mean_reward_target_by_backbone.clear()
    best_closed_group_mean_reward_target_by_backbone.clear()


def archive_snapshot_family_counts() -> Dict[str, int]:
    return dict(family_hash_archive_counts)


def recovery_marker_count() -> int:
    return len(discovery_family_hashes_seen)


def render_prompt_feedback_text(*, feedback_char_budget: int = 1200) -> str:
    state = _services().get_prompt_feedback_state()
    stage_name = _current_stage_name()
    current_metric = _stage_reward_target_metric(stage_name)
    header_lines = [
        f"- Current Stage: {stage_name}",
        f"- Reward Target Metric: {current_metric}",
        f"- Previous Closed Group Mean Target Acc: {_format_optional_metric(state['prev_closed_group_mean_reward_target_acc'])}",
        f"- Current Best Closed Group Mean Target Acc: {_format_optional_metric(state['best_closed_group_mean_reward_target_acc'])}",
        f"- Previous Closed Group Mean Frozen Train Acc: {_format_optional_metric(state['prev_closed_group_mean_train_acc'])}",
        f"- Previous Closed Group Mean Frozen Test Acc: {_format_optional_metric(state['prev_closed_group_mean_test_acc'])}",
        (
            "- Current Dominant Family To Avoid When Not Improving: "
            f"{state['dominant_family_hash'] or 'n/a'} "
            f"(share={float(state['dominant_family_share'] or 0.0):.2%})"
        ),
        (
            "- Rule: same backbone pair is acceptable; compare new models mainly against that pair's own recent baseline"
        ),
        (
            "- Rule: within the same backbone pair, change stem/project/fuse CNN layout, not just widths, ordering, or formatting"
        ),
    ]
    if stage_name != STAGE1_STRUCTURE_EXPLORE:
        header_lines.extend(
            [
                f"- Meaningful Reward Target: >= {_format_target_metric(state['prev_closed_group_mean_reward_target_acc'], GROUP_IMPROVEMENT_DELTA)}",
                f"- Stretch Target To Refresh Best: >= {_format_target_metric(state['best_closed_group_mean_reward_target_acc'], BEST_GROUP_REFRESH_DELTA)}",
                "- Rule: prioritize higher frozen test accuracy, not just easier train accuracy",
                "- Rule: dominant-family reuse or plain classifier-only fuse below target is penalized",
            ]
        )
    if stage_name == STAGE2_FORMAL_EXPLORE:
        training_context = dict(state.get("training_context") or {})
        context_guidance = _training_context_guidance(training_context)
        header_lines.extend(
            [
                (
                    "- Current Training Context: "
                    f"last50 best_loss={_format_optional_metric(training_context.get('recent_best_loss'))}, "
                    f"delta_best={_format_optional_signed_metric(training_context.get('delta_best_loss'))}; "
                    f"last50 avg_loss={_format_optional_metric(training_context.get('recent_avg_loss'))}, "
                    f"delta_avg={_format_optional_signed_metric(training_context.get('delta_avg_loss'))}"
                ),
                (
                    "- Training Trend: "
                    f"slope={_format_optional_signed_metric(training_context.get('loss_slope_recent'))}/epoch, "
                    f"variance={_format_optional_metric(training_context.get('loss_variance_recent'))}, "
                    f"since_best={training_context.get('epochs_since_last_improvement', 'n/a')}, "
                    f"plateau={float(training_context.get('plateau_score') or 0.0):.2f}, "
                    f"oscillation={float(training_context.get('oscillation_score') or 0.0):.2f}"
                ),
                f"- Training Guidance: {context_guidance}",
            ]
        )

    prev_lines = [
        f"  - {item['summary']}"
        for item in state.get("prev_group_feedback", [])[:FEEDBACK_SUMMARY_LIMIT]
    ]
    best_lines = [
        f"  - {item['summary']}"
        for item in state.get("best_group_feedback", [])[:FEEDBACK_SUMMARY_LIMIT]
    ]

    def _compose_lines(current_prev_lines: List[str], current_best_lines: List[str]) -> str:
        lines = list(header_lines)
        if current_prev_lines:
            lines.append("- Previous Group Strong Examples:")
            lines.extend(current_prev_lines)
        else:
            lines.append("- Previous Group Strong Examples: none yet")

        if current_best_lines:
            lines.append("- Current Best Group Strong Examples:")
            lines.extend(current_best_lines)
        else:
            lines.append("- Current Best Group Strong Examples: none yet")
        return "\n".join(lines)

    text = _compose_lines(prev_lines, best_lines)
    if len(text) <= feedback_char_budget:
        return text

    if len(best_lines) >= 2:
        best_lines = best_lines[:1]
    text = _compose_lines(prev_lines, best_lines)
    if len(text) <= feedback_char_budget:
        return text

    if len(prev_lines) >= 2:
        prev_lines = prev_lines[:1]
        text = _compose_lines(prev_lines, best_lines)
    return _truncate_text(text, feedback_char_budget)


def _base_discovery_reward_fn(
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
    reward_variant = resolve_reward_variant()
    stage_name = _current_stage_name()
    stage_profile = _stage_reward_profile(stage_name)
    stage_reward_metric = _stage_reward_target_metric(stage_name)
    prm = {
        'lr': 0.01,
        'batch': 64,
        'dropout': 0.3,
        'momentum': 0.9,
        'transform': FORMAL_REWARD_TRANSFORM,
        'epoch': 1,
    }
    block_code, init_code, forward_code = extract_reward_completion_blocks(completion)
    block_contributes_to_forward = _block_contributes_to_forward(init_code, forward_code)
    block_signature = (
        _block_signature_from_code(block_code)
        if block_contributes_to_forward
        else "incomplete_block"
    )
    plain_dual_backbone_concat = _is_plain_dual_backbone_concat(forward_code)
    backbone_model_names = _extract_backbone_model_names(init_code)
    if not block_code or not init_code or not forward_code:
        return _discovery_failure_result(
            -2.0,
            "Reconstruction failed (tags missing?)",
            seed_accuracy_baseline=seed_accuracy_baseline,
            backbone_model_names=backbone_model_names,
        )

    if "self.pattern" in forward_code:
        return _discovery_failure_result(
            -5.0,
            "CHEAT DETECTED: Accessed self.pattern inside forward block",
            seed_accuracy_baseline=seed_accuracy_baseline,
            backbone_model_names=backbone_model_names,
        )

    graph_info = graph_info or extract_graph_info(
        init_code,
        forward_code,
        legacy_patterns=SFTUtil.legacy_patterns,
    )
    prompt_target_pattern = str(prompt_target_pattern or "").strip()
    pattern_detection = detect_target_structure(
        prompt_target_pattern=prompt_target_pattern,
        graph_info=graph_info,
        block_contributes_to_forward=block_contributes_to_forward,
        block_signature=block_signature,
    )
    effective_pattern_name = graph_info.suggested_pattern_name
    pattern_override = graph_info.suggested_pattern_name if not graph_info.has_custom_pattern_name else ""

    final_code = reconstruct_code(completion, pattern_name_override=pattern_override)
    if not final_code:
        return _discovery_failure_result(
            -2.0,
            "Code reconstruction failed",
            seed_accuracy_baseline=seed_accuracy_baseline,
            backbone_model_names=backbone_model_names,
        )

    if precomputed_eval_result is not None:
        res = dict(precomputed_eval_result)
    else:
        formal_input_shape = _formal_reward_input_shape()
        eval_device = "cuda" if torch.cuda.is_available() else "cpu"
        res = evaluate_reward_code(
            final_code,
            in_shape=formal_input_shape,
            out_shape=(10,),
            prm=prm,
            device=eval_device,
            seed_accuracy_baseline=seed_accuracy_baseline,
            cfg=_invoke_eval_cfg_builder(
                reward_eval_cfg_builder(),
                stage_name=stage_name,
                in_shape=formal_input_shape,
                out_shape=(10,),
                prm=prm,
                device=eval_device,
            ),
            reward_batch_index=reward_batch_index,
            completion_index=completion_index,
            batch_last_item=batch_last_item,
        )

    if not res.get("built_ok"):
        res["r_build_partial"] = _compute_build_partial_reward(res)
    res.setdefault("backbone_model_names", backbone_model_names)
    backbone_signature = build_backbone_signature(backbone_model_names)
    cnn_signature = str(getattr(graph_info, "cnn_signature", "") or "incomplete_cnn")
    cnn_expr = str(getattr(graph_info, "cnn_expr", "") or "IncompleteCNN")
    backbone_cnn_pair_key = _backbone_cnn_pair_key(backbone_signature, cnn_signature)
    backbone_block_pair_key = _backbone_block_pair_key(backbone_signature, block_signature)

    training_context = summarize_stage_training_context(stage_name)
    shallow_one_shot = is_shallow_one_shot_fuse(graph_info)
    minimal_init_template = _is_minimal_backbone_classifier_template(init_code)
    batch_same_family_count = batch_family_hashes.count(graph_info.family_hash) if batch_family_hashes and graph_info.parse_ok else 0
    batch_same_graph_count = batch_graph_hashes.count(graph_info.graph_hash) if batch_graph_hashes and graph_info.parse_ok else 0
    batch_same_descriptor_count = (
        batch_descriptor_keys.count(graph_info.descriptor_key)
        if batch_descriptor_keys and graph_info.parse_ok
        else 0
    )
    batch_same_backbone_count = (
        batch_backbone_signatures.count(backbone_signature)
        if batch_backbone_signatures and graph_info.parse_ok
        else 0
    )
    batch_same_backbone_cnn_count = (
        sum(
            1
            for batch_backbone_signature, batch_cnn_signature in zip(batch_backbone_signatures or [], batch_cnn_signatures or [])
            if batch_backbone_signature == backbone_signature and batch_cnn_signature == cnn_signature
        )
        if graph_info.parse_ok
        else 0
    )
    batch_same_block_count = (
        batch_block_signatures.count(block_signature)
        if batch_block_signatures and block_signature and block_signature != "incomplete_block"
        else 0
    )
    batch_same_backbone_block_count = (
        batch_backbone_block_signatures.count(backbone_block_pair_key)
        if batch_backbone_block_signatures
        and graph_info.parse_ok
        and block_signature
        and block_signature != "incomplete_block"
        else 0
    )
    archive_snapshot_family_freq = int((archive_snapshot_family_counts or {}).get(graph_info.family_hash, 0)) if graph_info.parse_ok else 0
    archive_snapshot_descriptor_freq = (
        int((archive_snapshot_descriptor_counts or {}).get(graph_info.descriptor_key, 0))
        if graph_info.parse_ok
        else 0
    )
    archive_snapshot_backbone_freq = (
        int((archive_snapshot_backbone_signature_counts or {}).get(backbone_signature, 0))
        if graph_info.parse_ok
        else 0
    )
    archive_snapshot_cnn_freq = (
        int((archive_snapshot_cnn_signature_counts or {}).get(cnn_signature, 0))
        if graph_info.parse_ok
        else 0
    )
    archive_snapshot_graph_freq = (
        int((archive_snapshot_graph_counts or {}).get(graph_info.graph_hash, 0))
        if graph_info.parse_ok
        else 0
    )
    archive_snapshot_block_freq = (
        int((archive_snapshot_block_signature_counts or {}).get(block_signature, 0))
        if graph_info.parse_ok and block_signature and block_signature != "incomplete_block"
        else 0
    )
    archive_snapshot_backbone_cnn_freq = (
        int((archive_snapshot_backbone_cnn_pair_counts or {}).get(backbone_cnn_pair_key, 0))
        if graph_info.parse_ok
        else 0
    )
    archive_snapshot_backbone_block_freq = (
        int((archive_snapshot_backbone_block_pair_counts or {}).get(backbone_block_pair_key, 0))
        if graph_info.parse_ok and block_signature and block_signature != "incomplete_block"
        else 0
    )
    cell_best_quality_acc = _optional_float(
        (archive_snapshot_backbone_block_best_quality or {}).get(backbone_block_pair_key)
    )
    global_group_baseline_reward_target_acc = (
        group_baseline_reward_target_acc
        if group_baseline_reward_target_acc is not None
        else _prev_closed_group_mean_reward_target_acc()
    )
    use_backbone_baseline = bool(
        graph_info.parse_ok
        and archive_snapshot_backbone_freq >= BACKBONE_BASELINE_MIN_ARCHIVE_SAMPLES
        and backbone_signature in prev_closed_group_mean_reward_target_by_backbone
    )
    backbone_group_baseline_reward_target_acc = (
        prev_closed_group_mean_reward_target_by_backbone.get(backbone_signature)
        if use_backbone_baseline
        else None
    )
    best_backbone_group_mean_reward_target_acc = (
        best_closed_group_mean_reward_target_by_backbone.get(backbone_signature)
        if graph_info.parse_ok and archive_snapshot_backbone_freq >= BACKBONE_BASELINE_MIN_ARCHIVE_SAMPLES
        else None
    )
    effective_group_baseline_reward_target_acc = (
        backbone_group_baseline_reward_target_acc
        if backbone_group_baseline_reward_target_acc is not None
        else global_group_baseline_reward_target_acc
    )
    dominant_backbone_cnn_signature, dominant_backbone_cnn_share = StageState.dominant_counter_entry(
        {
            str(key): int(value)
            for key, value in (archive_snapshot_backbone_cnn_pair_counts or {}).items()
            if str(key).startswith(f"{backbone_signature}::")
        }
    )
    dominant_backbone_cnn_signature = (
        dominant_backbone_cnn_signature.split("::", 1)[1]
        if dominant_backbone_cnn_signature and "::" in dominant_backbone_cnn_signature
        else dominant_backbone_cnn_signature
    )

    novel_vs_trainset_family = False
    novel_vs_trainset_graph = False
    frozen_train_acc = _optional_float(res.get("frozen_train_acc", res.get("train_acc")))
    frozen_test_acc = _optional_float(res.get("frozen_test_acc", res.get("test_acc", res.get("val_metric"))))
    unfrozen_train_acc = _optional_float(res.get("unfrozen_train_acc"))
    unfrozen_test_acc = _optional_float(res.get("unfrozen_test_acc"))
    train_acc = frozen_train_acc
    test_acc = frozen_test_acc
    reward_target_value = _result_reward_target_value(res)
    if reward_target_value is None and stage_name != STAGE1_STRUCTURE_EXPLORE:
        reward_target_value = frozen_test_acc
    quality_acc_value = _optional_float(frozen_test_acc)
    if quality_acc_value is None:
        quality_acc_value = _optional_float(reward_target_value)
    goal_key = primary_goal_key(prompt_goal_tags or [], prompt_target_pattern)
    best_reward_target_for_goal = _best_reward_target_by_goal().get(goal_key)
    group_train_acc_gain = None
    group_train_acc_improved = False
    group_reward_target_gain = None
    group_reward_target_improved = False
    r_primary = 0.0
    r_tiebreak = 0.0
    r_dense = 0.0
    r_prev_group = 0.0
    r_best_group = 0.0
    r_prev_backbone_group = 0.0
    r_best_backbone_group = 0.0
    r_goal_best = 0.0
    r_goal_match = 0.0
    r_trainset_novelty = 0.0
    r_generalization = 0.0
    r_structure_group = 0.0
    r_structure_archive = 0.0
    r_batch_elite = 0.0
    r_repeat_family = 0.0
    r_plain_fuse_penalty = 0.0
    r_target_structure_penalty = 0.0
    target_structure_positive_suppressed = 0.0
    r_template_penalty = 0.0
    r_history_context = 0.0
    r_no_progress_penalty = 0.0
    r_descriptor_diversity = 0.0
    r_cnn_diversity = 0.0
    r_block_diversity = 0.0
    global_descriptor_archive_reward = 0.0
    global_cnn_archive_reward = 0.0
    block_archive_reward = 0.0
    r_formal_success_signal = 0.0
    stage1_validity_scale = 0.0
    dominant_family_repeat = False
    dominant_descriptor_repeat = False
    dominant_cnn_repeat = False
    plain_parallel_repeat = False
    descriptor_reward_cap_applied = False
    cnn_reward_cap_applied = False
    block_reward_cap_applied = False
    strong_repeat_penalty_applied = False
    repeated_graph_without_refresh = False
    strong_repeat_reasons: List[str] = []
    reward_variant_adjustment: Dict[str, Any] = {}
    executable_candidate = _is_executable_candidate(res, graph_info)
    formal_success_candidate = _is_trainable_candidate(res, graph_info)
    has_formal_epoch = _has_completed_formal_epoch(res)
    discovery_candidate = False
    goal_tag_hit_count, goal_tag_total_count, goal_tag_hit_rate = _goal_tag_match_stats(graph_info, prompt_goal_tags)
    prev_target_train_acc = None
    best_target_train_acc = None
    prev_target_reward_target_acc = None
    best_target_reward_target_acc = None
    backbone_prev_target_reward_target_acc = None
    backbone_best_target_reward_target_acc = None
    beat_prev_target = False
    beat_best_target = False
    beat_prev_backbone_target = False
    beat_best_backbone_target = False
    quality_diversity_eligible = False
    formal_progress_refresh = False
    backbone_reward_target_gain = None
    backbone_reward_target_improved = False

    if executable_candidate:
        novel_vs_trainset_family = graph_info.family_hash not in train_family_hashes
        novel_vs_trainset_graph = graph_info.graph_hash not in train_graph_hashes
        discovery_candidate = bool(graph_info.parse_ok and archive_snapshot_family_freq <= 0)
        if novel_vs_trainset_family:
            r_trainset_novelty = TRAINSET_NOVEL_FAMILY_BONUS
        elif novel_vs_trainset_graph:
            r_trainset_novelty = TRAINSET_NOVEL_GRAPH_BONUS
        r_structure_group, r_structure_archive = _structure_progress_components(
            graph_info,
            batch_same_family_count=batch_same_family_count,
            archive_snapshot_family_freq=archive_snapshot_family_freq,
            novel_vs_trainset_family=novel_vs_trainset_family,
            novel_vs_trainset_graph=novel_vs_trainset_graph,
            shallow_one_shot=shallow_one_shot,
            use_formal_archive_bonus=stage_name != STAGE1_STRUCTURE_EXPLORE,
        )
        if (
            (not group_warmup)
            and dominant_family_hash
            and graph_info.parse_ok
            and graph_info.family_hash == dominant_family_hash
            and not discovery_candidate
        ):
            dominant_family_repeat = True
            r_repeat_family = REPEAT_FAMILY_PENALTY
        if graph_info.is_plain_parallel_triple or plain_dual_backbone_concat:
            plain_parallel_repeat = True
            if stage_name == STAGE1_STRUCTURE_EXPLORE:
                if goal_tag_hit_rate < STAGE1_DISCOVERY_MIN_GOAL_HIT_RATE:
                    r_plain_fuse_penalty = min(r_plain_fuse_penalty, STAGE1_PLAIN_PARALLEL_PENALTY)
                else:
                    r_plain_fuse_penalty = min(r_plain_fuse_penalty, STAGE1_PLAIN_PARALLEL_WARMUP_PENALTY)
            elif (not group_warmup) and not discovery_candidate:
                r_plain_fuse_penalty = min(
                    r_plain_fuse_penalty,
                    PLAIN_DUAL_BACKBONE_FUSE_PENALTY if plain_dual_backbone_concat else PLAIN_FUSE_PENALTY,
                )
    if stage_name == STAGE1_STRUCTURE_EXPLORE:
        reward_target_value = None
        stage1_validity_scale = _stage1_validity_scale(res)
        r_dense = _stage1_validity_reward(res, graph_info)
        r_template_penalty = _template_penalty(
            stage_name=stage_name,
            shallow_one_shot=shallow_one_shot,
            minimal_init_template=minimal_init_template,
        )
        if formal_success_candidate:
            novelty_scale = max(0.35, float(stage1_validity_scale))
            goal_alignment_scale = float(goal_tag_hit_rate or 0.0)
            r_structure_group *= (
                STAGE1_STRUCTURE_GROUP_SCALE
                * float(stage1_validity_scale)
            )
            r_structure_archive *= (
                STAGE1_STRUCTURE_ARCHIVE_SCALE
                * float(stage1_validity_scale)
            )
            if batch_same_descriptor_count == 1:
                r_structure_group += (
                    STAGE1_DESCRIPTOR_BATCH_UNIQUE_BONUS
                    * novelty_scale
                )
                if batch_same_graph_count == 1:
                    r_structure_group += STAGE1_GRAPH_BATCH_UNIQUE_BONUS * novelty_scale
            elif batch_same_descriptor_count > 2:
                descriptor_batch_repeat_penalty = max(
                    STAGE1_DESCRIPTOR_BATCH_REPEAT_MAX_PENALTY,
                    STAGE1_DESCRIPTOR_BATCH_REPEAT_STEP_PENALTY * float(batch_same_descriptor_count - 2),
                )
                r_no_progress_penalty += descriptor_batch_repeat_penalty
                if batch_same_graph_count > 2:
                    graph_batch_repeat_penalty = max(
                        STAGE1_GRAPH_BATCH_REPEAT_MAX_PENALTY,
                        STAGE1_GRAPH_BATCH_REPEAT_STEP_PENALTY * float(batch_same_graph_count - 2),
                    )
                    r_no_progress_penalty += graph_batch_repeat_penalty
            if archive_snapshot_descriptor_freq <= 0:
                r_structure_archive += STAGE1_DESCRIPTOR_ARCHIVE_NOVEL_BONUS * novelty_scale
            elif archive_snapshot_descriptor_freq > 3:
                descriptor_archive_repeat_penalty = max(
                    STAGE1_DESCRIPTOR_ARCHIVE_REPEAT_MAX_PENALTY,
                    STAGE1_DESCRIPTOR_ARCHIVE_REPEAT_STEP_PENALTY * float(archive_snapshot_descriptor_freq - 3),
                )
                r_no_progress_penalty += descriptor_archive_repeat_penalty
            if discovery_candidate and goal_alignment_scale >= STAGE1_DISCOVERY_MIN_GOAL_HIT_RATE:
                r_goal_best = (
                    STAGE1_DISCOVERY_FAMILY_BONUS
                    * novelty_scale
                    * max(0.35, goal_alignment_scale)
                )
            elif novel_vs_trainset_graph and goal_alignment_scale >= STAGE1_DISCOVERY_MIN_GOAL_HIT_RATE:
                r_goal_best = (
                    STAGE1_DISCOVERY_GRAPH_BONUS
                    * novelty_scale
                    * max(0.35, goal_alignment_scale)
                )
            else:
                r_no_progress_penalty += STAGE1_NON_DISCOVERY_EXECUTABLE_PENALTY
            if goal_tag_total_count > 0:
                if goal_alignment_scale <= 0.0:
                    r_no_progress_penalty += STAGE1_ZERO_GOAL_HIT_PENALTY
                elif goal_alignment_scale < 0.5:
                    r_no_progress_penalty += STAGE1_LOW_GOAL_HIT_PENALTY
            if archive_snapshot_family_freq > 0:
                archive_repeat_penalty = max(
                    STAGE1_ARCHIVE_REPEAT_MAX_PENALTY,
                    STAGE1_ARCHIVE_REPEAT_STEP_PENALTY * float(archive_snapshot_family_freq),
                )
                r_no_progress_penalty += archive_repeat_penalty
            if batch_same_family_count >= 3:
                batch_repeat_penalty = max(
                    STAGE1_BATCH_REPEAT_MAX_PENALTY,
                    STAGE1_BATCH_REPEAT_STEP_PENALTY * float(batch_same_family_count - 2),
                )
                r_no_progress_penalty += batch_repeat_penalty
            r_goal_match = STAGE1_GOAL_MATCH_SCALE * goal_tag_hit_rate * novelty_scale
            r_repeat_family = _clip(r_repeat_family, STAGE1_DOMINANT_FAMILY_PENALTY, 0.0)
            r_plain_fuse_penalty = _clip(r_plain_fuse_penalty, STAGE1_PLAIN_PARALLEL_PENALTY, 0.0)
            reward_target_value = _clip(
                STAGE1_STATIC_BASE_SCORE
                + max(0.0, r_dense)
                + max(0.0, r_goal_best)
                + max(0.0, r_structure_group)
                + max(0.0, r_structure_archive),
                0.0,
                1.0,
            )
            if goal_tag_total_count > 0:
                if goal_alignment_scale <= 0.0:
                    reward_target_value = min(float(reward_target_value), STAGE1_ZERO_GOAL_HIT_REWARD_CAP)
                elif goal_alignment_scale < 0.5:
                    reward_target_value = min(float(reward_target_value), STAGE1_LOW_GOAL_HIT_REWARD_CAP)
            if graph_info.is_plain_parallel_triple or plain_dual_backbone_concat:
                reward_target_value = min(float(reward_target_value), STAGE1_PLAIN_PARALLEL_REWARD_CAP)
                if goal_alignment_scale < STAGE1_DISCOVERY_MIN_GOAL_HIT_RATE:
                    reward_target_value = min(float(reward_target_value), STAGE1_OFF_TARGET_PLAIN_PARALLEL_REWARD_CAP)
            shallow_pattern_repeat = bool(
                (not discovery_candidate)
                and shallow_one_shot
                and (archive_snapshot_family_freq > 0 or batch_same_family_count >= 3)
            )
            if (
                (not discovery_candidate and archive_snapshot_family_freq > 5)
                or shallow_pattern_repeat
                or dominant_family_repeat
                or minimal_init_template
            ):
                reward_target_value = min(float(reward_target_value), 0.26)
                if minimal_init_template:
                    reward_target_value = min(float(reward_target_value), 0.18)
            stage1_repeated_block = bool(
                executable_candidate
                and block_signature
                and block_signature != "incomplete_block"
                and not discovery_candidate
                and (archive_snapshot_block_freq > 0 or batch_same_block_count >= 3)
            )
            if stage1_repeated_block:
                reward_target_value = min(float(reward_target_value), STAGE1_REPEATED_BLOCK_REWARD_CAP)
                block_reward_cap_applied = True
        r_history_context = _history_context_reward(
            stage_name=stage_name,
            training_context=training_context,
            executable_candidate=executable_candidate,
            formal_success_candidate=formal_success_candidate,
            discovery_candidate=discovery_candidate,
            novel_vs_trainset_family=novel_vs_trainset_family,
            novel_vs_trainset_graph=novel_vs_trainset_graph,
            dominant_family_repeat=dominant_family_repeat,
            dominant_descriptor_repeat=False,
            shallow_one_shot=shallow_one_shot,
            plain_parallel_repeat=plain_parallel_repeat,
            minimal_init_template=minimal_init_template,
            batch_same_descriptor_count=batch_same_descriptor_count,
            validity_scale=stage1_validity_scale,
        )
        if (reward_target_value is not None) and (effective_group_baseline_reward_target_acc is not None) and (not group_warmup):
            group_reward_target_gain = float(reward_target_value - effective_group_baseline_reward_target_acc)
            group_reward_target_improved = bool(group_reward_target_gain >= GROUP_IMPROVEMENT_DELTA)
        if reward_variant == REWARD_VARIANT_NO_STRUCTURAL_NOVELTY:
            adjusted_components, reward_variant_adjustment = _remove_positive_structural_novelty_components(
                {
                    "r_trainset_novelty": r_trainset_novelty,
                    "r_structure_group": r_structure_group,
                    "r_structure_archive": r_structure_archive,
                }
            )
            r_trainset_novelty = adjusted_components["r_trainset_novelty"]
            r_structure_group = adjusted_components["r_structure_group"]
            r_structure_archive = adjusted_components["r_structure_archive"]
        (
            r_structure_group,
            r_structure_archive,
            r_target_structure_penalty,
            target_structure_positive_suppressed,
        ) = _apply_target_structure_reward_adjustment(
            pattern_detection,
            r_structure_group,
            r_structure_archive,
        )
        r_primary = (
            r_dense
            + r_goal_best
            + r_structure_group
            + r_structure_archive
            + r_repeat_family
            + r_plain_fuse_penalty
            + r_target_structure_penalty
            + r_template_penalty
            + r_history_context
            + r_no_progress_penalty
        )
        r_tiebreak = r_goal_match
        total_reward = _clip(r_primary + r_tiebreak, -2.0, 2.0)
        if block_reward_cap_applied:
            total_reward = min(total_reward, STAGE1_REPEATED_BLOCK_REWARD_CAP)
        total_reward = _apply_stage1_trainability_clamp(res, total_reward, graph_info)
    else:
        if (train_acc is not None) and (group_baseline_train_acc is not None) and (not group_warmup):
            group_train_acc_gain = float(train_acc - group_baseline_train_acc)
            group_train_acc_improved = bool(group_train_acc_gain >= GROUP_IMPROVEMENT_DELTA)
        if (reward_target_value is not None) and (effective_group_baseline_reward_target_acc is not None) and (not group_warmup):
            group_reward_target_gain = float(reward_target_value - effective_group_baseline_reward_target_acc)
            group_reward_target_improved = bool(group_reward_target_gain >= GROUP_IMPROVEMENT_DELTA)
        if (reward_target_value is not None) and (backbone_group_baseline_reward_target_acc is not None) and (not group_warmup):
            backbone_reward_target_gain = float(reward_target_value - backbone_group_baseline_reward_target_acc)
            backbone_reward_target_improved = bool(backbone_reward_target_gain >= GROUP_IMPROVEMENT_DELTA)

        if has_formal_epoch and reward_target_value is not None:
            train_acc_value = float(train_acc or 0.0)
            reward_target_float = float(reward_target_value)
            quality_diversity_eligible = bool(
                formal_success_candidate
                and pattern_detection.get("target_structure_match") is not False
            )
            r_dense = stage_profile["dense_scale"] * _clip(
                0.03 + 0.28 * reward_target_float + 0.04 * max(0.0, train_acc_value - 0.50),
                0.02,
                0.35,
            )
            if formal_success_candidate:
                r_formal_success_signal = FORMAL_SUCCESS_SIGNAL_BONUS
                if pattern_detection.get("target_structure_match") is not False:
                    r_formal_success_signal += TARGET_STRUCTURE_MATCH_BONUS
            if (not group_warmup) and (group_baseline_train_acc is not None):
                prev_target_train_acc = float(group_baseline_train_acc) + GROUP_IMPROVEMENT_DELTA
            best_group_train_acc = _best_closed_group_mean_train_acc()
            if (not group_warmup) and (best_group_train_acc is not None):
                best_target_train_acc = float(best_group_train_acc) + BEST_GROUP_REFRESH_DELTA
            if (not group_warmup) and (global_group_baseline_reward_target_acc is not None):
                prev_target_reward_target_acc = float(global_group_baseline_reward_target_acc) + GROUP_IMPROVEMENT_DELTA
                beat_prev_target = reward_target_float >= prev_target_reward_target_acc
                global_prev_group_reward = stage_profile["prev_group_scale"] * _clip(
                    10.0 * (reward_target_float - prev_target_reward_target_acc),
                    -1.8,
                    1.8,
                )
                r_prev_group = global_prev_group_reward
                if backbone_group_baseline_reward_target_acc is not None:
                    r_prev_group *= float(stage_profile["global_baseline_blend"])
                    backbone_prev_target_reward_target_acc = (
                        float(backbone_group_baseline_reward_target_acc) + GROUP_IMPROVEMENT_DELTA
                    )
                    beat_prev_backbone_target = reward_target_float >= backbone_prev_target_reward_target_acc
                    r_prev_backbone_group = stage_profile["backbone_prev_group_scale"] * _clip(
                        10.0 * (reward_target_float - backbone_prev_target_reward_target_acc),
                        -1.8,
                        1.8,
                    )
            best_group_reward_target_acc = _best_closed_group_mean_reward_target_acc()
            if (not group_warmup) and (best_group_reward_target_acc is not None):
                best_target_reward_target_acc = float(best_group_reward_target_acc) + BEST_GROUP_REFRESH_DELTA
                beat_best_target = reward_target_float >= best_target_reward_target_acc
                global_best_group_reward = stage_profile["best_group_scale"] * _clip(
                    12.0 * (reward_target_float - best_target_reward_target_acc),
                    -1.2,
                    1.2,
                )
                r_best_group = global_best_group_reward
                if best_backbone_group_mean_reward_target_acc is not None:
                    r_best_group *= float(stage_profile["global_baseline_blend"])
                    backbone_best_target_reward_target_acc = (
                        float(best_backbone_group_mean_reward_target_acc) + BEST_GROUP_REFRESH_DELTA
                    )
                    beat_best_backbone_target = reward_target_float >= backbone_best_target_reward_target_acc
                    r_best_backbone_group = stage_profile["backbone_best_group_scale"] * _clip(
                        12.0 * (reward_target_float - backbone_best_target_reward_target_acc),
                        -1.2,
                        1.2,
                    )
            if (
                (not group_warmup)
                and (best_reward_target_for_goal is not None)
                and reward_target_float >= float(best_reward_target_for_goal) + GOAL_REFRESH_DELTA
            ):
                r_goal_best = stage_profile["goal_best_scale"] * GOAL_REFRESH_BONUS
            effective_prev_target_reward_target_acc = (
                backbone_prev_target_reward_target_acc
                if backbone_prev_target_reward_target_acc is not None
                else prev_target_reward_target_acc
            )
            effective_beat_prev_target = (
                beat_prev_backbone_target
                if backbone_prev_target_reward_target_acc is not None
                else beat_prev_target
            )
            if (
                (not group_warmup)
                and effective_prev_target_reward_target_acc is not None
                and not effective_beat_prev_target
            ):
                r_no_progress_penalty = stage_profile["no_progress_scale"] * NO_PROGRESS_PENALTY
            if (frozen_train_acc is not None) and (frozen_test_acc is not None):
                overfit_gap = max(0.0, float(frozen_train_acc) - float(frozen_test_acc) - GENERALIZATION_GAP_TOLERANCE)
                r_generalization = _clip(
                    -GENERALIZATION_PENALTY_SCALE * overfit_gap,
                    GENERALIZATION_PENALTY_CAP,
                    0.0,
                )

        if backbone_prev_target_reward_target_acc is not None or backbone_best_target_reward_target_acc is not None:
            formal_progress_refresh = bool(
                beat_prev_backbone_target
                or beat_best_backbone_target
                or r_goal_best > 0.0
            )
        else:
            formal_progress_refresh = bool(
                beat_prev_target
                or beat_best_target
                or r_goal_best > 0.0
            )
        descriptor_progress_refresh = formal_progress_refresh
        if executable_candidate and graph_info.parse_ok and graph_info.descriptor_key:
            if quality_diversity_eligible and batch_same_descriptor_count == 1:
                r_descriptor_diversity += STAGE23_DESCRIPTOR_BATCH_UNIQUE_BONUS
            elif batch_same_descriptor_count > 1:
                r_descriptor_diversity += max(
                    STAGE23_DESCRIPTOR_BATCH_REPEAT_MAX_PENALTY,
                    STAGE23_DESCRIPTOR_BATCH_REPEAT_STEP_PENALTY * float(batch_same_descriptor_count - 1),
                )

            if quality_diversity_eligible and archive_snapshot_descriptor_freq <= 0:
                r_descriptor_diversity += STAGE23_DESCRIPTOR_ARCHIVE_NOVEL_BONUS
            elif archive_snapshot_descriptor_freq > 1:
                r_descriptor_diversity += max(
                    STAGE23_DESCRIPTOR_ARCHIVE_REPEAT_MAX_PENALTY,
                    STAGE23_DESCRIPTOR_ARCHIVE_REPEAT_STEP_PENALTY * float(archive_snapshot_descriptor_freq - 1),
                )
            if quality_diversity_eligible:
                if archive_snapshot_descriptor_freq <= 0:
                    global_descriptor_archive_reward = STAGE23_GLOBAL_DESCRIPTOR_ARCHIVE_NOVEL_BONUS
                else:
                    global_descriptor_archive_reward = max(
                        STAGE23_GLOBAL_DESCRIPTOR_ARCHIVE_REPEAT_MAX_PENALTY,
                        STAGE23_GLOBAL_DESCRIPTOR_ARCHIVE_REPEAT_MAX_PENALTY
                        * min(
                            1.0,
                            float(archive_snapshot_descriptor_freq)
                            / float(STAGE23_GLOBAL_DESCRIPTOR_ARCHIVE_REPEAT_WINDOW),
                        ),
                    )
                r_descriptor_diversity += global_descriptor_archive_reward

            if (
                (not group_warmup)
                and quality_diversity_eligible
                and dominant_descriptor_key
                and graph_info.descriptor_key != dominant_descriptor_key
                and float(dominant_descriptor_share or 0.0) >= STAGE23_DOMINANT_DESCRIPTOR_SOFT_SHARE
            ):
                r_descriptor_diversity += STAGE23_NON_DOMINANT_DESCRIPTOR_BONUS
            elif (
                (not group_warmup)
                and dominant_descriptor_key
                and graph_info.descriptor_key == dominant_descriptor_key
                and float(dominant_descriptor_share or 0.0) >= STAGE23_DOMINANT_DESCRIPTOR_SOFT_SHARE
                and not descriptor_progress_refresh
            ):
                dominant_descriptor_repeat = True
                if float(dominant_descriptor_share or 0.0) >= STAGE23_DOMINANT_DESCRIPTOR_STRONG_SHARE:
                    r_descriptor_diversity += STAGE23_DOMINANT_DESCRIPTOR_REPEAT_STRONG_PENALTY
                else:
                    r_descriptor_diversity += STAGE23_DOMINANT_DESCRIPTOR_REPEAT_PENALTY

        if executable_candidate and graph_info.parse_ok and cnn_signature:
            if quality_diversity_eligible and batch_same_backbone_cnn_count == 1:
                r_cnn_diversity += STAGE23_CNN_BATCH_UNIQUE_BONUS
            elif batch_same_backbone_cnn_count > 1:
                r_cnn_diversity += max(
                    STAGE23_CNN_BATCH_REPEAT_MAX_PENALTY,
                    STAGE23_CNN_BATCH_REPEAT_STEP_PENALTY * float(batch_same_backbone_cnn_count - 1),
                )

            if quality_diversity_eligible and archive_snapshot_backbone_cnn_freq <= 0:
                r_cnn_diversity += STAGE23_CNN_ARCHIVE_NOVEL_BONUS
            elif archive_snapshot_backbone_cnn_freq > 1:
                r_cnn_diversity += max(
                    STAGE23_CNN_ARCHIVE_REPEAT_MAX_PENALTY,
                    STAGE23_CNN_ARCHIVE_REPEAT_STEP_PENALTY * float(archive_snapshot_backbone_cnn_freq - 1),
                )
            if quality_diversity_eligible:
                if archive_snapshot_cnn_freq <= 0:
                    global_cnn_archive_reward = STAGE23_GLOBAL_CNN_ARCHIVE_NOVEL_BONUS
                else:
                    global_cnn_archive_reward = max(
                        STAGE23_GLOBAL_CNN_ARCHIVE_REPEAT_MAX_PENALTY,
                        STAGE23_GLOBAL_CNN_ARCHIVE_REPEAT_MAX_PENALTY
                        * min(
                            1.0,
                            float(archive_snapshot_cnn_freq)
                            / float(STAGE23_GLOBAL_CNN_ARCHIVE_REPEAT_WINDOW),
                        ),
                    )
                r_cnn_diversity += global_cnn_archive_reward

            if (
                (not group_warmup)
                and quality_diversity_eligible
                and dominant_cnn_signature
                and cnn_signature == dominant_cnn_signature
                and float(dominant_cnn_share or 0.0) >= STAGE23_DOMINANT_CNN_SOFT_SHARE
                and not descriptor_progress_refresh
            ):
                dominant_cnn_repeat = True
                if float(dominant_cnn_share or 0.0) >= STAGE23_DOMINANT_CNN_STRONG_SHARE:
                    r_cnn_diversity += STAGE23_GLOBAL_CNN_REPEAT_STRONG_PENALTY
                else:
                    r_cnn_diversity += STAGE23_GLOBAL_CNN_REPEAT_PENALTY

            if (
                (not group_warmup)
                and quality_diversity_eligible
                and archive_snapshot_backbone_freq >= BACKBONE_BASELINE_MIN_ARCHIVE_SAMPLES
                and dominant_backbone_cnn_signature
                and cnn_signature != dominant_backbone_cnn_signature
                and float(dominant_backbone_cnn_share or 0.0) >= STAGE23_DOMINANT_CNN_SOFT_SHARE
            ):
                r_cnn_diversity += STAGE23_NON_DOMINANT_CNN_BONUS
            elif (
                (not group_warmup)
                and archive_snapshot_backbone_freq >= BACKBONE_BASELINE_MIN_ARCHIVE_SAMPLES
                and dominant_backbone_cnn_signature
                and cnn_signature == dominant_backbone_cnn_signature
                and float(dominant_backbone_cnn_share or 0.0) >= STAGE23_DOMINANT_CNN_SOFT_SHARE
                and not descriptor_progress_refresh
            ):
                dominant_cnn_repeat = True
                if float(dominant_backbone_cnn_share or 0.0) >= STAGE23_DOMINANT_CNN_STRONG_SHARE:
                    r_cnn_diversity += STAGE23_DOMINANT_CNN_REPEAT_STRONG_PENALTY
                else:
                    r_cnn_diversity += STAGE23_DOMINANT_CNN_REPEAT_PENALTY

        if (
            executable_candidate
            and graph_info.parse_ok
            and quality_diversity_eligible
            and block_code
            and not block_contributes_to_forward
        ):
            r_block_diversity += STAGE23_DEAD_BLOCK_PENALTY

        if (
            executable_candidate
            and graph_info.parse_ok
            and quality_diversity_eligible
            and block_signature
            and block_signature != "incomplete_block"
        ):
            if batch_same_block_count == 1:
                r_block_diversity += STAGE23_BLOCK_BATCH_UNIQUE_BONUS
            elif batch_same_block_count > 1:
                r_block_diversity += max(
                    STAGE23_BLOCK_BATCH_REPEAT_MAX_PENALTY,
                    STAGE23_BLOCK_BATCH_REPEAT_STEP_PENALTY * float(batch_same_block_count - 1),
                )
            if archive_snapshot_block_freq <= 0:
                block_archive_reward = STAGE23_BLOCK_ARCHIVE_NOVEL_BONUS
            else:
                block_archive_reward = max(
                    STAGE23_BLOCK_ARCHIVE_REPEAT_MAX_PENALTY,
                    STAGE23_BLOCK_ARCHIVE_REPEAT_MAX_PENALTY
                    * min(
                        1.0,
                        float(archive_snapshot_block_freq)
                        / float(STAGE23_BLOCK_ARCHIVE_REPEAT_WINDOW),
                    ),
                )
            r_block_diversity += block_archive_reward

        r_goal_match = stage_profile["goal_match_scale"] * GOAL_MATCH_REWARD_SCALE * goal_tag_hit_rate
        r_template_penalty = _template_penalty(
            stage_name=stage_name,
            shallow_one_shot=shallow_one_shot,
            minimal_init_template=minimal_init_template,
        )
        r_history_context = _history_context_reward(
            stage_name=stage_name,
            training_context=training_context,
            executable_candidate=executable_candidate,
            formal_success_candidate=formal_success_candidate,
            discovery_candidate=discovery_candidate,
            novel_vs_trainset_family=novel_vs_trainset_family,
            novel_vs_trainset_graph=novel_vs_trainset_graph,
            dominant_family_repeat=dominant_family_repeat,
            dominant_descriptor_repeat=dominant_descriptor_repeat,
            shallow_one_shot=shallow_one_shot,
            plain_parallel_repeat=plain_parallel_repeat,
            minimal_init_template=minimal_init_template,
            batch_same_descriptor_count=batch_same_descriptor_count,
        )
        r_structure_group *= stage_profile["structure_scale"]
        r_structure_archive *= stage_profile["structure_scale"]
        r_repeat_family *= stage_profile["repeat_family_scale"]
        r_plain_fuse_penalty *= stage_profile["plain_fuse_scale"]

        if reward_variant == REWARD_VARIANT_NO_STRUCTURAL_NOVELTY:
            adjusted_components, reward_variant_adjustment = _remove_positive_structural_novelty_components(
                {
                    "r_trainset_novelty": r_trainset_novelty,
                    "r_structure_group": r_structure_group,
                    "r_structure_archive": r_structure_archive,
                    "r_descriptor_diversity": r_descriptor_diversity,
                    "r_cnn_diversity": r_cnn_diversity,
                    "r_block_diversity": r_block_diversity,
                    "global_descriptor_archive_reward": global_descriptor_archive_reward,
                    "global_cnn_archive_reward": global_cnn_archive_reward,
                    "block_archive_reward": block_archive_reward,
                }
            )
            r_trainset_novelty = adjusted_components["r_trainset_novelty"]
            r_structure_group = adjusted_components["r_structure_group"]
            r_structure_archive = adjusted_components["r_structure_archive"]
            r_descriptor_diversity = adjusted_components["r_descriptor_diversity"]
            r_cnn_diversity = adjusted_components["r_cnn_diversity"]
            r_block_diversity = adjusted_components["r_block_diversity"]
            global_descriptor_archive_reward = adjusted_components["global_descriptor_archive_reward"]
            global_cnn_archive_reward = adjusted_components["global_cnn_archive_reward"]
            block_archive_reward = adjusted_components["block_archive_reward"]

        (
            r_structure_group,
            r_structure_archive,
            r_target_structure_penalty,
            target_structure_positive_suppressed,
        ) = _apply_target_structure_reward_adjustment(
            pattern_detection,
            r_structure_group,
            r_structure_archive,
        )
        gated_novelty_components = _stage23_gate_positive_novelty_by_quality(
            quality_acc_value,
            {
                "r_structure_group": r_structure_group,
                "r_structure_archive": r_structure_archive,
                "r_descriptor_diversity": r_descriptor_diversity,
                "r_cnn_diversity": r_cnn_diversity,
                "r_block_diversity": r_block_diversity,
            },
        )
        r_structure_group = gated_novelty_components["r_structure_group"]
        r_structure_archive = gated_novelty_components["r_structure_archive"]
        r_descriptor_diversity = gated_novelty_components["r_descriptor_diversity"]
        r_cnn_diversity = gated_novelty_components["r_cnn_diversity"]
        r_block_diversity = gated_novelty_components["r_block_diversity"]
        r_primary = (
            r_dense
            + r_formal_success_signal
            + r_prev_group
            + r_best_group
            + r_prev_backbone_group
            + r_best_backbone_group
            + r_goal_best
            + r_generalization
            + r_structure_group
            + r_structure_archive
            + r_descriptor_diversity
            + r_cnn_diversity
            + r_block_diversity
            + r_batch_elite
            + r_repeat_family
            + r_plain_fuse_penalty
            + r_target_structure_penalty
            + r_template_penalty
            + r_history_context
            + r_no_progress_penalty
        )
        r_tiebreak = r_goal_match
        total_reward = _clip(r_primary + r_tiebreak, -2.0, 2.0)
        effective_prev_target_reward_target_acc = (
            backbone_prev_target_reward_target_acc
            if backbone_prev_target_reward_target_acc is not None
            else prev_target_reward_target_acc
        )
        effective_beat_prev_target = (
            beat_prev_backbone_target
            if backbone_prev_target_reward_target_acc is not None
            else beat_prev_target
        )
        if has_formal_epoch and effective_prev_target_reward_target_acc is not None and not effective_beat_prev_target:
            total_reward = min(total_reward, stage_profile["non_improving_cap"])
        if has_formal_epoch and dominant_descriptor_repeat:
            total_reward = min(total_reward, stage_profile["descriptor_non_improving_cap"])
            descriptor_reward_cap_applied = True
        if has_formal_epoch and dominant_cnn_repeat:
            total_reward = min(total_reward, stage_profile["descriptor_non_improving_cap"])
            cnn_reward_cap_applied = True
        block_repeat_quality_refresh = bool(r_goal_best > 0.0 or beat_best_backbone_target or beat_best_target)
        repeated_block_without_refresh = bool(
            has_formal_epoch
            and formal_success_candidate
            and block_signature
            and block_signature != "incomplete_block"
            and (archive_snapshot_block_freq > 0 or batch_same_block_count > 1)
            and not block_repeat_quality_refresh
        )
        repeated_graph_without_refresh = bool(
            has_formal_epoch
            and formal_success_candidate
            and graph_info.parse_ok
            and graph_info.graph_hash
            and (archive_snapshot_graph_freq > 0 or batch_same_graph_count > 1)
            and not block_repeat_quality_refresh
        )
        if repeated_block_without_refresh:
            total_reward = min(total_reward, STAGE23_REPEATED_BLOCK_REWARD_CAP)
            block_reward_cap_applied = True
        if dominant_descriptor_repeat:
            strong_repeat_reasons.append("descriptor")
        if dominant_cnn_repeat:
            strong_repeat_reasons.append("backbone_cnn")
        if repeated_block_without_refresh:
            strong_repeat_reasons.append("block")
        if repeated_graph_without_refresh:
            strong_repeat_reasons.append("graph")
        if reward_variant == REWARD_VARIANT_STRONG_REPEAT_PENALTY and strong_repeat_reasons:
            total_reward = min(total_reward, 0.0)
            strong_repeat_penalty_applied = True
        total_reward = _stage23_local_competition_reward(
            total_reward,
            generation_total=_current_generation_total(),
            target_ok=pattern_detection.get("target_structure_match") is not False,
            has_formal_epoch=has_formal_epoch,
            formal_success_candidate=formal_success_candidate,
            quality_acc_value=quality_acc_value,
            cell_archive_freq=archive_snapshot_backbone_block_freq,
            batch_same_cell_count=batch_same_backbone_block_count,
            cell_best_quality_acc=cell_best_quality_acc,
        )
        total_reward = _apply_executability_clamp(res, total_reward, graph_info)

    reward_target_value_for_payload = reward_target_value
    reward_metric_for_payload = stage_reward_metric

    warmup_dense_reward = None
    if stage_name != STAGE1_STRUCTURE_EXPLORE and group_warmup and has_formal_epoch:
        warmup_dense_reward = _compute_warmup_dense_reward(reward_target_value)
        total_reward = float(warmup_dense_reward or 0.0)
        total_reward = _apply_executability_clamp(res, total_reward, graph_info)
    total_reward = _apply_target_structure_final_clamp(
        pattern_detection,
        total_reward,
        r_target_structure_penalty,
    )

    res['reward'] = total_reward
    res['test_acc'] = test_acc
    res['train_acc'] = train_acc
    res['frozen_train_acc'] = frozen_train_acc
    res['frozen_test_acc'] = frozen_test_acc
    res['unfrozen_train_acc'] = unfrozen_train_acc
    res['unfrozen_test_acc'] = unfrozen_test_acc
    res['val_metric'] = frozen_test_acc
    res['seed_accuracy_baseline'] = seed_accuracy_baseline
    res['group_baseline_train_acc'] = group_baseline_train_acc
    res['group_train_acc_gain'] = group_train_acc_gain
    res['group_train_acc_improved'] = group_train_acc_improved
    res['reward_target_metric'] = reward_metric_for_payload
    res['reward_target_value'] = reward_target_value_for_payload
    res['global_group_baseline_reward_target_acc'] = global_group_baseline_reward_target_acc
    res['group_baseline_reward_target_acc'] = effective_group_baseline_reward_target_acc
    res['group_backbone_baseline_reward_target_acc'] = backbone_group_baseline_reward_target_acc
    res['group_reward_target_gain'] = group_reward_target_gain
    res['group_reward_target_improved'] = group_reward_target_improved
    res['backbone_reward_target_gain'] = backbone_reward_target_gain
    res['backbone_reward_target_improved'] = backbone_reward_target_improved
    res['reward_batch_index'] = reward_batch_index
    res['reward_group_id'] = reward_group_id
    res['group_warmup'] = group_warmup
    res['warmup_dense_reward'] = warmup_dense_reward
    res['current_stage_name'] = stage_name
    res['current_stage_index'] = RL_STAGE_TO_INDEX.get(stage_name, 0)
    res['stage_uses_formal_eval'] = _stage_uses_formal_eval(stage_name)
    res['stage_uses_static_only'] = _stage_uses_static_only(stage_name)
    res['best_closed_group_mean_reward_target_acc'] = _best_closed_group_mean_reward_target_acc()
    res['best_closed_group_mean_train_acc'] = _best_closed_group_mean_train_acc()
    res['best_closed_group_mean_test_acc'] = _best_closed_group_mean_test_acc()
    res['best_backbone_group_mean_reward_target_acc'] = best_backbone_group_mean_reward_target_acc
    res['best_reward_target_for_goal'] = best_reward_target_for_goal
    res['r_dense'] = r_dense
    res['r_prev_group'] = r_prev_group
    res['r_best_group'] = r_best_group
    res['r_prev_backbone_group'] = r_prev_backbone_group
    res['r_best_backbone_group'] = r_best_backbone_group
    res['r_goal_best'] = r_goal_best
    res['r_goal_match'] = r_goal_match
    res['r_trainset_novelty'] = r_trainset_novelty
    res['r_generalization'] = r_generalization
    res['r_structure_group'] = r_structure_group
    res['r_structure_archive'] = r_structure_archive
    res['r_descriptor_diversity'] = r_descriptor_diversity
    res['r_cnn_diversity'] = r_cnn_diversity
    res['r_block_diversity'] = r_block_diversity
    res['r_formal_success_signal'] = r_formal_success_signal
    res['r_batch_elite'] = r_batch_elite
    res['r_repeat_family'] = r_repeat_family
    res['r_plain_fuse_penalty'] = r_plain_fuse_penalty
    res['r_target_structure_penalty'] = r_target_structure_penalty
    res['target_structure_positive_suppressed'] = target_structure_positive_suppressed
    res['r_template_penalty'] = r_template_penalty
    res['r_history_context'] = r_history_context
    res['r_no_progress_penalty'] = r_no_progress_penalty
    res['batch_elite_rank'] = None
    res['batch_elite_tier'] = "none"
    res['batch_elite_threshold_passed'] = False
    res['goal_tag_hit_count'] = goal_tag_hit_count
    res['goal_tag_total_count'] = goal_tag_total_count
    res['goal_tag_hit_rate'] = goal_tag_hit_rate
    res['prev_target_reward_target_acc'] = prev_target_reward_target_acc
    res['best_target_reward_target_acc'] = best_target_reward_target_acc
    res['backbone_prev_target_reward_target_acc'] = backbone_prev_target_reward_target_acc
    res['backbone_best_target_reward_target_acc'] = backbone_best_target_reward_target_acc
    res['prev_target_train_acc'] = prev_target_train_acc
    res['best_target_train_acc'] = best_target_train_acc
    res['executable_candidate'] = executable_candidate
    res['discovery_candidate'] = discovery_candidate
    res['formal_success_candidate'] = formal_success_candidate
    res['signature'] = f"{normalize_pattern_name(effective_pattern_name)}_{graph_info.graph_hash[:6]}"
    res['graph_hash'] = graph_info.graph_hash
    res['family_id'] = graph_info.family_id
    res['family_expr'] = graph_info.family_expr
    res['family_hash'] = graph_info.family_hash
    res['backbone_signature'] = backbone_signature
    res['cnn_signature'] = cnn_signature
    res['cnn_expr'] = cnn_expr
    res['block_signature'] = block_signature
    res['block_contributes_to_forward'] = block_contributes_to_forward
    res['backbone_block_pair_key'] = backbone_block_pair_key
    res['plain_dual_backbone_concat'] = plain_dual_backbone_concat
    res['archive_snapshot_backbone_freq'] = archive_snapshot_backbone_freq
    res['archive_snapshot_cnn_freq'] = archive_snapshot_cnn_freq
    res['archive_snapshot_graph_freq'] = archive_snapshot_graph_freq
    res['archive_snapshot_block_freq'] = archive_snapshot_block_freq
    res['archive_snapshot_backbone_cnn_freq'] = archive_snapshot_backbone_cnn_freq
    res['archive_snapshot_backbone_block_freq'] = archive_snapshot_backbone_block_freq
    res['cell_best_quality_acc'] = cell_best_quality_acc
    res['batch_same_graph_count'] = batch_same_graph_count
    res['batch_same_backbone_count'] = batch_same_backbone_count
    res['batch_same_backbone_cnn_count'] = batch_same_backbone_cnn_count
    res['batch_same_block_count'] = batch_same_block_count
    res['batch_same_backbone_block_count'] = batch_same_backbone_block_count
    res['descriptor_key'] = graph_info.descriptor_key
    res['dominant_descriptor_key'] = dominant_descriptor_key
    res['dominant_descriptor_share'] = dominant_descriptor_share
    res['dominant_backbone_cnn_signature'] = dominant_backbone_cnn_signature
    res['dominant_backbone_cnn_share'] = dominant_backbone_cnn_share
    res['unique_descriptor_count'] = len(descriptor_archive_counts)
    res['dominant_descriptor_repeat'] = dominant_descriptor_repeat
    res['dominant_cnn_repeat'] = dominant_cnn_repeat
    res['global_descriptor_archive_reward'] = global_descriptor_archive_reward
    res['global_cnn_archive_reward'] = global_cnn_archive_reward
    res['block_archive_reward'] = block_archive_reward
    res['descriptor_reward_cap_applied'] = descriptor_reward_cap_applied
    res['cnn_reward_cap_applied'] = cnn_reward_cap_applied
    res['block_reward_cap_applied'] = block_reward_cap_applied
    res['reward_variant'] = reward_variant
    res['reward_variant_adjustment'] = reward_variant_adjustment
    res['repeated_graph_without_refresh'] = repeated_graph_without_refresh
    res['strong_repeat_penalty_applied'] = strong_repeat_penalty_applied
    res['strong_repeat_penalty_reasons'] = list(dict.fromkeys(strong_repeat_reasons))
    res['history_exploration_pressure'] = float(training_context.get('exploration_pressure') or 0.0)
    res['minimal_init_template'] = minimal_init_template
    res.update(pattern_detection)
    res['declared_pattern_name'] = pattern_detection["declared_pattern"]
    res['actual_pattern_name'] = pattern_detection["actual_pattern"]
    res['target_pattern_match'] = pattern_detection["target_structure_match"]
    res['graph_expr'] = graph_info.graph_expr
    res['pattern_name'] = effective_pattern_name
    res['suggested_pattern_name'] = graph_info.suggested_pattern_name
    res['open_discovery'] = {
        'r_primary': r_primary,
        'r_tiebreak': r_tiebreak,
        'r_trainset_novelty': r_trainset_novelty,
        'r_dense': r_dense,
        'r_formal_success_signal': r_formal_success_signal,
        'r_prev_group': r_prev_group,
        'r_best_group': r_best_group,
        'r_prev_backbone_group': r_prev_backbone_group,
        'r_best_backbone_group': r_best_backbone_group,
        'r_goal_best': r_goal_best,
        'r_goal_match': r_goal_match,
        'r_generalization': r_generalization,
        'r_structure_group': r_structure_group,
        'r_structure_archive': r_structure_archive,
        'r_descriptor_diversity': r_descriptor_diversity,
        'r_cnn_diversity': r_cnn_diversity,
        'r_block_diversity': r_block_diversity,
        'r_batch_elite': r_batch_elite,
        'r_repeat_family': r_repeat_family,
        'r_plain_fuse_penalty': r_plain_fuse_penalty,
        'r_target_structure_penalty': r_target_structure_penalty,
        'target_structure_positive_suppressed': target_structure_positive_suppressed,
        'r_template_penalty': r_template_penalty,
        'r_history_context': r_history_context,
        'r_no_progress_penalty': r_no_progress_penalty,
        'batch_elite_rank': None,
        'batch_elite_tier': "none",
        'batch_elite_threshold_passed': False,
        'group_baseline_train_acc': group_baseline_train_acc,
        'global_group_baseline_reward_target_acc': global_group_baseline_reward_target_acc,
        'group_baseline_reward_target_acc': effective_group_baseline_reward_target_acc,
        'group_backbone_baseline_reward_target_acc': backbone_group_baseline_reward_target_acc,
        'reward_target_metric': reward_metric_for_payload,
        'reward_target_value': reward_target_value_for_payload,
        'best_closed_group_mean_reward_target_acc': _best_closed_group_mean_reward_target_acc(),
        'best_closed_group_mean_train_acc': _best_closed_group_mean_train_acc(),
        'best_closed_group_mean_test_acc': _best_closed_group_mean_test_acc(),
        'best_backbone_group_mean_reward_target_acc': best_backbone_group_mean_reward_target_acc,
        'best_reward_target_for_goal': best_reward_target_for_goal,
        'goal_tag_hit_count': goal_tag_hit_count,
        'goal_tag_total_count': goal_tag_total_count,
        'goal_tag_hit_rate': goal_tag_hit_rate,
        'prev_target_reward_target_acc': prev_target_reward_target_acc,
        'best_target_reward_target_acc': best_target_reward_target_acc,
        'backbone_prev_target_reward_target_acc': backbone_prev_target_reward_target_acc,
        'backbone_best_target_reward_target_acc': backbone_best_target_reward_target_acc,
        'prev_target_train_acc': prev_target_train_acc,
        'best_target_train_acc': best_target_train_acc,
        'group_train_acc_gain': group_train_acc_gain,
        'group_train_acc_improved': group_train_acc_improved,
        'group_reward_target_gain': group_reward_target_gain,
        'group_reward_target_improved': group_reward_target_improved,
        'backbone_reward_target_gain': backbone_reward_target_gain,
        'backbone_reward_target_improved': backbone_reward_target_improved,
        'reward_batch_index': reward_batch_index,
        'reward_group_id': reward_group_id,
        'group_warmup': group_warmup,
        'prompt_goal_tags': list(prompt_goal_tags or []),
        'batch_same_graph_count': batch_same_graph_count,
        'batch_same_family_count': batch_same_family_count,
        'batch_same_descriptor_count': batch_same_descriptor_count,
        'batch_same_backbone_count': batch_same_backbone_count,
        'batch_same_backbone_cnn_count': batch_same_backbone_cnn_count,
        'batch_same_block_count': batch_same_block_count,
        'batch_same_backbone_block_count': batch_same_backbone_block_count,
        'archive_snapshot_family_freq': archive_snapshot_family_freq,
        'archive_snapshot_descriptor_freq': archive_snapshot_descriptor_freq,
        'archive_snapshot_backbone_freq': archive_snapshot_backbone_freq,
        'archive_snapshot_cnn_freq': archive_snapshot_cnn_freq,
        'archive_snapshot_graph_freq': archive_snapshot_graph_freq,
        'archive_snapshot_block_freq': archive_snapshot_block_freq,
        'archive_snapshot_backbone_cnn_freq': archive_snapshot_backbone_cnn_freq,
        'archive_snapshot_backbone_block_freq': archive_snapshot_backbone_block_freq,
        'cell_best_quality_acc': cell_best_quality_acc,
        'macro_structure_ok': passes_macro_structure_gate(graph_info),
        'is_multi_stage_architecture': is_multi_stage_architecture(graph_info),
        'is_shallow_one_shot_fuse': shallow_one_shot,
        'family_id': graph_info.family_id,
        'family_hash': graph_info.family_hash,
        'backbone_signature': backbone_signature,
        'cnn_signature': cnn_signature,
        'cnn_expr': cnn_expr,
        'block_signature': block_signature,
        'block_contributes_to_forward': block_contributes_to_forward,
        'backbone_block_pair_key': backbone_block_pair_key,
        'plain_dual_backbone_concat': plain_dual_backbone_concat,
        'descriptor_key': graph_info.descriptor_key,
        'dominant_descriptor_key': dominant_descriptor_key,
        'dominant_descriptor_share': dominant_descriptor_share,
        'dominant_backbone_cnn_signature': dominant_backbone_cnn_signature,
        'dominant_backbone_cnn_share': dominant_backbone_cnn_share,
        'unique_descriptor_count': len(descriptor_archive_counts),
        'dominant_descriptor_repeat': dominant_descriptor_repeat,
        'dominant_cnn_repeat': dominant_cnn_repeat,
        'global_descriptor_archive_reward': global_descriptor_archive_reward,
        'global_cnn_archive_reward': global_cnn_archive_reward,
        'block_archive_reward': block_archive_reward,
        'descriptor_reward_cap_applied': descriptor_reward_cap_applied,
        'cnn_reward_cap_applied': cnn_reward_cap_applied,
        'block_reward_cap_applied': block_reward_cap_applied,
        'reward_variant': reward_variant,
        'reward_variant_adjustment': reward_variant_adjustment,
        'repeated_graph_without_refresh': repeated_graph_without_refresh,
        'strong_repeat_penalty_applied': strong_repeat_penalty_applied,
        'strong_repeat_penalty_reasons': list(dict.fromkeys(strong_repeat_reasons)),
        'history_exploration_pressure': float(training_context.get('exploration_pressure') or 0.0),
        'minimal_init_template': minimal_init_template,
        'depth': graph_info.depth,
        'merges': graph_info.merges,
        'max_fan_in': graph_info.max_fan_in,
        'backbone_calls': graph_info.backbone_calls,
        'fractal_calls': graph_info.fractal_calls,
        'stem_calls': graph_info.stem_calls,
        'project_calls': graph_info.project_calls,
        'fuse_calls': graph_info.fuse_calls,
        'is_plain_parallel_triple': graph_info.is_plain_parallel_triple,
        'is_legacy_pattern_name': graph_info.is_legacy_pattern_name,
        'parse_ok': graph_info.parse_ok,
        'novel_vs_trainset_family': novel_vs_trainset_family,
        'novel_vs_trainset_graph': novel_vs_trainset_graph,
        'archive_snapshot_family_freq': archive_snapshot_family_freq,
        'batch_same_family_count': batch_same_family_count,
        'stage_name': stage_name,
        'stage_index': RL_STAGE_TO_INDEX.get(stage_name, 0),
        'stage_uses_formal_eval': _stage_uses_formal_eval(stage_name),
        'stage_uses_static_only': _stage_uses_static_only(stage_name),
        'executable_candidate': executable_candidate,
        'discovery_candidate': discovery_candidate,
        'formal_success_candidate': formal_success_candidate,
    }
    return res


def base_discovery_reward_fn(*args, **kwargs):
    return _base_discovery_reward_fn(*args, **kwargs)


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
        entry["rank"] = _services().distributed_rank()
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
    rank = _services().distributed_rank()
    local_rank = _services().env_int("LOCAL_RANK", 0)
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
    batched_eval_results = _services().evaluate_reward_code_batch(batched_eval_specs)
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
    archive_snapshot_descriptor_counts = dict(descriptor_archive_counts)
    archive_snapshot_backbone_signature_counts = dict(backbone_signature_archive_counts)
    archive_snapshot_cnn_signature_counts = dict(cnn_signature_archive_counts)
    archive_snapshot_graph_counts = dict(graph_archive_counts)
    archive_snapshot_block_signature_counts = dict(block_signature_archive_counts)
    archive_snapshot_backbone_cnn_pair_counts = dict(backbone_cnn_pair_archive_counts)
    archive_snapshot_backbone_block_pair_counts = dict(backbone_block_pair_archive_counts)
    archive_snapshot_backbone_block_best_quality = dict(best_quality_acc_by_backbone_block)
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
        _backbone_block_pair_key(backbone_signature, block_signature)
        for backbone_signature, block_signature in zip(batch_backbone_signatures, batch_block_signatures)
    ]
    scored_results = []

    for position, entry in enumerate(entries):
        index = int(entry["local_index"])
        completion_index = int(entry.get("global_index", index))
        _services().code_logger.log_to_file("=" * 50)
        try:
            res = _services().reward_task_reward_fn(
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
            res = _services().attach_group_context(
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
                _services().code_logger.log_to_file(
                    f"[Reward Dispatch] rank={entry['rank']} batch_index={index}, " + ", ".join(dispatch_parts)
                )
            _services().log_reward_failure_trace(entry, res)
            score = float(res.get("reward", -2.0))
        except _services().persistent_eval_worker_error:
            raise
        except Exception as exc:
            _services().code_logger.log_to_file(f"Reward calculation failed at rank={entry['rank']} index={index}: {exc}")
            res = _services().reward_failure_result(
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


def _recompute_discovery_reward(
    res: Dict[str, Any],
    graph_info,
) -> Tuple[float, float, float]:
    stage_name = str(res.get("current_stage_name") or _current_stage_name())
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


def _apply_batch_elite_bonuses(scored_results: List[Dict[str, Any]], group_context: Dict[str, Any]) -> None:
    if group_context["group_warmup"] or str(group_context.get("current_stage_name")) == STAGE1_STRUCTURE_EXPLORE:
        return

    eligible: List[Tuple[float, Dict[str, Any]]] = []

    for item in scored_results:
        res = item["result"]
        graph_info = item["graph_info"]
        reward_target_value = _result_reward_target_value(res)
        if not _is_executable_candidate(res, graph_info):
            continue
        if not _has_completed_formal_epoch(res):
            continue
        if reward_target_value is None:
            continue
        if res.get("target_structure_match") is False:
            continue
        if _is_repeated_block_without_refresh(res):
            continue
        improved = bool(res.get("group_reward_target_improved") or res.get("backbone_reward_target_improved"))
        if (res.get("dominant_descriptor_repeat") or res.get("dominant_cnn_repeat")) and not improved:
            continue
        if res.get("plain_dual_backbone_concat") and not improved:
            continue
        if _reward_variant_is_strong_repeat_penalty() and _is_strong_repeat_without_refresh(res):
            continue
        eligible.append((float(reward_target_value), item))

    eligible.sort(key=lambda pair: pair[0], reverse=True)
    elite_summaries: List[str] = []
    max_elites = min(len(BATCH_ELITE_SOFT_BONUSES), len(BATCH_ELITE_IMPROVING_BONUSES))
    for rank, (reward_target_float, item) in enumerate(eligible[:max_elites]):
        threshold_baseline = _optional_float(
            item["result"].get("group_backbone_baseline_reward_target_acc", item["result"].get("group_baseline_reward_target_acc"))
        )
        threshold = (
            float(threshold_baseline) + GROUP_IMPROVEMENT_DELTA
            if threshold_baseline is not None
            else None
        )
        threshold_passed = threshold is not None and reward_target_float >= threshold
        tier = "improving" if threshold_passed else "soft"
        bonus = (
            BATCH_ELITE_IMPROVING_BONUSES[rank]
            if threshold_passed
            else BATCH_ELITE_SOFT_BONUSES[rank]
        )
        res = item["result"]
        graph_info = item["graph_info"]
        old_reward = float(res.get("reward", -2.0))
        old_discovery_reward, _, _ = _recompute_discovery_reward(res, graph_info)
        postprocess_delta = old_reward - float(old_discovery_reward)
        if (
            (not _is_repeated_block_without_refresh(res))
            and not (_reward_variant_is_strong_repeat_penalty() and _is_strong_repeat_without_refresh(res))
            and float(res.get("r_no_progress_penalty", 0.0) or 0.0) < 0.0
        ):
            res["r_no_progress_penalty"] = 0.0
        res["r_batch_elite"] = bonus
        res["batch_elite_rank"] = rank + 1
        res["batch_elite_tier"] = tier
        res["batch_elite_threshold_passed"] = threshold_passed
        total_reward, r_primary, r_tiebreak = _recompute_discovery_reward(res, graph_info)
        res["reward"] = _clip(float(total_reward) + postprocess_delta, -2.0, 2.0)
        open_discovery = res.setdefault("open_discovery", {})
        open_discovery["r_batch_elite"] = bonus
        open_discovery["r_primary"] = r_primary
        open_discovery["r_tiebreak"] = r_tiebreak
        open_discovery["batch_elite_rank"] = rank + 1
        open_discovery["batch_elite_tier"] = tier
        open_discovery["batch_elite_threshold_passed"] = threshold_passed
        item["score"] = float(res["reward"])
        elite_summaries.append(
            f"#{rank + 1} target={reward_target_float:.4f} tier={tier} bonus={bonus:.3f} "
            f"struct={float(res.get('r_structure_group', 0.0) or 0.0) + float(res.get('r_structure_archive', 0.0) or 0.0):.3f}"
        )
    if elite_summaries:
        code_logger.log_to_file(
            f"[Reward Batch Elite] reward_batch_index={group_context['reward_batch_index']} "
            + "; ".join(elite_summaries)
        )


def _finalize_scored_results(scored_results: List[Dict[str, Any]]) -> None:
    current_batch_results: List[Dict[str, Any]] = []
    stage_name = _current_stage_name()
    for item in scored_results:
        index = int(item["local_index"])
        prompt = item["prompt"]
        completion = item["completion"]
        graph_info = item["graph_info"]
        goal_key = item["goal_key"]
        res = item["result"]
        score = float(item["score"])
        sig = res.get("signature", "unknown")
        backbone_signature = _result_backbone_signature(res)
        cnn_signature = _result_cnn_signature(res, graph_info)
        block_signature = _result_block_signature(res, completion)
        backbone_cnn_pair_key = _backbone_cnn_pair_key(backbone_signature, cnn_signature)
        backbone_block_pair_key = _backbone_block_pair_key(backbone_signature, block_signature)

        is_executable = _is_executable_candidate(res, graph_info)
        is_trainable = _is_trainable_candidate(res, graph_info)
        reward_target_value = _result_reward_target_value(res)
        quality_acc_value = _optional_float(res.get("frozen_test_acc"))
        if quality_acc_value is None:
            quality_acc_value = _optional_float(reward_target_value)
        if reward_target_value is not None:
            current_batch_results.append(res)
        if is_executable:
            if stage_name != STAGE1_STRUCTURE_EXPLORE or bool(res.get("discovery_candidate")):
                _record_current_group_trainable_sample(goal_key, res, graph_info)
            graph_archive_counts[graph_info.graph_hash] += 1
            family_archive_counts[graph_info.family_id] += 1
            family_hash_archive_counts[graph_info.family_hash] += 1
            descriptor_archive_key = str(res.get("descriptor_key") or getattr(graph_info, "descriptor_key", "") or "")
            if descriptor_archive_key:
                descriptor_archive_counts[descriptor_archive_key] += 1
            if backbone_signature:
                backbone_signature_archive_counts[backbone_signature] += 1
            if cnn_signature:
                cnn_signature_archive_counts[cnn_signature] += 1
            if block_signature and block_signature != "incomplete_block":
                block_signature_archive_counts[block_signature] += 1
            if backbone_signature and cnn_signature:
                backbone_cnn_pair_archive_counts[backbone_cnn_pair_key] += 1
            if backbone_signature and block_signature and block_signature != "incomplete_block":
                backbone_block_pair_archive_counts[backbone_block_pair_key] += 1
                if is_trainable and quality_acc_value is not None:
                    best_quality_acc_by_backbone_block[backbone_block_pair_key] = max(
                        float(best_quality_acc_by_backbone_block.get(backbone_block_pair_key, float("-inf"))),
                        float(quality_acc_value),
                    )
            motif_name_counts[res.get("pattern_name", graph_info.suggested_pattern_name)] += 1
            get_goal_counter(goal_graph_archive_counts, goal_key)[graph_info.graph_hash] += 1
            get_goal_counter(goal_family_hash_archive_counts, goal_key)[graph_info.family_hash] += 1
            current_best = family_metric_best.get(graph_info.family_hash, float("-inf"))
            gain_value = res.get("group_reward_target_gain")
            family_metric_best[graph_info.family_hash] = max(
                current_best,
                float(gain_value if gain_value is not None else float("-inf")),
            )
            if bool(res.get("discovery_candidate")):
                discovery_family_hashes_seen.add(str(graph_info.family_hash))

        _services().code_logger.log_to_file(
            f"Rank {item['rank']} batch index {index}, Motif: {res.get('pattern_name')}, Signature: {sig}, Result: {res}"
        )

        should_save = (
            bool(graph_info)
            and graph_info.parse_ok
            and res.get("built_ok")
            and res.get("forward_shape_ok")
            and res.get("backward_ok")
            and _has_completed_formal_epoch(res)
        )
        save_gate_reason = "ok"
        if should_save and backbone_signature and cnn_signature:
            saved_best = saved_best_reward_target_by_backbone_cnn.get(backbone_cnn_pair_key)
            if (
                saved_backbone_cnn_pair_counts.get(backbone_cnn_pair_key, 0) > 0
                and (
                    reward_target_value is None
                    or (
                        saved_best is not None
                        and float(reward_target_value) < float(saved_best) + SAVE_DUPLICATE_BACKBONE_CNN_DELTA
                    )
                )
            ):
                should_save = False
                save_gate_reason = "duplicate_backbone_cnn_signature"

        if should_save:
            pattern_override = "" if graph_info.has_custom_pattern_name else res.get("suggested_pattern_name", "")
            block_code, init_code, forward_code = extract_reward_completion_blocks(completion)
            if pattern_override:
                init_code = ensure_pattern_name(init_code, pattern_override)
            final_code = reconstruct_code(completion, pattern_name_override=pattern_override)
            normalized_completion = render_completion_xml(block_code, init_code, forward_code)
            out_path = reward_run_epoch_dir(0)
            archive_index = _archive_index()
            model_dir = synth_dir(out_path) / f"B{archive_index}"
            model_dir.mkdir(exist_ok=True, parents=True)

            code_file = model_dir / new_nn_file
            with open(code_file, "w") as handle:
                handle.write(final_code)

            create_file(model_dir, new_out_file, normalized_completion)
            _services().code_logger.log_to_file(f"[INFO] Saved successful code to B{archive_index} (Signature: {sig})")
            saved_graph_counts[graph_info.graph_hash] += 1
            saved_family_hash_counts[graph_info.family_hash] += 1
            if backbone_signature:
                saved_backbone_signature_counts[backbone_signature] += 1
            if cnn_signature:
                saved_cnn_signature_counts[cnn_signature] += 1
            if backbone_signature and cnn_signature:
                saved_backbone_cnn_pair_counts[backbone_cnn_pair_key] += 1
                saved_best_reward_target_by_backbone_cnn[backbone_cnn_pair_key] = max(
                    float(saved_best_reward_target_by_backbone_cnn.get(backbone_cnn_pair_key, float("-inf"))),
                    float(reward_target_value if reward_target_value is not None else float("-inf")),
                )
            if backbone_signature and block_signature and block_signature != "incomplete_block":
                saved_backbone_block_pair_counts[backbone_block_pair_key] += 1
            get_goal_counter(saved_goal_family_hash_counts, goal_key)[graph_info.family_hash] += 1
            _set_archive_index(archive_index + 1)
        elif (
            bool(graph_info)
            and graph_info.parse_ok
            and res.get("built_ok")
            and res.get("forward_shape_ok")
            and res.get("backward_ok")
            and _has_completed_formal_epoch(res)
        ):
            _services().code_logger.log_to_file(
                f"[INFO] Skipped save for signature={sig} backbone={backbone_signature} cnn={cnn_signature} "
                f"reason={save_gate_reason} reward_target={reward_target_value!r}"
            )

        generation_total = _current_generation_total() + 1
        _record_generation_event(
            {
                "generation_total": generation_total,
                "reward_batch_index": res.get("reward_batch_index"),
                "reward_group_id": res.get("reward_group_id"),
                "stage_name": str(res.get("current_stage_name") or _current_stage_name()),
                "stage_index": int(res.get("current_stage_index") or RL_STAGE_TO_INDEX.get(_current_stage_name(), 0)),
                "family_hash": str(res.get("family_hash") or getattr(graph_info, "family_hash", "") or ""),
                "graph_hash": str(res.get("graph_hash") or getattr(graph_info, "graph_hash", "") or ""),
                "descriptor_key": str(res.get("descriptor_key") or getattr(graph_info, "descriptor_key", "") or ""),
                "backbone_signature": backbone_signature,
                "cnn_signature": cnn_signature,
                "block_signature": block_signature,
                "backbone_cnn_pair_key": backbone_cnn_pair_key,
                "backbone_block_pair_key": backbone_block_pair_key,
                "reward": score,
                "reward_target_metric": str(res.get("reward_target_metric") or ""),
                "reward_target_value": reward_target_value,
                "formal_reward_epochs": list(res.get("formal_reward_epochs") or []),
                "formal_reward_max_epoch": int(res.get("formal_reward_max_epoch", 0) or 0),
                "formal_horizon_test_acc": dict(res.get("formal_horizon_test_acc") or {}),
                "formal_horizon_train_acc": dict(res.get("formal_horizon_train_acc") or {}),
                "formal_horizon_scores": dict(res.get("formal_horizon_scores") or {}),
                "formal_reward_target_value": _optional_float(res.get("formal_reward_target_value")),
                "loss_end": _optional_float(res.get("loss_end")),
                "best_epoch_loss": _optional_float(res.get("best_epoch_loss")),
                "avg_epoch_loss": _optional_float(res.get("avg_epoch_loss")),
                "epochs_completed": int(res.get("epochs_completed", 0) or 0),
                "training_context_metric_name": str(res.get("training_context_metric_name") or ""),
                "training_context_metric_value": _optional_float(res.get("training_context_metric_value")),
                "trained_step_ok": bool(res.get("trained_step_ok")),
                "backward_ok": bool(res.get("backward_ok")),
                "loss_drop_ok": bool(res.get("loss_drop_ok")),
                "executable_candidate": bool(res.get("executable_candidate", is_executable)),
                "discovery_candidate": bool(res.get("discovery_candidate")),
                "formal_success_candidate": bool(res.get("formal_success_candidate", is_trainable)),
                "dominant_backbone_signature": dominant_backbone_signature,
                "dominant_backbone_share": dominant_backbone_share,
                "dominant_cnn_signature": dominant_cnn_signature,
                "dominant_cnn_share": dominant_cnn_share,
                "dominant_backbone_cnn_pair": dominant_backbone_cnn_pair,
                "dominant_backbone_cnn_share": dominant_backbone_cnn_share,
            }
        )

        _services().code_logger.log_generation(prompt, completion, score, res)

    update_current_group_metrics(current_batch_results)
    group_close_result = close_reward_group_if_needed()
    if group_close_result is not None:
        _services().code_logger.log_to_file(f"[Reward Group] {group_close_result}")


def apply_batch_elite_bonuses(scored_results, group_context: Dict[str, Any]) -> None:
    _apply_batch_elite_bonuses(scored_results, group_context)


def finalize_scored_results(scored_results) -> None:
    _finalize_scored_results(scored_results)


def print_discovery_metrics() -> None:
    total_valid = sum(family_hash_archive_counts.values())
    unique_count = len(graph_archive_counts)
    unique_families = len(family_archive_counts)
    unique_skeletons = len(family_hash_archive_counts)
    unique_descriptors = len(descriptor_archive_counts)
    unique_backbones = len(backbone_signature_archive_counts)
    unique_cnns = len(cnn_signature_archive_counts)
    unique_blocks = len(block_signature_archive_counts)
    unique_backbone_cnn_pairs = len(backbone_cnn_pair_archive_counts)
    unique_backbone_block_pairs = len(backbone_block_pair_archive_counts)

    if total_valid > 0:
        most_common_count = family_hash_archive_counts.most_common(1)[0][1]
        dominant_share = most_common_count / total_valid
        import math
        entropy = -sum(
            (count / total_valid) * math.log2(count / total_valid)
            for count in family_hash_archive_counts.values()
            if count > 0
        )
    else:
        dominant_share = 0.0
        entropy = 0.0

    print(
        f"\n[Discovery Metrics] Unique Graphs: {unique_count}, "
        f"Families: {unique_families}, Skeletons: {unique_skeletons}, Descriptors: {unique_descriptors}, "
        f"Backbone Buckets: {unique_backbones}, CNN Signatures: {unique_cnns}, Block Signatures: {unique_blocks}, "
        f"Backbone+CNN Pairs: {unique_backbone_cnn_pairs}, Backbone+Block Cells: {unique_backbone_block_pairs}, "
        f"Dominant Family Share: {dominant_share:.2%}, Entropy: {entropy:.2f}"
    )
    print(f"[Graph Archive] Top 5 Exact Graphs: {dict(graph_archive_counts.most_common(5))}")
    print(f"[Family Archive] Top 5 Family IDs: {dict(family_archive_counts.most_common(5))}")
    print(f"[Family Archive] Top 5 Skeletons: {dict(family_hash_archive_counts.most_common(5))}")
    print(f"[Descriptor Archive] Top 5: {dict(descriptor_archive_counts.most_common(5))}")
    print(f"[Backbone Archive] Top 5: {dict(backbone_signature_archive_counts.most_common(5))}")
    print(f"[CNN Archive] Top 5: {dict(cnn_signature_archive_counts.most_common(5))}")
    print(f"[Block Archive] Top 5: {dict(block_signature_archive_counts.most_common(5))}")
    print(f"[Backbone+CNN Archive] Top 5: {dict(backbone_cnn_pair_archive_counts.most_common(5))}")
    print(f"[Backbone+Block Archive] Top 5: {dict(backbone_block_pair_archive_counts.most_common(5))}")
    print(f"[Motif Names] Top 5: {dict(motif_name_counts.most_common(5))}")
    goal_summary = {
        goal_key: len(counter)
        for goal_key, counter in goal_family_hash_archive_counts.items()
    }
    print(f"[Goal Skeleton Coverage] {goal_summary}")


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


def _normalize_backbone_signature_names(backbone_model_names: list[str] | None) -> list[str]:
    normalized = [
        str(name).strip()
        for name in list(backbone_model_names or [])
        if str(name).strip()
    ]
    normalized.sort()
    return normalized


def _build_backbone_signature(backbone_model_names: list[str] | None) -> str:
    normalized = _normalize_backbone_signature_names(backbone_model_names)
    return " + ".join(normalized) if normalized else "unknown_backbone_pair"


def build_backbone_signature(backbone_model_names: list[str] | None) -> str:
    return _build_backbone_signature(backbone_model_names)


def _backbone_cnn_pair_key(backbone_signature: str, cnn_signature: str) -> str:
    return f"{str(backbone_signature or 'unknown_backbone_pair')}::{str(cnn_signature or 'incomplete_cnn')}"


def _backbone_block_pair_key(backbone_signature: str, block_signature: str) -> str:
    return f"{str(backbone_signature or 'unknown_backbone_pair')}::{str(block_signature or 'incomplete_block')}"


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


def _is_plain_dual_backbone_concat(forward_code: str) -> bool:
    source = str(forward_code or "")
    if not ("self.backbone_a" in source and "self.backbone_b" in source):
        return False
    if "torch.cat" not in source or "adaptive_pool_flatten" not in source:
        return False
    structural_tokens = (
        "_feature_to_input_image",
        "self.features",
        "self.stem",
        "self.project",
        "self.bridge",
        "self.adapter",
        "self.fuse",
        "self.fractal",
        "drop_conv3x3_block",
    )
    return not any(token in source for token in structural_tokens)


def _result_backbone_signature(res: Dict[str, Any]) -> str:
    signature = str(res.get("backbone_signature") or "").strip()
    if signature:
        return signature
    return build_backbone_signature(res.get("backbone_model_names"))


def _result_cnn_signature(res: Dict[str, Any], graph_info) -> str:
    signature = str(res.get("cnn_signature") or "").strip()
    if signature:
        return signature
    if graph_info is not None:
        signature = str(getattr(graph_info, "cnn_signature", "") or "").strip()
        if signature:
            return signature
    return "incomplete_cnn"


def _result_block_signature(res: Dict[str, Any], completion: str = "") -> str:
    if res and res.get("block_contributes_to_forward") is False:
        return "incomplete_block"
    signature = str(res.get("block_signature") or "").strip()
    if signature:
        return signature
    block_code, init_code, forward_code = extract_completion_blocks_strict(str(completion or ""))
    if not _block_contributes_to_forward(init_code, forward_code):
        return "incomplete_block"
    return _block_signature_from_code(block_code)


def _record_completion(record: Dict[str, Any]) -> str:
    return str(record.get("completion") or record.get("raw_completion") or "")


def _record_api_result(record: Dict[str, Any]) -> Dict[str, Any]:
    value = record.get("api_result")
    return value if isinstance(value, dict) else {}


def _record_seed_accuracy(record: Dict[str, Any]) -> float:
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
            return _coerce_accuracy_baseline(value, context="replay record accuracy")
    return 0.10


def _entry_from_record(record: Dict[str, Any], *, index: int) -> Dict[str, Any]:
    completion = _record_completion(record)
    block_code, init_code, forward_code = extract_completion_blocks_strict(completion)
    section_info = describe_code_sections(
        block_code=block_code,
        init_code=init_code,
        forward_code=forward_code,
    )
    prompt = str(record.get("prompt") or "")
    prompt_goal_tags = extract_prompt_goal_tags(prompt)
    prompt_target_pattern = extract_prompt_target_pattern(prompt)
    entry = {
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
        "goal_key": primary_goal_key(prompt_goal_tags, prompt_target_pattern),
        "seed_accuracy_baseline": _record_seed_accuracy(record),
    }
    api_result = _record_api_result(record)
    if api_result:
        entry["precomputed_eval_result"] = dict(api_result)
    return entry


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
    eval_cfg_builder = reward_eval_cfg_builder()
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
        final_code = reconstruct_code(completion, pattern_name_override=pattern_override)
        if not final_code:
            continue

        formal_input_shape = _formal_reward_input_shape()
        prm = {
            "lr": 0.01,
            "batch": 64,
            "dropout": 0.3,
            "momentum": 0.9,
            "transform": FORMAL_REWARD_TRANSFORM,
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
                stage_name=str(group_context.get("current_stage_name") or _current_stage_name()),
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
    transform = str(FORMAL_REWARD_TRANSFORM)
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
