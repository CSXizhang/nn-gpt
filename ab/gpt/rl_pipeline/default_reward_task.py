from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ab.gpt.rl_pipeline import backbone_reward_runtime as BackboneRewardRuntime


class OpenDiscoveryRewardTask:
    name = "open_discovery"

    def __init__(self, rl_module: Any) -> None:
        self._rl = rl_module

    def configure_runtime_services(self, services: Any) -> None:
        BackboneRewardRuntime.configure_runtime_services(services)

    @property
    def model_source(self) -> str:
        return self._rl.base_model

    @property
    def tokenizer_source(self) -> str:
        return self._rl.tokenizer_source

    @property
    def load_existing_model(self) -> bool:
        return self._rl.LOAD_EXISTING_MODEL

    @property
    def saved_model_path(self) -> str:
        return self._rl.SAVED_MODEL_PATH

    @property
    def prompt_template(self) -> str:
        return BackboneRewardRuntime.PROMPT_TEMPLATE

    def extract_completion_blocks(self, completion: str) -> Tuple[str, str, str]:
        return BackboneRewardRuntime.extract_completion_blocks(completion)

    def clear_extraction_meta_cache(self) -> None:
        self._rl.clear_extraction_meta_cache()

    def evaluate_code_and_reward(self, *args, **kwargs):
        return self._rl.evaluate_code_and_reward(*args, **kwargs)

    def evaluate_code_and_reward_batch(self, specs):
        return self._rl.evaluate_code_and_reward_batch(specs)

    def build_eval_cfg(self, *args, **kwargs):
        return self._rl.build_stage_eval_cfg(*args, **kwargs)

    def reward_fn(self, *args, **kwargs):
        return BackboneRewardRuntime.base_discovery_reward_fn(*args, **kwargs)

    def load_rl_dataset(self, tokenizer):
        return BackboneRewardRuntime.load_rl_dataset(tokenizer)

    def extract_seed_context(self, kwargs: Dict[str, Any], expected_count: int):
        return self._rl.require_sample_accuracy_baselines(kwargs, expected_count)

    def bootstrap_trainset_reference_library(self, data) -> None:
        BackboneRewardRuntime.bootstrap_trainset_reference_library(data)

    def prepare_entries(
        self,
        prompts,
        completions,
        *,
        seed_contexts,
        group_context: Dict[str, Any],
        precompute_eval: bool,
    ) -> List[Dict[str, Any]]:
        return BackboneRewardRuntime.prepare_entries(
            prompts,
            completions,
            seed_contexts=seed_contexts,
            group_context=group_context,
            precompute_eval=precompute_eval,
        )

    def precompute_entries(self, entries: List[Dict[str, Any]], *, group_context: Dict[str, Any]) -> None:
        BackboneRewardRuntime.precompute_entries(entries, group_context=group_context)

    def score_entries(
        self,
        entries: List[Dict[str, Any]],
        *,
        group_context: Dict[str, Any],
        archive_snapshot_counts: Dict[str, int],
    ) -> List[Dict[str, Any]]:
        return BackboneRewardRuntime.score_entries(
            entries,
            group_context=group_context,
            archive_snapshot_family_counts=archive_snapshot_counts,
        )

    def entries_from_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return BackboneRewardRuntime.entries_from_records(records)

    def describe_code_sections(self, *, block_code: str, init_code: str, forward_code: str) -> Dict[str, Any]:
        return BackboneRewardRuntime.describe_code_sections(
            block_code=block_code,
            init_code=init_code,
            forward_code=forward_code,
        )

    def apply_batch_elite_bonuses(self, scored_results: List[Dict[str, Any]], group_context: Dict[str, Any]) -> None:
        BackboneRewardRuntime.apply_batch_elite_bonuses(scored_results, group_context)

    def finalize_scored_results(self, scored_results: List[Dict[str, Any]]) -> None:
        BackboneRewardRuntime.finalize_scored_results(scored_results)

    def capture_runtime_state(self) -> Dict[str, Any]:
        return BackboneRewardRuntime.capture_runtime_state()

    def restore_runtime_state(self, state) -> None:
        BackboneRewardRuntime.restore_runtime_state(state)

    def reset_runtime_state(self) -> None:
        BackboneRewardRuntime.reset_runtime_state()

    def group_context_fields(self) -> Dict[str, Any]:
        return BackboneRewardRuntime.group_context_fields()

    def update_group_metrics(self, results) -> None:
        BackboneRewardRuntime.update_group_metrics(results)

    def close_group_payload(self) -> Dict[str, Any]:
        return BackboneRewardRuntime.close_group_payload()

    def reset_current_group_state(self) -> None:
        BackboneRewardRuntime.reset_current_group_state()

    def reset_stage_comparison_state(self) -> None:
        BackboneRewardRuntime.reset_stage_comparison_state()

    def archive_snapshot_counts(self) -> Dict[str, int]:
        return BackboneRewardRuntime.archive_snapshot_family_counts()

    def recovery_marker_count(self) -> int:
        return BackboneRewardRuntime.recovery_marker_count()

    def print_metrics(self) -> None:
        BackboneRewardRuntime.print_discovery_metrics()

    def build_group_feedback_summary(
        self,
        *,
        goal_key: str,
        res: Dict[str, Any],
        candidate_info,
        reward_group_id: int,
    ) -> Dict[str, Any]:
        return BackboneRewardRuntime.build_group_feedback_summary(
            goal_key=goal_key,
            res=res,
            graph_info=candidate_info,
            reward_group_id=reward_group_id,
        )

    def render_prompt_feedback_text(self, *, feedback_char_budget: int = 1200) -> str:
        return BackboneRewardRuntime.render_prompt_feedback_text(feedback_char_budget=feedback_char_budget)

    def training_context_guidance(self, summary: Dict[str, Any]) -> str:
        return BackboneRewardRuntime.training_context_guidance(summary)

    def stage1_gate_ready(self) -> bool:
        return BackboneRewardRuntime.stage1_gate_ready(self._rl)

    def stage1_trainable_stable_ready(self):
        return BackboneRewardRuntime.stage1_trainable_stable_ready(self._rl)

    def stage1_force_promotion_ready(self):
        return BackboneRewardRuntime.stage1_force_promotion_ready(self._rl)

    def stage2_gate_ready(self) -> bool:
        return BackboneRewardRuntime.stage2_gate_ready(self._rl)

    def stage_gate_snapshot(self) -> Dict[str, Any]:
        return BackboneRewardRuntime.stage_gate_snapshot(self._rl)

    def log_reward_failure_trace(self, entry: Dict[str, Any], res: Dict[str, Any]) -> None:
        BackboneRewardRuntime.log_reward_failure_trace(entry, res)

    def run_log_dir(self) -> str:
        return self._rl.run_log_dir()

    def run_model_out(self) -> str:
        return self._rl.run_model_out()

    def run_epoch_dir(self, *args):
        return self._rl.run_epoch_dir(*args)


def create_default_reward_task(rl_module: Any) -> OpenDiscoveryRewardTask:
    return OpenDiscoveryRewardTask(rl_module)


__all__ = ["OpenDiscoveryRewardTask", "create_default_reward_task"]
