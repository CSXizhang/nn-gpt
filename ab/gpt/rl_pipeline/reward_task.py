from typing import Any, Dict, Optional, Protocol, Tuple


class RewardTask(Protocol):
    """Task-owned reward hooks used by the generic RL orchestration layer."""

    name: str
    model_source: str
    tokenizer_source: str
    load_existing_model: bool
    saved_model_path: str
    prompt_template: str

    def extract_completion_blocks(self, completion: str) -> Tuple[str, str, str]:
        ...

    def clear_extraction_meta_cache(self) -> None:
        ...

    def evaluate_code_and_reward(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        ...

    def evaluate_code_and_reward_batch(self, specs: Any) -> Any:
        ...

    def build_eval_cfg(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def reward_fn(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        ...

    def load_rl_dataset(self, tokenizer: Any) -> Any:
        ...

    def extract_seed_context(self, kwargs: Dict[str, Any], expected_count: int) -> Any:
        ...

    def prepare_entries(
        self,
        prompts: Any,
        completions: Any,
        *,
        seed_contexts: Any,
        group_context: Dict[str, Any],
        precompute_eval: bool,
    ) -> Any:
        ...

    def precompute_entries(self, entries: Any, *, group_context: Dict[str, Any]) -> None:
        ...

    def score_entries(
        self,
        entries: Any,
        *,
        group_context: Dict[str, Any],
        archive_snapshot_family_counts: Dict[str, int],
    ) -> Any:
        ...

    def entries_from_records(self, records: Any) -> Any:
        ...

    def describe_code_sections(self, *, block_code: str, init_code: str, forward_code: str) -> Dict[str, Any]:
        ...

    def apply_batch_elite_bonuses(self, scored_results: Any, group_context: Dict[str, Any]) -> None:
        ...

    def finalize_scored_results(self, scored_results: Any) -> None:
        ...

    def capture_runtime_state(self) -> Dict[str, Any]:
        ...

    def restore_runtime_state(self, state: Optional[Dict[str, Any]]) -> None:
        ...

    def reset_runtime_state(self) -> None:
        ...

    def group_context_fields(self) -> Dict[str, Any]:
        ...

    def update_group_metrics(self, results: Any) -> None:
        ...

    def close_group_payload(self) -> Dict[str, Any]:
        ...

    def reset_current_group_state(self) -> None:
        ...

    def reset_stage_comparison_state(self) -> None:
        ...

    def archive_snapshot_family_counts(self) -> Dict[str, int]:
        ...

    def render_prompt_feedback_text(self, *, feedback_char_budget: int = 1200) -> str:
        ...

    def run_log_dir(self) -> str:
        ...

    def run_model_out(self) -> str:
        ...

    def run_epoch_dir(self, *args: Any) -> Any:
        ...


__all__ = ["RewardTask"]
