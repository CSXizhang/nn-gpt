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

    def apply_batch_elite_bonuses(self, scored_results: Any, group_context: Dict[str, Any]) -> None:
        ...

    def finalize_scored_results(self, scored_results: Any) -> None:
        ...

    def run_log_dir(self) -> str:
        ...

    def run_model_out(self) -> str:
        ...

    def run_epoch_dir(self, *args: Any) -> Any:
        ...


__all__ = ["RewardTask"]
