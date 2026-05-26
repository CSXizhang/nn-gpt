import ast
import re
import unittest
from pathlib import Path

from ab.gpt.util import SFTUtil


REPO_ROOT = Path(__file__).resolve().parent


def _source(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _function_source(path: str, function_name: str) -> str:
    text = _source(path)
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            segment = ast.get_source_segment(text, node)
            if segment is None:
                break
            return segment
    raise AssertionError(f"function not found: {path}:{function_name}")


class RewardSmokeMemoryTest(unittest.TestCase):
    def test_torchvision_weights_are_disabled_only_under_smoke_env(self):
        skeleton = SFTUtil.skeleton_code

        self.assertIn("import os", skeleton)
        smoke_index = skeleton.index('os.environ.get("NNGPT_SMOKE_PREVALIDATE") == "1"')
        disable_index = skeleton.index("weights = None", smoke_index)
        get_model_index = skeleton.index("torchvision.models.get_model", disable_index)
        self.assertLess(smoke_index, disable_index)
        self.assertLess(disable_index, get_model_index)
        self.assertIn('weights.strip().lower() in {"", "none"}', skeleton)

    def test_precompute_uses_formal_reward_resize_not_224(self):
        tunerl_source = _source("ab/gpt/TuneRL.py")
        self.assertRegex(tunerl_source, r'FORMAL_REWARD_TRANSFORM\s*=\s*"norm_128_flip"')
        self.assertIn("def _formal_reward_input_shape", tunerl_source)

        for function_name in ("base_discovery_reward_fn", "_build_batched_eval_specs"):
            body = _function_source("ab/gpt/TuneRL.py", function_name)
            self.assertIn("_formal_reward_input_shape()", body)
            self.assertNotIn("(1, 3, 224, 224)", body)

    def test_cpu_smoke_keeps_forward_check_but_cleans_memory(self):
        body = _function_source("ab/gpt/util/Reward.py", "_cpu_smoke_prevalidate_reward_code_impl")

        self.assertIn('NNGPT_SMOKE_PREVALIDATE"] = "1"', body)
        self.assertIn('_safe_bool_env("NNGPT_RL_CPU_SMOKE_STRICT_FORWARD", True)', body)
        self.assertIn('train_setup = getattr(model, "train_setup", None)', body)
        self.assertIn("train_setup(safe_prm)", body)
        self.assertNotIn("model.eval()", body)
        self.assertIn("_clear_exception_frames(exc)", body)
        self.assertIn("_trim_cpu_allocator()", body)
        strict_index = body.index("if strict_forward:")
        forward_index = body.index("output = model(forward_input)")
        self.assertGreater(forward_index, strict_index)
        self.assertEqual(body.count("output = model(forward_input)"), 1)

    def test_cpu_smoke_is_isolated_in_subprocess_by_default(self):
        wrapper = _function_source("ab/gpt/util/Reward.py", "_cpu_smoke_prevalidate_reward_code")
        runner = _function_source("ab/gpt/util/Reward.py", "_run_cpu_smoke_prevalidate_in_subprocess")
        mem_limit = _function_source("ab/gpt/util/Reward.py", "_cpu_smoke_subprocess_memory_limit_bytes")
        child_runtime = _function_source("ab/gpt/util/Reward.py", "_configure_cpu_smoke_child_runtime")

        self.assertIn('_safe_bool_env("NNGPT_RL_CPU_SMOKE_SUBPROCESS", True)', wrapper)
        self.assertIn("_run_cpu_smoke_prevalidate_in_subprocess", wrapper)
        self.assertIn('mp.get_context(method)', runner)
        self.assertIn("ctx.Process", runner)
        self.assertIn("SmokeSubprocessKilled", runner)
        self.assertIn("SLURM_MEM_PER_NODE", mem_limit)
        self.assertIn("0.75", mem_limit)
        self.assertIn('os.environ["CUDA_VISIBLE_DEVICES"] = ""', child_runtime)
        self.assertIn('os.environ["NNGPT_SMOKE_PREVALIDATE"] = "1"', child_runtime)

    def test_no_extra_pre_reward_logging_was_added(self):
        tunerl_source = _source("ab/gpt/TuneRL.py")

        self.assertNotIn("raw_pre_reward", tunerl_source)
        self.assertNotIn("pre_reward_samples", tunerl_source)


if __name__ == "__main__":
    unittest.main()
