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
        body = _function_source("ab/gpt/util/Reward.py", "_cpu_smoke_prevalidate_reward_code")

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

    def test_no_extra_pre_reward_logging_was_added(self):
        tunerl_source = _source("ab/gpt/TuneRL.py")

        self.assertNotIn("raw_pre_reward", tunerl_source)
        self.assertNotIn("pre_reward_samples", tunerl_source)

    def test_block_contribution_detects_fractal_unit_feature_path(self):
        function_source = _function_source("ab/gpt/TuneRL.py", "_block_contributes_to_forward")
        namespace = {"re": re}
        exec(function_source, namespace)
        contributes = namespace["_block_contributes_to_forward"]

        init_code = """
def __init__(self, in_shape, out_shape, prm, device):
    self.features = nn.Sequential()
    self.features.add_module("unit1", FractalUnit(3, 64, 2, 0.15, 0.1))
"""
        forward_code = """
def forward(self, x, is_probing=False):
    x_f_map = self.features(x)
    return adaptive_pool_flatten(x_f_map)
"""
        bypass_forward_code = """
def forward(self, x, is_probing=False):
    return self.backbone_a(x)
"""

        self.assertTrue(contributes(init_code, forward_code))
        self.assertFalse(contributes(init_code, bypass_forward_code))


if __name__ == "__main__":
    unittest.main()
