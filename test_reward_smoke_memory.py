import ast
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from ab.gpt.util import SFTUtil
from ab.gpt.util.ArchDiscovery import normalize_pattern_name


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

    def test_prompt_target_pattern_parser_reads_sft_contract(self):
        function_source = _function_source("ab/gpt/TuneRL.py", "extract_prompt_target_pattern")
        namespace = {"re": re}
        exec(function_source, namespace)
        parse_target = namespace["extract_prompt_target_pattern"]

        prompt = "- Target pattern: `Fractal_to_DualBackbone`. Set self.pattern to this value."
        self.assertEqual(parse_target(prompt), "Fractal_to_DualBackbone")
        self.assertEqual(parse_target("Generate a model."), "")

    def test_target_structure_detection_uses_forward_graph_not_declared_pattern(self):
        namespace = {
            "re": re,
            "Any": object,
            "Dict": dict,
            "List": list,
            "Tuple": tuple,
            "normalize_pattern_name": normalize_pattern_name,
            "TARGET_STRUCTURE_DEAD_BLOCK_PENALTY": -0.25,
            "TARGET_STRUCTURE_DUAL_BACKBONE_PENALTY": -0.20,
            "TARGET_STRUCTURE_PATH_PENALTY": -0.15,
            "TARGET_STRUCTURE_PARSE_PENALTY": -0.30,
            "TARGET_STRUCTURE_PENALTY_FLOOR": -0.45,
        }
        for function_name in (
            "_compact_graph_expr",
            "_graph_has_block_before_backbone",
            "_graph_has_backbone_before_block",
            "build_actual_structure_signature",
            "detect_target_structure",
            "_target_structure_penalty",
            "_apply_target_structure_reward_adjustment",
            "_apply_target_structure_final_clamp",
        ):
            exec(_function_source("ab/gpt/TuneRL.py", function_name), namespace)
        detect_target_structure = namespace["detect_target_structure"]
        apply_adjustment = namespace["_apply_target_structure_reward_adjustment"]
        apply_final_clamp = namespace["_apply_target_structure_final_clamp"]

        dead_direct_fuse = SimpleNamespace(
            pattern_name="B_to_Fractal_plus_A",
            suggested_pattern_name="DualBackbone_Fuse_Deep_123abc",
            parse_ok=True,
            family_id="DualBackboneFuse_Shallow",
            descriptor_key="d4|m1|bb2|fr0|st0|pr0|fu1|fan2",
            backbone_calls=2,
            family_hash="f" * 40,
            cnn_signature="c" * 40,
            graph_expr=(
                "classifier(Cat(PoolFlat(Backbone[efficientnet_b1](Input)), "
                "PoolFlat(Backbone[resnet50](Input))))"
            ),
        )
        dead_result = detect_target_structure(
            prompt_target_pattern="B_to_Fractal_plus_A",
            graph_info=dead_direct_fuse,
            block_contributes_to_forward=False,
            block_signature="deadblock",
        )

        self.assertTrue(dead_result["declared_pattern_matches_prompt"])
        self.assertFalse(dead_result["target_structure_match"])
        self.assertIn("target_fractal_but_block_dead", dead_result["target_structure_mismatch_reasons"])
        self.assertIn("block_dead", dead_result["actual_structure_signature"])
        group, archive, penalty, suppressed = apply_adjustment(dead_result, 0.12, 0.05)
        self.assertEqual((group, archive), (0.0, 0.0))
        self.assertAlmostEqual(penalty, -0.25)
        self.assertAlmostEqual(suppressed, 0.17)
        self.assertAlmostEqual(apply_final_clamp(dead_result, 0.18, penalty), -0.25)
        self.assertAlmostEqual(apply_final_clamp(dead_result, -0.30, -0.45), -0.45)

        live_fractal_branch = SimpleNamespace(
            pattern_name="A_to_Fractal_plus_B",
            suggested_pattern_name="DualBackbone_Fractal_Fuse_456def",
            parse_ok=True,
            family_id="Fractal_DualBackbone_Fuse",
            descriptor_key="d7|m2|bb2|fr0|st0|pr0|fu2|fan3",
            backbone_calls=2,
            family_hash="a" * 40,
            cnn_signature="b" * 40,
            graph_expr=(
                "classifier(Cat(PoolFlat(Sequential[Block](_feature_to_input_image("
                "Backbone[resnet50](Input), 'a'))), PoolFlat(Backbone[efficientnet_b1](Input))))"
            ),
        )
        live_result = detect_target_structure(
            prompt_target_pattern="A_to_Fractal_plus_B",
            graph_info=live_fractal_branch,
            block_contributes_to_forward=True,
            block_signature="liveblock123456789",
        )

        self.assertTrue(live_result["target_structure_match"])
        self.assertIn("block_live", live_result["actual_structure_signature"])
        self.assertIn("liveblock123", live_result["actual_structure_signature"])
        group, archive, penalty, suppressed = apply_adjustment(live_result, 0.12, 0.05)
        self.assertEqual((group, archive, penalty, suppressed), (0.12, 0.05, 0.0, 0.0))
        self.assertAlmostEqual(apply_final_clamp(live_result, 0.18, penalty), 0.18)

    def test_target_structure_clamp_runs_after_warmup_override(self):
        body = _function_source("ab/gpt/TuneRL.py", "base_discovery_reward_fn")

        warmup_index = body.index("warmup_dense_reward = _compute_warmup_dense_reward")
        clamp_index = body.index("_apply_target_structure_final_clamp", warmup_index)
        reward_write_index = body.index("res['reward'] = total_reward", clamp_index)
        self.assertLess(warmup_index, clamp_index)
        self.assertLess(clamp_index, reward_write_index)

    def test_target_structure_gate_covers_reward_postprocessing_exits(self):
        recompute_body = _function_source("ab/gpt/TuneRL.py", "_recompute_discovery_reward")
        recompute_gate_index = recompute_body.index("_apply_target_structure_reward_gate")
        recompute_return_index = recompute_body.index("return total_reward")
        self.assertLess(recompute_gate_index, recompute_return_index)

        elite_body = _function_source("ab/gpt/TuneRL.py", "_apply_batch_elite_bonuses")
        target_skip_index = elite_body.index('res.get("target_structure_match") is False')
        eligible_index = elite_body.index("eligible.append")
        self.assertLess(target_skip_index, eligible_index)

        sft_body = _function_source("ab/gpt/TuneRLSft.py", "raw_reward_fn")
        warmup_index = sft_body.index("warmup_dense_reward")
        compactness_index = sft_body.index("_completion_compactness_penalty")
        gate_index = sft_body.index("_apply_target_structure_reward_gate")
        raw_meta_index = sft_body.index('res["raw_extraction"]')
        self.assertLess(warmup_index, gate_index)
        self.assertLess(compactness_index, gate_index)
        self.assertLess(gate_index, raw_meta_index)

    def test_loss_drop_is_not_a_reward_gate(self):
        gated_functions = [
            _function_source("ab/gpt/TuneRL.py", "_stage1_validity_reward"),
            _function_source("ab/gpt/TuneRL.py", "_apply_trainability_clamp"),
            _function_source("ab/gpt/TuneRL.py", "_apply_stage1_trainability_clamp"),
            _function_source("ab/gpt/TuneRLSft.py", "_is_trainable_architecture"),
            _function_source("ab/gpt/TuneRLSft.py", "_reapply_trainability_clamp"),
        ]

        for function_source in gated_functions:
            self.assertNotIn('res.get("loss_drop_ok")', function_source)

    def test_stage23_local_competition_preserves_early_search_space(self):
        namespace = {
            "Optional": Optional,
            "_clip": lambda value, lower, upper: max(lower, min(upper, float(value))),
            "STAGE23_EARLY_LOCAL_COMPETITION_GENERATIONS": 240,
            "STAGE23_EARLY_CELL_REPEAT_REWARD_CAP": 0.0,
            "STAGE23_DUPLICATE_LOW_ACC_THRESHOLD": 0.92,
            "STAGE23_DUPLICATE_LOW_ACC_REWARD_CAP": 0.03,
            "STAGE23_NEW_CELL_BONUS": 0.04,
            "STAGE23_CELL_IMPROVEMENT_DELTA": 0.003,
            "STAGE23_CELL_IMPROVEMENT_BONUS": 0.08,
            "STAGE23_HIGH_ACC_BONUS_THRESHOLD": 0.92,
            "STAGE23_HIGH_ACC_BONUS": 0.08,
            "STAGE23_HIGH_ACC_STRONG_THRESHOLD": 0.925,
            "STAGE23_HIGH_ACC_STRONG_BONUS": 0.12,
            "STAGE23_HIGH_ACC_ELITE_THRESHOLD": 0.93,
            "STAGE23_HIGH_ACC_ELITE_BONUS": 0.18,
        }
        exec(_function_source("ab/gpt/TuneRL.py", "_stage23_local_competition_reward"), namespace)
        adjust = namespace["_stage23_local_competition_reward"]

        repeated_low_acc = adjust(
            0.22,
            generation_total=32,
            target_ok=True,
            has_formal_epoch=True,
            formal_success_candidate=True,
            quality_acc_value=0.91,
            cell_archive_freq=2,
            batch_same_cell_count=4,
            cell_best_quality_acc=0.918,
        )
        self.assertEqual(repeated_low_acc, 0.0)

        repeated_new_cell_in_batch = adjust(
            0.18,
            generation_total=32,
            target_ok=True,
            has_formal_epoch=True,
            formal_success_candidate=True,
            quality_acc_value=0.89,
            cell_archive_freq=0,
            batch_same_cell_count=2,
            cell_best_quality_acc=None,
        )
        self.assertEqual(repeated_new_cell_in_batch, 0.0)

        new_cell = adjust(
            0.10,
            generation_total=32,
            target_ok=True,
            has_formal_epoch=True,
            formal_success_candidate=True,
            quality_acc_value=0.86,
            cell_archive_freq=0,
            batch_same_cell_count=1,
            cell_best_quality_acc=None,
        )
        self.assertAlmostEqual(new_cell, 0.14)

        improved_cell = adjust(
            0.10,
            generation_total=320,
            target_ok=True,
            has_formal_epoch=True,
            formal_success_candidate=True,
            quality_acc_value=0.922,
            cell_archive_freq=8,
            batch_same_cell_count=3,
            cell_best_quality_acc=0.915,
        )
        self.assertGreaterEqual(improved_cell, 0.26)


if __name__ == "__main__":
    unittest.main()
