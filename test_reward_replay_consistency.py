import json
import math
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.compare_reward_full_noop import BOOL_PATHS, LIST_PATHS, NUMERIC_PATHS, STRING_PATHS, _compare_pair


ROOT = Path(__file__).resolve().parent
FIXTURE_DIR = ROOT / "test" / "fixtures" / "reward_replay"
INPUTS_PATH = FIXTURE_DIR / "current_reward_inputs.jsonl"
GOLDEN_PATH = FIXTURE_DIR / "current_reward_golden.jsonl"
DATA_ARCHIVE_INPUTS_PATH = FIXTURE_DIR / "data_archive_reward_inputs.jsonl"
DATA_ARCHIVE_GOLDEN_PATH = FIXTURE_DIR / "data_archive_reward_golden.jsonl"


def _read_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class RewardReplayConsistencyTest(unittest.TestCase):
    def test_real_training_fixture_exists(self):
        self.assertTrue(INPUTS_PATH.exists(), f"missing replay inputs: {INPUTS_PATH}")
        self.assertTrue(GOLDEN_PATH.exists(), f"missing replay golden: {GOLDEN_PATH}")
        inputs = _read_jsonl(INPUTS_PATH)
        golden = _read_jsonl(GOLDEN_PATH)
        self.assertGreaterEqual(len(inputs), 24)
        self.assertEqual(len(inputs), len(golden))

        self.assertTrue(DATA_ARCHIVE_INPUTS_PATH.exists(), f"missing replay inputs: {DATA_ARCHIVE_INPUTS_PATH}")
        self.assertTrue(DATA_ARCHIVE_GOLDEN_PATH.exists(), f"missing replay golden: {DATA_ARCHIVE_GOLDEN_PATH}")
        archive_inputs = _read_jsonl(DATA_ARCHIVE_INPUTS_PATH)
        archive_golden = _read_jsonl(DATA_ARCHIVE_GOLDEN_PATH)
        self.assertGreaterEqual(len(archive_inputs), 80)
        self.assertEqual(len(archive_inputs), len(archive_golden))

    def test_compare_paths_cover_reward_contract(self):
        numeric = {path for label, path in NUMERIC_PATHS if label in {"api", "open_discovery"}}
        bools = {path for label, path in BOOL_PATHS if label in {"api", "open_discovery"}}
        for key in (
            ("reward",),
            ("r_primary",),
            ("r_tiebreak",),
            ("r_cnn_diversity",),
            ("r_block_diversity",),
            ("r_prev_backbone_group",),
            ("r_best_backbone_group",),
            ("backbone_reward_target_gain",),
            ("archive_snapshot_backbone_freq",),
            ("archive_snapshot_cnn_freq",),
            ("archive_snapshot_block_freq",),
            ("archive_snapshot_backbone_cnn_freq",),
            ("archive_snapshot_backbone_block_freq",),
            ("batch_same_backbone_cnn_count",),
            ("batch_same_backbone_block_count",),
            ("reward_target_value",),
            ("formal_reward_target_value",),
        ):
            self.assertIn(key, numeric)
        for key in (
            ("cnn_reward_cap_applied",),
            ("block_reward_cap_applied",),
            ("xml_incomplete_length_cap",),
        ):
            self.assertIn(key, bools)
        strings = {path for label, path in STRING_PATHS if label in {"api", "open_discovery"}}
        for key in (
            ("backbone_signature",),
            ("cnn_signature",),
            ("block_signature",),
            ("backbone_block_pair_key",),
            ("reward_variant",),
        ):
            self.assertIn(key, strings)
        lists = {path for label, path in LIST_PATHS if label in {"api", "open_discovery"}}
        for key in (
            ("strong_repeat_penalty_reasons",),
            ("target_structure_mismatch_reasons",),
        ):
            self.assertIn(key, lists)

    def _assert_replay_matches(self, inputs_path: Path, golden_path: Path) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "rescored.jsonl"
            env = dict(os.environ)
            env["NNGPT_REWARD_REPLAY_LIGHTWEIGHT"] = "1"
            subprocess.run(
                [
                    "python3",
                    str(ROOT / "scripts" / "score_reward_records.py"),
                    "--input",
                    str(inputs_path),
                    "--output",
                    str(output),
                    "--reward-variant",
                    "full",
                ],
                cwd=str(ROOT),
                env=env,
                check=True,
            )
            golden = _read_jsonl(golden_path)
            rescored = _read_jsonl(output)
        self.assertEqual(len(golden), len(rescored))
        mismatches = []
        for index, (expected, actual) in enumerate(zip(golden, rescored), start=1):
            mismatches.extend(_compare_pair(index, expected, actual, 1e-12))
            expected_reward = float(expected.get("reward", math.nan))
            actual_reward = float(actual.get("reward", math.nan))
            self.assertEqual(expected_reward > 0.0, actual_reward > 0.0)
        self.assertEqual([], mismatches[:20])

    def test_current_reward_replay_matches_frozen_golden(self):
        self._assert_replay_matches(INPUTS_PATH, GOLDEN_PATH)

    def test_data_archive_reward_replay_matches_frozen_golden(self):
        self._assert_replay_matches(DATA_ARCHIVE_INPUTS_PATH, DATA_ARCHIVE_GOLDEN_PATH)

    def test_registered_sft_task_preserves_formal_eval_contract(self):
        code = """
import json
from scripts import reward_replay_stubs
reward_replay_stubs.install()
from ab.gpt import TuneRL, TuneRLSft
TuneRLSft.configure_sft_runtime()
cfg = TuneRL.reward_eval_cfg_builder()(
    stage_name=TuneRL.STAGE2_FORMAL_EXPLORE,
    in_shape=TuneRL._formal_reward_input_shape(),
    out_shape=(10,),
    prm={"epoch": 1, "batch": 64},
    device="cpu",
)
print(json.dumps({
    "task": TuneRL.current_reward_task().name,
    "metric": cfg.reward_target_metric,
    "formal_nn_eval": cfg.formal_nn_eval,
    "transform": TuneRL.FORMAL_REWARD_TRANSFORM,
    "input_shape": list(cfg.input_shape),
    "formal_epochs_default": __import__("os").environ.get("NNGPT_RL_FORMAL_REWARD_EPOCHS", "1,5,10"),
}))
"""
        env = dict(os.environ)
        env["NNGPT_REWARD_REPLAY_LIGHTWEIGHT"] = "1"
        env.pop("NNGPT_RL_FORMAL_REWARD_EPOCHS", None)
        result = subprocess.run(
            ["python3", "-c", code],
            cwd=str(ROOT),
            env=env,
            check=True,
            text=True,
            capture_output=True,
        )
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(payload["task"], "backbone_sft")
        self.assertEqual(payload["metric"], "formal_multi_horizon_acc")
        self.assertTrue(payload["formal_nn_eval"])
        self.assertEqual(payload["transform"], "norm_128_flip")
        self.assertEqual(payload["input_shape"], [1, 3, 128, 128])
        self.assertEqual(payload["formal_epochs_default"], "1,5,10")


if __name__ == "__main__":
    unittest.main()
