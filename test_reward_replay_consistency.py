import json
import math
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.compare_reward_full_noop import BOOL_PATHS, NUMERIC_PATHS, _compare_pair


ROOT = Path(__file__).resolve().parent
FIXTURE_DIR = ROOT / "test" / "fixtures" / "reward_replay"
INPUTS_PATH = FIXTURE_DIR / "current_reward_inputs.jsonl"
GOLDEN_PATH = FIXTURE_DIR / "current_reward_golden.jsonl"


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

    def test_compare_paths_cover_reward_contract(self):
        numeric = {path for label, path in NUMERIC_PATHS if label in {"api", "open_discovery"}}
        bools = {path for label, path in BOOL_PATHS if label in {"api", "open_discovery"}}
        for key in (
            ("reward",),
            ("r_primary",),
            ("r_tiebreak",),
            ("r_cnn_diversity",),
            ("r_block_diversity",),
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

    def test_current_reward_replay_matches_frozen_golden(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "rescored.jsonl"
            env = dict(os.environ)
            env["NNGPT_REWARD_REPLAY_LIGHTWEIGHT"] = "1"
            subprocess.run(
                [
                    "python3",
                    str(ROOT / "scripts" / "score_reward_records.py"),
                    "--input",
                    str(INPUTS_PATH),
                    "--output",
                    str(output),
                    "--reward-variant",
                    "full",
                ],
                cwd=str(ROOT),
                env=env,
                check=True,
            )
            golden = _read_jsonl(GOLDEN_PATH)
            rescored = _read_jsonl(output)
        self.assertEqual(len(golden), len(rescored))
        mismatches = []
        for index, (expected, actual) in enumerate(zip(golden, rescored), start=1):
            mismatches.extend(_compare_pair(index, expected, actual, 1e-12))
            expected_reward = float(expected.get("reward", math.nan))
            actual_reward = float(actual.get("reward", math.nan))
            self.assertEqual(expected_reward > 0.0, actual_reward > 0.0)
        self.assertEqual([], mismatches[:20])


if __name__ == "__main__":
    unittest.main()
