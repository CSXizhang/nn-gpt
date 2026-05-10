import unittest

from ab.gpt.util import SFTUtil


class SFTPromptTest(unittest.TestCase):
    def test_backbone_prompt_keeps_backbone_class_without_training_tail(self):
        prompt = SFTUtil.format_backbone_prompt(
            accuracy=0.95,
            target_pattern="A_to_Fractal_plus_B",
        )

        self.assertIn("class TorchVision", prompt)
        self.assertIn("def _feature_to_input_image", prompt)
        self.assertIn("must start with the exact `def __init__", prompt)
        self.assertIn("in_shape` is a 4D batch shape `(B, C, H, W)`", prompt)
        self.assertIn("never use `in_shape[0]` as the channel count", prompt)
        self.assertIn("never set it to `in_shape`", prompt)
        self.assertIn("do not change `self._input_spec` after that call", prompt)
        self.assertIn("supported_hyperparameters()` returns parameter names", prompt)
        self.assertNotIn("def train_setup", prompt)
        self.assertNotIn("def learn", prompt)


if __name__ == "__main__":
    unittest.main()
