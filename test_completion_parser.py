import unittest

from ab.gpt.rl_pipeline.completion import (
    BLOCK_SIGNATURE,
    FORWARD_SIGNATURE,
    INIT_SIGNATURE,
    extract_completion_payload_strict,
)


def _completion(forward_signature: str) -> str:
    return f"""<block>
{BLOCK_SIGNATURE}
    return nn.Sequential()
</block>
<init>
{INIT_SIGNATURE}
    self.backbone_a = TorchVision(model="resnet18")
    self.backbone_b = TorchVision(model="resnet34")
</init>
<forward>
{forward_signature}
    x = self.backbone_a(x) + self.backbone_b(x)
    return x
</forward>"""


class CompletionParserTest(unittest.TestCase):
    def test_accepts_forward_default_spacing_alias(self):
        alias = "def forward(self, x: torch.Tensor, is_probing: bool=False) -> torch.Tensor:"
        (block_code, init_code, forward_code), meta = extract_completion_payload_strict(
            _completion(alias)
        )

        self.assertTrue(block_code)
        self.assertTrue(init_code)
        self.assertTrue(forward_code)
        self.assertTrue(meta["exact_forward_signature"])
        self.assertTrue(forward_code.startswith(FORWARD_SIGNATURE))

    def test_rejects_different_forward_signature(self):
        (_, _, forward_code), meta = extract_completion_payload_strict(
            _completion("def forward(self, x: torch.Tensor) -> torch.Tensor:")
        )

        self.assertFalse(forward_code)
        self.assertFalse(meta["exact_forward_signature"])


if __name__ == "__main__":
    unittest.main()
