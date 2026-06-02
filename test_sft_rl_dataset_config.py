import importlib
import os
import sys
import types
import unittest
from pathlib import Path


def _install_import_stubs():
    util_pkg_mod = types.ModuleType("ab.gpt.util")
    util_pkg_mod.__path__ = [os.path.join(os.path.dirname(__file__), "ab", "gpt", "util")]
    sys.modules["ab.gpt.util"] = util_pkg_mod

    torch_mod = types.ModuleType("torch")
    torch_mod.Tensor = object
    torch_mod.dtype = object
    torch_mod.float32 = "float32"
    torch_mod.float16 = "float16"
    torch_mod.bfloat16 = "bfloat16"
    torch_mod.device = lambda name: name
    torch_mod.no_grad = lambda fn=None: (lambda inner: inner) if fn is None else fn
    torch_mod.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        is_bf16_supported=lambda: False,
        empty_cache=lambda: None,
        device_count=lambda: 0,
    )
    torch_mod.distributed = types.SimpleNamespace(
        is_available=lambda: False,
        is_initialized=lambda: False,
    )
    nn_mod = types.ModuleType("torch.nn")
    nn_mod.Module = object
    nn_mod.functional = types.ModuleType("torch.nn.functional")
    torch_mod.nn = nn_mod
    utils_mod = types.ModuleType("torch.utils")
    data_mod = types.ModuleType("torch.utils.data")
    data_mod.Dataset = object
    checkpoint_mod = types.ModuleType("torch.utils.checkpoint")
    utils_mod.data = data_mod
    utils_mod.checkpoint = checkpoint_mod
    torch_mod.utils = utils_mod
    sys.modules["torch"] = torch_mod
    sys.modules["torch.nn"] = nn_mod
    sys.modules["torch.nn.functional"] = nn_mod.functional
    sys.modules["torch.utils"] = utils_mod
    sys.modules["torch.utils.data"] = data_mod
    sys.modules["torch.utils.checkpoint"] = checkpoint_mod

    transformers_mod = types.ModuleType("transformers")
    transformers_mod.AutoModelForCausalLM = object
    transformers_mod.AutoTokenizer = object
    transformers_mod.TrainingArguments = object
    transformers_mod.BitsAndBytesConfig = object
    sys.modules["transformers"] = transformers_mod

    peft_mod = types.ModuleType("peft")
    peft_mod.LoraConfig = object
    peft_mod.PeftModel = object
    peft_mod.get_peft_model = lambda model, _config: model
    peft_mod.prepare_model_for_kbit_training = lambda model, *args, **kwargs: model
    sys.modules["peft"] = peft_mod

    trl_mod = types.ModuleType("trl")
    trl_trainer_mod = types.ModuleType("trl.trainer")
    grpo_trainer_mod = types.ModuleType("trl.trainer.grpo_trainer")
    grpo_config_mod = types.ModuleType("trl.trainer.grpo_config")
    grpo_trainer_mod.GRPOTrainer = object
    grpo_config_mod.GRPOConfig = object
    sys.modules["trl"] = trl_mod
    sys.modules["trl.trainer"] = trl_trainer_mod
    sys.modules["trl.trainer.grpo_trainer"] = grpo_trainer_mod
    sys.modules["trl.trainer.grpo_config"] = grpo_config_mod

    datasets_mod = types.ModuleType("datasets")
    datasets_mod.Dataset = object
    sys.modules["datasets"] = datasets_mod

    arch_mod = types.ModuleType("ab.gpt.util.ArchDiscovery")
    arch_mod.ensure_pattern_name = lambda value, *args, **kwargs: value
    arch_mod.extract_graph_info = lambda *args, **kwargs: types.SimpleNamespace()
    arch_mod.normalize_pattern_name = lambda value, *args, **kwargs: value
    sys.modules["ab.gpt.util.ArchDiscovery"] = arch_mod
    util_pkg_mod.ArchDiscovery = arch_mod

    generation_dtype_mod = types.ModuleType("ab.gpt.util.generation_dtype")
    generation_dtype_mod.align_generation_head_dtype = lambda model, _dtype: model
    sys.modules["ab.gpt.util.generation_dtype"] = generation_dtype_mod
    util_pkg_mod.generation_dtype = generation_dtype_mod

    logger_mod = types.ModuleType("ab.gpt.util.simple_logger")
    logger_mod.SimpleCodeLogger = object
    sys.modules["ab.gpt.util.simple_logger"] = logger_mod
    util_pkg_mod.simple_logger = logger_mod

    const_mod = types.ModuleType("ab.gpt.util.Const")
    const_mod.conf_dir = Path("/tmp")
    const_mod.conf_train_dir = lambda *args, **kwargs: ""
    const_mod.conf_test_dir = lambda *args, **kwargs: ""
    const_mod.epoch_dir = lambda *args, **kwargs: ""
    const_mod.new_nn_file = "new_nn.py"
    const_mod.new_out_file = "new_out.txt"
    const_mod.synth_dir = lambda path: path
    sys.modules["ab.gpt.util.Const"] = const_mod
    util_pkg_mod.Const = const_mod

    util_mod = types.ModuleType("ab.gpt.util.Util")
    util_mod.extract_str = lambda text, start, end: ""
    sys.modules["ab.gpt.util.Util"] = util_mod
    util_pkg_mod.Util = util_mod

    sftutil_mod = types.ModuleType("ab.gpt.util.SFTUtil")
    sftutil_mod.legacy_patterns = []
    sftutil_mod.available_backbones = []
    sftutil_mod.open_discovery_goal_profiles = []
    sftutil_mod.open_discovery_prompt_template = ""
    sftutil_mod.open_discovery_rl_prompt_template = ""
    sftutil_mod.open_discovery_skeleton_code = ""
    sys.modules["ab.gpt.util.SFTUtil"] = sftutil_mod
    util_pkg_mod.SFTUtil = sftutil_mod

    reward_mod = types.ModuleType("ab.gpt.util.Reward")

    class _EvalConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    reward_mod.EvalConfig = _EvalConfig
    reward_mod.FORMAL_MULTI_HORIZON_REWARD_TARGET_METRIC = "frozen_test_acc"
    reward_mod.PersistentEvalWorkerError = RuntimeError
    reward_mod.evaluate_code_and_reward = lambda *args, **kwargs: {}
    reward_mod.evaluate_code_and_reward_batch = lambda *args, **kwargs: []
    reward_mod.get_distributed_runtime_info = lambda: {
        "distributed": False,
        "rank": 0,
        "local_rank": 0,
        "visible_gpu_count": 0,
        "visible_gpu_tokens": [],
    }
    reward_mod.get_eval_worker_diagnostics = lambda: None
    reward_mod.prewarm_eval_workers = lambda *args, **kwargs: None
    reward_mod.shutdown_eval_worker = lambda: None
    sys.modules["ab.gpt.util.Reward"] = reward_mod
    util_pkg_mod.Reward = reward_mod

    ab_nn_mod = types.ModuleType("ab.nn")
    ab_nn_api_mod = types.ModuleType("ab.nn.api")
    ab_nn_api_mod.data = lambda *args, **kwargs: None
    ab_nn_mod.api = ab_nn_api_mod
    ab_nn_util_mod = types.ModuleType("ab.nn.util")
    ab_nn_util_util_mod = types.ModuleType("ab.nn.util.Util")
    ab_nn_util_util_mod.create_file = lambda *args, **kwargs: None
    ab_nn_util_mod.Util = ab_nn_util_util_mod
    sys.modules["ab.nn"] = ab_nn_mod
    sys.modules["ab.nn.api"] = ab_nn_api_mod
    sys.modules["ab.nn.util"] = ab_nn_util_mod
    sys.modules["ab.nn.util.Util"] = ab_nn_util_util_mod


class SFTRLDataSetConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_import_stubs()
        sys.path.insert(0, os.path.dirname(__file__))
        cls.tunerl = importlib.import_module("ab.gpt.TuneRL")
        cls.tunerlsft = importlib.import_module("ab.gpt.TuneRLSft")

    def _build_cfg_for_dataset(self, dataset: str):
        old_dataset = os.environ.get("NNGPT_RL_FORMAL_DATASET")
        os.environ["NNGPT_RL_FORMAL_DATASET"] = dataset
        try:
            return self.tunerlsft.build_sft_reward_eval_cfg(
                stage_name=self.tunerl.STAGE2_FORMAL_EXPLORE,
                device="cpu",
            )
        finally:
            if old_dataset is None:
                os.environ.pop("NNGPT_RL_FORMAL_DATASET", None)
            else:
                os.environ["NNGPT_RL_FORMAL_DATASET"] = old_dataset

    def test_cifar100_uses_100_class_output_shape(self):
        cfg = self._build_cfg_for_dataset("cifar-100")

        self.assertEqual(cfg.formal_dataset, "cifar-100")
        self.assertEqual(cfg.n_classes, 100)
        self.assertEqual(tuple(cfg.input_shape), (1, 3, 256, 256))

    def test_imagenette_keeps_10_class_output_shape(self):
        cfg = self._build_cfg_for_dataset("imagenette")

        self.assertEqual(cfg.formal_dataset, "imagenette")
        self.assertEqual(cfg.n_classes, 10)

    def test_cifar100_reward_call_receives_100_class_out_shape(self):
        captured = {}

        def _capture_eval(_code, *, out_shape, cfg, **_kwargs):
            captured["out_shape"] = tuple(out_shape)
            captured["n_classes"] = int(cfg.n_classes)
            captured["formal_dataset"] = cfg.formal_dataset
            return {"reward": 0.0}

        old_eval = self.tunerlsft.RewardUtil.evaluate_code_and_reward
        old_dataset = os.environ.get("NNGPT_RL_FORMAL_DATASET")
        self.tunerlsft.RewardUtil.evaluate_code_and_reward = _capture_eval
        os.environ["NNGPT_RL_FORMAL_DATASET"] = "cifar-100"
        try:
            result = self.tunerlsft.evaluate_code_and_reward_cifar(
                "class Net: pass",
                out_shape=(10,),
                prm={"lr": 0.01, "epoch": 1, "batch": 64},
            )
        finally:
            self.tunerlsft.RewardUtil.evaluate_code_and_reward = old_eval
            if old_dataset is None:
                os.environ.pop("NNGPT_RL_FORMAL_DATASET", None)
            else:
                os.environ["NNGPT_RL_FORMAL_DATASET"] = old_dataset

        self.assertEqual(result["reward"], 0.0)
        self.assertEqual(captured["formal_dataset"], "cifar-100")
        self.assertEqual(captured["n_classes"], 100)
        self.assertEqual(captured["out_shape"], (100,))

    def test_cifar100_batched_eval_specs_use_100_class_out_shape(self):
        old_dataset = os.environ.get("NNGPT_RL_FORMAL_DATASET")
        old_extract = self.tunerl.extract_completion_blocks
        old_reconstruct = self.tunerl.reconstruct_code
        old_builder = getattr(
            self.tunerl.evaluate_code_and_reward,
            "_nngpt_eval_cfg_builder",
            None,
        )
        had_builder = hasattr(self.tunerl.evaluate_code_and_reward, "_nngpt_eval_cfg_builder")
        self.tunerl.extract_completion_blocks = lambda _completion: ("block", "init", "forward")
        self.tunerl.reconstruct_code = lambda *_args, **_kwargs: "class Net: pass"
        self.tunerl.evaluate_code_and_reward._nngpt_eval_cfg_builder = (
            self.tunerlsft.build_sft_reward_eval_cfg
        )
        os.environ["NNGPT_RL_FORMAL_DATASET"] = "cifar-100"
        try:
            _entries, specs = self.tunerl._build_batched_eval_specs(
                [
                    {
                        "completion": "<ok>",
                        "graph_info": types.SimpleNamespace(
                            suggested_pattern_name="ParallelTriple_Shallow",
                            has_custom_pattern_name=False,
                        ),
                        "seed_accuracy_baseline": 0.10,
                        "local_index": 0,
                        "global_index": 0,
                    }
                ],
                group_context={
                    "reward_batch_index": 7,
                    "current_stage_name": self.tunerl.STAGE2_FORMAL_EXPLORE,
                },
            )
        finally:
            self.tunerl.extract_completion_blocks = old_extract
            self.tunerl.reconstruct_code = old_reconstruct
            if had_builder:
                self.tunerl.evaluate_code_and_reward._nngpt_eval_cfg_builder = old_builder
            else:
                delattr(self.tunerl.evaluate_code_and_reward, "_nngpt_eval_cfg_builder")
            if old_dataset is None:
                os.environ.pop("NNGPT_RL_FORMAL_DATASET", None)
            else:
                os.environ["NNGPT_RL_FORMAL_DATASET"] = old_dataset

        self.assertEqual(len(specs), 1)
        self.assertEqual(tuple(specs[0]["out_shape"]), (100,))
        self.assertEqual(specs[0]["cfg"].n_classes, 100)


if __name__ == "__main__":
    unittest.main()
