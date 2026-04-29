from typing import Dict

import torch


class DTypeSafeLinearWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module

    @property
    def weight(self):
        return getattr(self.module, "weight", None)

    @property
    def bias(self):
        return getattr(self.module, "bias", None)

    def forward(self, inputs, *args, **kwargs):
        weight = getattr(self.module, "weight", None)
        if weight is not None and hasattr(inputs, "dtype") and inputs.dtype != weight.dtype:
            inputs = inputs.to(weight.dtype)
        return self.module(inputs, *args, **kwargs)


def align_generation_head_dtype(model, torch_dtype: torch.dtype, log_prefix: str = "[GEN]") -> None:
    aligned_modules = []
    wrapped_modules = []
    visited_models = set()
    visited_modules = set()
    wrapper_cache: Dict[int, DTypeSafeLinearWrapper] = {}

    def _cast_module(module, label: str) -> None:
        if module is None or not hasattr(module, "weight"):
            return
        if isinstance(module, DTypeSafeLinearWrapper):
            module = module.module
        module_id = id(module)
        if module_id in visited_modules:
            return
        visited_modules.add(module_id)
        weight = getattr(module, "weight", None)
        if weight is None:
            return
        before_dtype = weight.dtype
        if before_dtype == torch_dtype:
            return
        module.to(dtype=torch_dtype)
        aligned_modules.append(f"{label}:{before_dtype}->{torch_dtype}")

    def _ensure_wrapper(module, label: str):
        if module is None or not hasattr(module, "weight"):
            return module
        if isinstance(module, DTypeSafeLinearWrapper):
            return module
        module_id = id(module)
        wrapped = wrapper_cache.get(module_id)
        if wrapped is None:
            wrapped = DTypeSafeLinearWrapper(module)
            wrapper_cache[module_id] = wrapped
            wrapped_modules.append(label)
        return wrapped

    def _walk_model_tree(current_model, prefix: str) -> None:
        if current_model is None:
            return
        model_id = id(current_model)
        if model_id in visited_models:
            return
        visited_models.add(model_id)

        _cast_module(getattr(current_model, "lm_head", None), f"{prefix}.lm_head")
        try:
            _cast_module(current_model.get_output_embeddings(), f"{prefix}.output_embeddings")
        except Exception:
            pass

        head_module = getattr(current_model, "lm_head", None)
        wrapped_head = _ensure_wrapper(head_module, f"{prefix}.lm_head")
        if wrapped_head is not head_module:
            try:
                setattr(current_model, "lm_head", wrapped_head)
            except Exception:
                pass

        try:
            output_module = current_model.get_output_embeddings()
        except Exception:
            output_module = None
        wrapped_output = _ensure_wrapper(output_module, f"{prefix}.output_embeddings")
        if wrapped_output is not output_module and hasattr(current_model, "set_output_embeddings"):
            try:
                current_model.set_output_embeddings(wrapped_output)
            except Exception:
                pass

        for attr_name in ("base_model", "model", "module"):
            nested_model = getattr(current_model, attr_name, None)
            if nested_model is not None and nested_model is not current_model:
                _walk_model_tree(nested_model, f"{prefix}.{attr_name}")

    _walk_model_tree(model, "model")

    config = getattr(model, "config", None)
    if config is not None:
        try:
            config.torch_dtype = torch_dtype
        except Exception:
            pass

    if aligned_modules:
        print(f"{log_prefix} Output dtype alignment: {', '.join(aligned_modules)}")
    if wrapped_modules:
        print(f"{log_prefix} Output dtype safety wrappers: {', '.join(wrapped_modules)}")
