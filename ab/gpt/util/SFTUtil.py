import ast
import textwrap

available_backbones = ['convnext_tiny', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_v2_s', 'googlenet', 'inception_v3', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3', 'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small', 'regnet_x_1_6gf', 'regnet_x_3_2gf', 'regnet_x_400mf', 'regnet_x_800mf', 'regnet_y_1_6gf', 'regnet_y_3_2gf', 'regnet_y_400mf', 'regnet_y_800mf', 'resnet18', 'resnet34', 'resnet50', 'resnext50_32x4d', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'squeezenet1_0', 'squeezenet1_1', 'swin_t', 'swin_v2_t']

available_patterns = ['Parallel_Triple', 'Ensemble_Backbones_to_Fractal', 'Split_A_Parallel_BF']

skeleton_code = """import torch
import torch.nn as nn
import numpy as np
import gc
import torchvision
from torch.nn import MaxPool2d
from torch.amp import autocast, GradScaler

# ==========================================
# 1. FIXED INFRASTRUCTURE (DO NOT MODIFY)
# ==========================================
class TorchVision(nn.Module):
    def __init__(self, model: str, weights: str = "DEFAULT", unwrap: bool = True, truncate: int = 1, in_channels: int = 3):
        super().__init__()
        self.adapter = nn.Conv2d(in_channels, 3, kernel_size=1) if in_channels != 3 else nn.Identity()
        kwargs = {"aux_logits": False} if "inception" in model.lower() else {}
        try:
            if hasattr(torchvision.models, "get_model"):
                self.m = torchvision.models.get_model(model, weights=weights, **kwargs)
            else:
                self.m = torchvision.models.__dict__[model](pretrained=bool(weights), **kwargs)
        except:
            if hasattr(torchvision.models, "get_model"):
                self.m = torchvision.models.get_model(model, weights=weights)
            else:
                self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
        
        if unwrap:
            layers = []
            for name, module in self.m.named_children():
                if "aux" in name.lower(): continue
                layers.append(module)
            self.m = nn.Sequential(*(layers[:-truncate] if truncate else layers))
        else:
            self.m.head = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        return self.m(self.adapter(x))

def adaptive_pool_flatten(x):
    if x.ndim == 4: return torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
    if x.ndim == 3: return x.mean(dim=1)
    return x.flatten(1) if x.ndim > 2 else x

def autocast_ctx(enabled=True):
    return autocast("cuda", enabled=enabled)
def make_scaler(enabled=True):
    return GradScaler("cuda", enabled=enabled)

def supported_hyperparameters():
    return { 'lr', 'dropout', 'momentum' }

# ==========================================
# 2. DYNAMIC COMPONENTS (TO BE IMPLEMENTED)
# ==========================================

def drop_conv3x3_block(in_channels, out_channels, stride=1, padding=1, bias=False, dropout_prob=0.0):

class FractalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_columns, loc_drop_prob, dropout_prob):
        super().__init__()
        self.num_columns = int(num_columns)
        depth = 2 ** max(self.num_columns - 1, 0)
        blocks = []
        for i in range(depth):
            level = nn.ModuleList()
            for j in range(self.num_columns):
                if (i + 1) % (2 ** j) == 0:
                    in_ch_ij = in_channels if (i + 1 == 2 ** j) else out_channels
                    level.append(drop_conv3x3_block(in_ch_ij, out_channels, dropout_prob=dropout_prob))
            blocks.append(level)
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        outs = [x] * self.num_columns
        for level_block in self.blocks:
            outs_i = [blk(inp) for blk, inp in zip(level_block, outs)]
            joined = torch.stack(outs_i, dim=0).mean(dim=0)
            outs[:len(level_block)] = [joined] * len(level_block)
        return outs[0]

class FractalUnit(nn.Module):
    def __init__(self, in_channels, out_channels, num_columns, loc_drop_prob, dropout_prob):
        super().__init__()
        self.block = FractalBlock(in_channels, out_channels, num_columns, loc_drop_prob, dropout_prob)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(self.block(x))

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        

    def infer_dimensions_dynamically(self, in_shape, num_classes):
        self.to(self.device)
        self.eval()
        with torch.no_grad():
            C = in_shape[1] if len(in_shape)==4 else in_shape[0]
            dummy = torch.zeros(1, C, 224, 224).to(self.device)
            output_feat = self.forward(dummy, is_probing=True)
            dim_fused = output_feat.shape[1]
        self.classifier = nn.Linear(dim_fused, num_classes)
        self.train()

    @staticmethod
    def _norm4d(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4: return x
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            return x.reshape(B * T, C, H, W)
        raise ValueError(f"Expected 4D/5D input, got {tuple(x.shape)}")

    def forward(self, x: torch.Tensor, is_probing: bool = False) -> torch.Tensor:
        
    def train_setup(self, prm):
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])
        self._scaler = make_scaler(enabled=self.use_amp)

    def learn(self, train_data):
        self.train()
        scaler = self._scaler
        train_iter = iter(train_data)
        try:
            for batch_idx, (inputs, labels) in enumerate(train_iter):
                inputs = inputs.to(self.device).float()
                labels = labels.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                with autocast_ctx(enabled=self.use_amp):
                    outputs = self(inputs)
                    loss = self.criterion(outputs, labels)
                if not torch.isfinite(loss): continue
                if self.use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.parameters(), 3.0)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.parameters(), 3.0)
                    self.optimizer.step()
        finally:
            if hasattr(train_iter, 'shutdown'): train_iter.shutdown()
            del train_iter
            gc.collect()
"""

torchvision_skeleton_code = """
class TorchVision(nn.Module):
    def __init__(self, model: str, weights: str = "DEFAULT", unwrap: bool = True, truncate: int = 1, in_channels: int = 3):
        super().__init__()
        self.adapter = nn.Conv2d(in_channels, 3, kernel_size=1) if in_channels != 3 else nn.Identity()
        kwargs = {"aux_logits": False} if "inception" in model.lower() else {}
        try:
            if hasattr(torchvision.models, "get_model"):
                self.m = torchvision.models.get_model(model, weights=weights, **kwargs)
            else:
                self.m = torchvision.models.__dict__[model](pretrained=bool(weights), **kwargs)
        except:
            if hasattr(torchvision.models, "get_model"):
                self.m = torchvision.models.get_model(model, weights=weights)
            else:
                self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
        
        if unwrap:
            layers = []
            for name, module in self.m.named_children():
                if "aux" in name.lower(): continue
                layers.append(module)
            self.m = nn.Sequential(*(layers[:-truncate] if truncate else layers))
        else:
            self.m.head = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        return self.m(self.adapter(x))
"""

prompt_template = """
Role & Context
You are an expert AI architect specialized in deep learning. Your task is to implement the missing components (`pass` blocks) of a strict PyTorch model skeleton achieves an accuracy of {accuracy}.

Input Context
The following code skeleton provides fixed infrastructure (Wrappers, AMP, Training Loop). You must keep all existing code intact and ONLY implement the missing logic.

[CODE SKELETON START]
{skeleton_code}
[CODE SKELETON END]

Implementation Requirements

1. **Implement `drop_conv3x3_block`**:
   - Return an `nn.Sequential` block containing a valid chain of Conv2d, BatchNorm, Activations (e.g., ReLU/SiLU), and Dropout.

2. **Implement `Net.__init__`**:
   - select `self.pattern from "{available_patterns}"` 
   - available_backbones is [{available_backbones}]
   - Initialize `self.backbone_a` and `self.backbone_b` from `available_backbones` using the `TorchVision` wrapper, Use specific model names and correct `in_channels` as the parameter.
   - Initialize `self.features` as an `nn.Sequential` of `FractalUnit` modules with growing channels, use only one or two layers of FractalUnit to avoid excessively small output.
   - Call `self.infer_dimensions_dynamically(in_shape, out_shape[0])` and initialize `self._scaler`.

3. **Implement `Net.forward` (CRITICAL)**:
   - Implement the forward pass based on the selected `self.pattern`
   - **Dimension Safety**: If the topology feeds the output of a Backbone/Vector into `self.features` (Fractal), you MUST reshape the vector to 4D (`unsqueeze`) and upscale it using `torch.nn.functional.interpolate` to at least `(14, 14)` before passing it to the fractal network.
   - Example safety check for Vector inputs:
     `if tensor.dim() == 2: tensor = tensor.unsqueeze(-1).unsqueeze(-1)`
     `if tensor.shape[-1] < 14: tensor = F.interpolate(tensor, size=(14,14))`

Output Instructions:
You must output ONLY the implementation of the three missing components, wrapped in the following XML tags:
1. <block> ... code for drop_conv3x3_block ... </block>
2. <init> ... code for Net.__init__ ... </init>
3. <forward> ... code for Net.forward ... </forward>

Do not output the full file. Do not output any markdown formatting (like ```python).
"""

def parse_nn_code(code_str):
    try:
        tree = ast.parse(code_str)
        lines = code_str.splitlines()

        block_code = None
        init_code = None
        forward_code = None

        def get_source(node):
            return ast.get_source_segment(code_str, node)

        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == 'drop_conv3x3_block':
                block_code = get_source(node)

            elif isinstance(node, ast.ClassDef) and node.name == 'Net':
                for sub_node in node.body:
                    if isinstance(sub_node, ast.FunctionDef):
                        if sub_node.name == '__init__':
                            init_code = get_source(sub_node)
                        elif sub_node.name == 'forward':
                            forward_code = get_source(sub_node)

        def clean_code(c):
            return textwrap.dedent(c).strip() if c else None

        return clean_code(block_code), clean_code(init_code), clean_code(forward_code)

    except Exception as e:
        print(f"AST Parsing Failed: {e}")
        print(f"Code snippet: {code_str[:100]}...")
        return None, None, None
