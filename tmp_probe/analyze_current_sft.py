import ast
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

root = Path("/home/s471802/nn-gpt/out/nngpt/llm/epoch_1pattern_aligned_20260520_2324_h100x8_nofti_rebuilddb")
cycles = [7, 9, 14, 19]


def norm(text: str) -> str:
    text = re.sub(r"#.*", "", text)
    return re.sub(r"\s+", " ", text).strip()


def net_text(text: str) -> str:
    idx = text.find("class Net")
    return text[idx:] if idx >= 0 else text


def forward_text(text: str) -> str:
    match = re.search(r"class Net\(nn\.Module\):([\s\S]*)", text)
    body = match.group(1) if match else text
    match = re.search(r"\n    def forward\([^\n]*\):\n([\s\S]*?)(?=\n    def |\nclass |\Z)", body)
    return match.group(0) if match else ""


def accuracy_of(run_dir: Path):
    path = run_dir / "1.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        row = data[0] if isinstance(data, list) else data
        return float(row.get("accuracy", 0.0)) * 100.0
    except Exception:
        return None


def error_kind(run_dir: Path) -> str:
    path = run_dir / "error.txt"
    if not path.exists():
        return "no_artifact"
    text = path.read_text(errors="ignore")
    if "NN already exists" in text:
        return "duplicate"
    if "Code is missing" in text:
        return "missing_code"
    if "expected input" in text and "channels" in text:
        return "channel_mismatch"
    if "Sizes of tensors" in text or "stack expects" in text or "same number of dimensions" in text:
        return "shape_mismatch"
    if "mat1 and mat2" in text:
        return "linear_shape"
    if "Accuracy is too low" in text:
        return "low_acc"
    if "CUDA out of memory" in text:
        return "oom"
    return text.splitlines()[0][:80] if text.splitlines() else "unknown"


for cycle in cycles:
    synth = root / f"A{cycle}" / "synth_nn"
    run_dirs = sorted([p for p in synth.glob("B*") if p.is_dir()], key=lambda p: int(p.name[1:]))
    net_hash = Counter()
    forward_hash = Counter()
    backbones = Counter()
    patterns = Counter()
    errors = Counter()
    ast_bad = []
    random_select = 0
    feature_calls = 0
    accuracies = []
    success = 0

    for run_dir in run_dirs:
        if (run_dir / "1.json").exists():
            success += 1
        else:
            errors[error_kind(run_dir)] += 1

        accuracy = accuracy_of(run_dir)
        if accuracy is not None:
            accuracies.append(accuracy)

        code_path = run_dir / "new_nn.py"
        if not code_path.exists():
            continue
        text = code_path.read_text(errors="ignore")
        net = net_text(text)
        forward = forward_text(text)
        net_hash[hashlib.md5(norm(net).encode()).hexdigest()] += 1
        forward_hash[hashlib.md5(norm(forward).encode()).hexdigest()] += 1

        if re.search(r"random\.choice|np\.random|np\.random\.randint|torch\.rand", net):
            random_select += 1
        if re.search(r"self\._feature_to_input_image\(", net):
            feature_calls += 1
        try:
            ast.parse(text)
        except Exception as exc:
            ast_bad.append((run_dir.name, type(exc).__name__, str(exc).splitlines()[0]))

        for match in re.finditer(r"TorchVision\(([^\n)]*)\)", net):
            quoted = re.search(r"""['"]([a-zA-Z0-9_]+)['"]""", match.group(1))
            if quoted:
                backbones[quoted.group(1)] += 1
        pattern_match = re.search(r"""self\.pattern\s*=\s*['"]([^'"]+)""", net)
        if pattern_match:
            patterns[pattern_match.group(1)] += 1

    print(f"\nA{cycle}")
    print(
        "success", success, "/", len(run_dirs),
        "dup", errors.get("duplicate", 0),
        "best", round(max(accuracies), 2) if accuracies else None,
        "mean", round(sum(accuracies) / len(accuracies), 2) if accuracies else None,
    )
    print("errors", dict(errors.most_common()))
    print("ast_bad", len(ast_bad), ast_bad[:5])
    print("random_select", random_select, "feature_to_input_calls", feature_calls)
    print("unique_net", len(net_hash), "top_net_repeats", net_hash.most_common(3))
    print("unique_forward", len(forward_hash), "top_forward_repeats", forward_hash.most_common(3))
    print("patterns", dict(patterns.most_common()))
    print("top_backbone_literals", backbones.most_common(10))
