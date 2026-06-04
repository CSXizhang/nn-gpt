import ast
import hashlib
import re
from collections import Counter
from pathlib import Path

root = Path("/home/s471802/nn-gpt/out/nngpt/llm/epoch_1pattern_aligned_20260520_2324_h100x8_nofti_rebuilddb")


def norm(text):
    return re.sub(r"\s+", " ", re.sub(r"#.*", "", text)).strip()


def method_source(text, method_name):
    try:
        tree = ast.parse(text)
    except Exception:
        return ""
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Net":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return ast.get_source_segment(text, item) or ""
    return ""


for cycle in [7, 9, 14, 19]:
    synth = root / f"A{cycle}" / "synth_nn"
    dirs = sorted([p for p in synth.glob("B*") if p.is_dir()], key=lambda p: int(p.name[1:]))
    counters = {
        "forward_all": Counter(),
        "forward_success": Counter(),
        "init_success": Counter(),
        "backbone_pair_success": Counter(),
        "classifier_success": Counter(),
    }
    examples = {}
    for d in dirs:
        py = d / "new_nn.py"
        if not py.exists():
            continue
        text = py.read_text(errors="ignore")
        fwd = method_source(text, "forward")
        init = method_source(text, "__init__")
        fhash = hashlib.md5(norm(fwd).encode()).hexdigest()
        counters["forward_all"][fhash] += 1
        examples.setdefault(fhash, (d.name, fwd[:700]))
        ok = (d / "1.json").exists()
        if not ok:
            continue
        counters["forward_success"][fhash] += 1
        counters["init_success"][hashlib.md5(norm(init).encode()).hexdigest()] += 1
        tv = re.findall(r"self\.backbone_[ab]\s*=\s*TorchVision\((.*?)\)", init)
        names = []
        for item in tv:
            q = re.search(r"""['"]([A-Za-z0-9_]+)['"]""", item)
            names.append(q.group(1) if q else item[:40])
        counters["backbone_pair_success"][tuple(names)] += 1
        cls = re.findall(r"self\.classifier\s*=\s*([^\n]+)", init)
        counters["classifier_success"][cls[-1].strip() if cls else "<none>"] += 1
    print(f"\nA{cycle}")
    for k, c in counters.items():
        print(k, "unique", len(c), "top", c.most_common(8))
    print("forward_examples")
    for h, n in counters["forward_all"].most_common(3):
        print("hash", h, "count", n, "from", examples[h][0])
        print(examples[h][1].replace("\n", "\\n")[:700])
