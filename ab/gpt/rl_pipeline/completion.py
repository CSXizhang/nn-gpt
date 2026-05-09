"""Shared completion parsing helpers for TuneRL pipelines."""

from __future__ import annotations

import hashlib
import re
import textwrap
from typing import Dict, List, Sequence, Tuple

REQUIRED_BACKBONE_NAMES = ("backbone_a", "backbone_b")
BLOCK_SIGNATURE = "def drop_conv3x3_block(in_channels, out_channels, stride=1, padding=1, bias=False, dropout_prob=0.0):"
INIT_SIGNATURE = "def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:"
FORWARD_SIGNATURE = "def forward(self, x: torch.Tensor, is_probing: bool = False) -> torch.Tensor:"
FORWARD_SIGNATURE_ALIASES = (
    "def forward(self, x: torch.Tensor, is_probing: bool=False) -> torch.Tensor:",
)

_BLOCKED_ATTRS = {
    "device",
    "use_amp",
    "_input_spec",
    "pattern",
    "classifier",
    "infer_dimensions_dynamically",
    "train_setup",
    "learn",
    "criterion",
    "optimizer",
    "_scaler",
}

_EXTRACTION_META_CACHE: Dict[str, Dict[str, object]] = {}


def clear_extraction_meta_cache() -> None:
    _EXTRACTION_META_CACHE.clear()


def _strip_outer_code_fences(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"^```(?:python)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _clean_block(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"^```python\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_xml_tag(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.IGNORECASE | re.DOTALL)
    return _clean_block(match.group(1)) if match else ""


def _dedupe_keep_order(items: Sequence[str]) -> List[str]:
    deduped: List[str] = []
    seen = set()
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _completion_cache_key(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


def _infer_attr_role(attr_name: str) -> str:
    lowered = attr_name.lower()
    if "fractal" in lowered:
        return "fractal"
    if lowered.startswith("backbone"):
        return "backbone"
    if "stem" in lowered:
        return "stem"
    if any(token in lowered for token in ("project", "bridge", "adapter", "align")):
        return "project"
    if any(token in lowered for token in ("fuse", "merge", "gate", "mixer")):
        return "fuse"
    return "generic"


def _has_structural_attr(attrs: Sequence[str]) -> bool:
    return any(
        _infer_attr_role(attr) in {"stem", "project", "fuse", "backbone", "fractal"}
        for attr in attrs
    )


def _scan_raw_attrs(*texts: str) -> List[str]:
    attrs: List[str] = []
    for text in texts:
        if not text:
            continue
        for attr in re.findall(r"self\.([A-Za-z_]\w*)\s*(?:\(|=)", text):
            if attr in _BLOCKED_ATTRS or attr.startswith("__"):
                continue
            attrs.append(attr)
    return _dedupe_keep_order(attrs)


def _accept_exact_function(
    code: str,
    signature: str,
    *,
    aliases: Sequence[str] = (),
) -> str:
    code = _strip_outer_code_fences(code)
    code = textwrap.dedent(code).strip()
    if code.startswith(signature):
        return code
    for alias in aliases:
        if code.startswith(alias):
            return signature + code[len(alias):]
    if not code.startswith(signature):
        return ""
    return code


def _extract_defined_backbones(init_code: str) -> List[str]:
    return _dedupe_keep_order(re.findall(r"self\.(backbone_[A-Za-z]\w*)\s*=", init_code or ""))


def _extract_used_backbones(forward_code: str) -> List[str]:
    return _dedupe_keep_order(re.findall(r"self\.(backbone_[A-Za-z]\w*)\b", forward_code or ""))


def _extract_backbone_model_names(init_code: str) -> List[str]:
    matches: Dict[str, str] = {}
    patterns = (
        r"self\.(backbone_[ab])\s*=\s*TorchVision\(\s*model\s*=\s*['\"]([^'\"]+)['\"]",
        r"self\.(backbone_[ab])\s*=\s*TorchVision\(\s*['\"]([^'\"]+)['\"]",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, init_code or ""):
            matches.setdefault(match.group(1), match.group(2))
    return [matches[name] for name in REQUIRED_BACKBONE_NAMES if name in matches]


def _count_xml_tags(text: str, tag: str) -> Tuple[int, int]:
    return (
        len(re.findall(rf"<{tag}>", text, re.IGNORECASE)),
        len(re.findall(rf"</{tag}>", text, re.IGNORECASE)),
    )


def _build_extraction_meta(
    completion: str,
    candidate: str,
    block_code: str,
    init_code: str,
    forward_code: str,
) -> Dict[str, object]:
    xml_tag_count = sum(bool(code) for code in (block_code, init_code, forward_code))
    xml_counts = {tag: _count_xml_tags(candidate, tag) for tag in ("block", "init", "forward")}
    class_count = len(re.findall(r"^\s*class\s+\w+", candidate, re.MULTILINE))
    import_count = len(re.findall(r"^\s*(?:from|import)\s+\w+", candidate, re.MULTILINE))
    bad_signature_count = len(re.findall(r"\)\s*-\s*:", candidate))
    raw_attrs = _scan_raw_attrs(candidate, block_code, init_code, forward_code)
    structural_attr_detected = _has_structural_attr(raw_attrs)

    defined_backbones = _extract_defined_backbones(init_code)
    used_backbones = _extract_used_backbones(forward_code)
    backbone_model_names = _extract_backbone_model_names(init_code)
    required_backbone_set = set(REQUIRED_BACKBONE_NAMES)
    dual_backbone_init_ok = set(defined_backbones) == required_backbone_set and len(defined_backbones) == 2
    dual_backbone_forward_ok = required_backbone_set.issubset(set(used_backbones)) and len(set(used_backbones)) == 2
    dual_backbone_ok = dual_backbone_init_ok and dual_backbone_forward_ok

    exact_xml = all(start_count == 1 and end_count == 1 for start_count, end_count in xml_counts.values())
    exact_signatures = {
        "block": block_code.startswith(BLOCK_SIGNATURE),
        "init": init_code.startswith(INIT_SIGNATURE),
        "forward": forward_code.startswith(FORWARD_SIGNATURE)
        or any(forward_code.startswith(alias) for alias in FORWARD_SIGNATURE_ALIASES),
    }

    quality_score = 0
    quality_score += 2 if exact_xml else 0
    quality_score += sum(1 for ok in exact_signatures.values() if ok)
    quality_score += 2 if dual_backbone_ok else 0
    quality_score += 1 if structural_attr_detected else 0
    quality_score -= min(class_count, 2)
    quality_score -= min(import_count, 2)
    quality_score -= min(bad_signature_count, 2)

    return {
        "xml_tag_count": xml_tag_count,
        "xml_tag_exact": exact_xml,
        "xml_counts": xml_counts,
        "class_count": class_count,
        "import_count": import_count,
        "bad_signature_count": bad_signature_count,
        "structural_attr_detected": structural_attr_detected,
        "quality_score": quality_score,
        "exact_block_signature": exact_signatures["block"],
        "exact_init_signature": exact_signatures["init"],
        "exact_forward_signature": exact_signatures["forward"],
        "defined_backbones": defined_backbones,
        "used_backbones": used_backbones,
        "backbone_model_names": backbone_model_names,
        "dual_backbone_init_ok": dual_backbone_init_ok,
        "dual_backbone_forward_ok": dual_backbone_forward_ok,
        "dual_backbone_ok": dual_backbone_ok,
        "candidate_line_count": len(candidate.splitlines()),
    }


def extract_completion_payload_strict(completion: str) -> Tuple[Tuple[str, str, str], Dict[str, object]]:
    cache_key = _completion_cache_key(completion or "")
    cached = _EXTRACTION_META_CACHE.get(cache_key)
    if cached:
        return (
            (cached["block_code"], cached["init_code"], cached["forward_code"]),
            dict(cached["meta"]),
        )

    candidate = _strip_outer_code_fences(completion or "").lstrip()
    raw_block_code = _extract_xml_tag(candidate, "block")
    raw_init_code = _extract_xml_tag(candidate, "init")
    raw_forward_code = _extract_xml_tag(candidate, "forward")
    meta = _build_extraction_meta(
        completion or "",
        candidate,
        raw_block_code,
        raw_init_code,
        raw_forward_code,
    )
    if meta.get("xml_tag_exact"):
        block_code = _accept_exact_function(raw_block_code, BLOCK_SIGNATURE)
        init_code = _accept_exact_function(raw_init_code, INIT_SIGNATURE)
        forward_code = _accept_exact_function(
            raw_forward_code,
            FORWARD_SIGNATURE,
            aliases=FORWARD_SIGNATURE_ALIASES,
        )
    else:
        block_code = ""
        init_code = ""
        forward_code = ""

    _EXTRACTION_META_CACHE[cache_key] = {
        "block_code": block_code,
        "init_code": init_code,
        "forward_code": forward_code,
        "meta": meta,
    }
    return ((block_code, init_code, forward_code), meta)


def extract_completion_blocks_strict(completion: str) -> Tuple[str, str, str]:
    blocks, _ = extract_completion_payload_strict(completion)
    return blocks


def extract_completion_meta(completion: str) -> Dict[str, object]:
    _, meta = extract_completion_payload_strict(completion)
    return meta
