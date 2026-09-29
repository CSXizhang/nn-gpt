#!/usr/bin/env python3
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "remote_evidence"
REMOTE_PREFIX = "/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    stats = {}
    for line in (EVIDENCE / "remote_stat_records.txt").read_text().splitlines():
        path, size, epoch, mtime = line.split("|", 3)
        stats[path] = {"size": int(size), "mtime_epoch": int(epoch), "mtime": mtime}

    mappings = {
        EVIDENCE / "wide38" / "proxy_manifest_wide38.jsonl": REMOTE_PREFIX + "parallel_runs/20260727_proxy_wide38_unfrozen20_standard/proxy_manifest_wide38.jsonl",
        EVIDENCE / "wide38" / "00_4pattern_seed42_743.json": REMOTE_PREFIX + "parallel_runs/20260727_proxy_wide38_unfrozen20_standard/results/00_4pattern_seed42_743.json",
        EVIDENCE / "wide38" / "proxy-wide38-smoke-2968490_0.out": REMOTE_PREFIX + "parallel_runs/20260727_proxy_wide38_unfrozen20_standard/slurm/proxy-wide38-smoke-2968490_0.out",
        EVIDENCE / "wide38" / "proxy-wide38-smoke-2968490_0.err": REMOTE_PREFIX + "parallel_runs/20260727_proxy_wide38_unfrozen20_standard/slurm/proxy-wide38-smoke-2968490_0.err",
    }
    provenance_root = EVIDENCE / "provenance"
    for local in provenance_root.rglob("run_state.json"):
        remote = "/" + str(local.relative_to(provenance_root))
        mappings[local] = remote
    imagenette_root = EVIDENCE / "imagenette"
    imagenette_remote = REMOTE_PREFIX + "parallel_runs/20260605_0925_dscoder_imagenette_h100/"
    mappings[imagenette_root / "generation_samples.jsonl"] = imagenette_remote + "rl_output/generation_samples.jsonl"
    extracted_base = imagenette_root / "data" / "42-julia-hpc-ai-cv-students" / "s471802" / "nn-gpt-runs" / "parallel_runs" / "20260605_0925_dscoder_imagenette_h100"
    mappings[extracted_base / "run_state.json"] = imagenette_remote + "run_state.json"
    mappings[extracted_base / "rl_output" / "run_config.json"] = imagenette_remote + "rl_output/run_config.json"

    fetched = datetime.now(timezone.utc).isoformat()
    files = []
    for local, remote in sorted(mappings.items(), key=lambda x: x[1]):
        entry = {
            "host": "julia2.hpc.uni-wuerzburg.de",
            "remote_path": remote,
            "local_path": str(local.relative_to(ROOT)),
            "sha256": sha256(local),
            "size": local.stat().st_size,
            "fetched_at": fetched,
        }
        entry.update({f"remote_{k}": v for k, v in stats.get(remote, {}).items()})
        files.append(entry)
    result = {
        "generated_at": fetched,
        "host": "julia2.hpc.uni-wuerzburg.de",
        "search_roots": ["/home/s471802/nn-gpt", "/home/s471802/nn-dataset", REMOTE_PREFIX.rstrip("/")],
        "files": files,
        "generated_evidence": [
            "remote_evidence/wide38/sacct_2968450_2968490_2968494_raw.txt",
            "remote_evidence/wide38/gpu_task0_recomputed.json",
            "remote_evidence/imagenette_and_nndataset_raw.txt",
        ],
    }
    (ROOT / "remote_evidence_manifest.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
