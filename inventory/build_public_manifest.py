#!/usr/bin/env python3
"""Build the branch/release manifest from the verified local staging set."""

import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT.with_name("nn-gpt-2026-09-29")
FIELDS = (
    "category", "source_host", "source_path", "archive_path", "size", "sha256",
    "experiment", "seed", "job_id", "code_commit", "description", "status", "notes",
)


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


with (LOCAL / "MANIFEST.tsv").open() as handle:
    original = list(csv.DictReader(handle, delimiter="\t"))
by_path = {row["archive_path"]: row for row in original if row["archive_path"]}
assets = list(csv.DictReader((ROOT / "ASSETS.tsv").open(), delimiter="\t"))
asset_names = {row["asset_name"] for row in assets}
rows = []
branch_checksums = []

for path in sorted(ROOT.rglob("*")):
    if not path.is_file() or ".git" in path.parts or "release-assets" in path.parts:
        continue
    rel = path.relative_to(ROOT).as_posix()
    if rel in ("MANIFEST.tsv", "CHECKSUMS.sha256"):
        continue
    row = dict(by_path.get(rel, {}))
    row.update(category=rel.split("/", 1)[0], archive_path=rel,
               size=str(path.stat().st_size), sha256=digest(path), status="published_git")
    row.setdefault("source_host", "local")
    row.setdefault("source_path", "")
    row.setdefault("experiment", "")
    row.setdefault("seed", "")
    row.setdefault("job_id", "")
    row.setdefault("code_commit", "")
    row.setdefault("description", path.name)
    row.setdefault("notes", "")
    rows.append(row)
    branch_checksums.append((row["sha256"], rel))

for asset in assets:
    rel = asset["archive_path"]
    old = dict(by_path.get(rel, {}))
    host = ("julia2" if rel.startswith(("models/adapters/julia2/", "experiments/julia2-"))
            or asset["asset_name"].startswith("julia2-") else
            "workstation" if asset["asset_name"].startswith("workstation-") else
            old.get("source_host") or "local")
    source_path = old.get("source_path", "")
    if "retained_rl" in Path(rel).parts:
        parts = Path(rel).parts
        source_path = ("/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/"
                       + "/".join(parts[parts.index("retained_rl") + 1:]))
    elif asset["asset_name"] == "workstation-prototype-raw-generation_samples.jsonl":
        source_path = "/shared/ssd/home/b-x-0522/nn-gpt_exp/rl_output/raw/generation_samples.jsonl"
    row = {field: old.get(field, "") for field in FIELDS}
    row.update(category=rel.split("/", 1)[0], source_host=host, source_path=source_path,
               archive_path="release/" + asset["asset_name"],
               size=asset["bytes"], sha256=asset["sha256"],
               description=asset["description"], status="published_release",
               notes="local source: " + rel)
    rows.append(row)

for old in original:
    archive_path = old["archive_path"]
    if "::" not in archive_path:
        continue
    package, member = archive_path.split("::", 1)
    name = Path(package).name
    if name not in asset_names:
        continue
    row = dict(old)
    row["archive_path"] = "release/" + name + "::" + member
    row["status"] = "included_in_release"
    rows.append(row)

for old in original:
    if old["archive_path"] or old["status"] in ("archived", ""):
        continue
    rows.append(old)

with (ROOT / "MANIFEST.tsv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

branch_checksums.append((digest(ROOT / "MANIFEST.tsv"), "MANIFEST.tsv"))
with (ROOT / "CHECKSUMS.sha256").open("w") as handle:
    for sha, rel in branch_checksums:
        handle.write(f"{sha}  {rel}\n")

print(f"branch_files={len(branch_checksums)} release_assets={len(assets)} manifest_rows={len(rows)}")
