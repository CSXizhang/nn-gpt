#!/usr/bin/env python3
"""Build the branch/release manifest from the verified local staging set."""

import csv
import hashlib
import re
import subprocess
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
with (ROOT / "inventory/archived_run_coverage.tsv").open() as handle:
    coverage = {row["run_id"]: row for row in csv.DictReader(handle, delimiter="\t")}
rows = []
branch_checksums = []
published_sources = set()

tracked = subprocess.check_output(["git", "-C", str(ROOT), "ls-files", "-z"]).decode().split("\0")
for rel in sorted(filter(None, tracked)):
    path = ROOT / rel
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
    elif rel.startswith("experiments/workstation-prototype/"):
        source_path = ("/shared/ssd/home/b-x-0522/nn-gpt_exp/"
                       + rel.removeprefix("experiments/workstation-prototype/"))
    row = {field: old.get(field, "") for field in FIELDS}
    row.update(category=rel.split("/", 1)[0], source_host=host, source_path=source_path,
               archive_path="release/" + asset["asset_name"],
               size=asset["bytes"], sha256=asset["sha256"],
               description=asset["description"], status="published_release",
               notes="archive mapping: " + rel)
    if "retained_rl" in Path(rel).parts:
        run = Path(rel).parts[Path(rel).parts.index("retained_rl") + 1]
        row["experiment"] = run
        seed = re.search(r"seed(\d+)", run)
        row["seed"] = seed.group(1) if seed else ""
        row["job_id"] = coverage.get(run, {}).get("job_ids", "")
        row["code_commit"] = coverage.get(run, {}).get("code_commit", "")
    rows.append(row)
    if source_path:
        published_sources.add(source_path)

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
    if old["source_path"] in published_sources:
        continue
    rows.append(old)

for rel in (
    "git/nn-dataset/nn-dataset-all.bundle",
    "git/nn-dataset/julia2-nn-dataset-all.bundle",
    "git/workstation/nn-dataset/nn-dataset-all.bundle",
):
    old = by_path[rel]
    row = dict(old)
    row.update(archive_path="", sha256="", status="excluded_secret_history",
               notes="Credential-like pattern in Git history; bundle withheld from public release")
    rows.append(row)

with (ROOT / "MANIFEST.tsv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

branch_checksums.append((digest(ROOT / "MANIFEST.tsv"), "MANIFEST.tsv"))
with (ROOT / "CHECKSUMS.sha256").open("w") as handle:
    for sha, rel in branch_checksums:
        handle.write(f"{sha}  {rel}\n")

print(f"branch_files={len(branch_checksums)} release_assets={len(assets)} manifest_rows={len(rows)}")
