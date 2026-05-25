#!/usr/bin/env python3
"""Rebuild the active nn-dataset SQLite DB and verify exact prefix counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _unlink_db_family(db_file: Path) -> list[str]:
    removed: list[str] = []
    for path in (db_file, Path(str(db_file) + "-wal"), Path(str(db_file) + "-shm")):
        if path.exists():
            path.unlink()
            removed.append(str(path))
    return removed


def _count_files(repo_root: Path, prefix: str) -> int:
    return sum(1 for _ in (repo_root / "ab/nn/nn").glob(f"{prefix}-*.py"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default="/home/s471802/nn-dataset")
    parser.add_argument("--prefix", action="append", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    from ab.nn.util.Const import db_file

    removed = _unlink_db_family(Path(db_file))

    # Importing Write after removing the DB triggers the normal source-backed population path.
    import ab.nn.util.db.Write  # noqa: F401
    import ab.nn.api as api

    repo_root = Path(args.repo_root).expanduser().resolve()
    counts: dict[str, dict[str, int]] = {}
    for prefix in args.prefix:
        df = api.data(task="img-classification", nn_prefixes=(prefix,), unique_nn=True)
        counts[prefix] = {
            "file_count": _count_files(repo_root, prefix),
            "api_unique_count": int(len(df)),
        }

    report = {
        "db_file": str(db_file),
        "removed": removed,
        "counts": counts,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
