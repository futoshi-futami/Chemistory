#!/usr/bin/env python3
"""Materialize data folders from Git-friendly archive parts when necessary."""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVES = {
    "gpr_handoff_data.zip": (ROOT / "data" / "gpr_handoff", "01_base_summary_first_angle.csv"),
    "dist_auto_data.zip": (ROOT / "data" / "dist_auto", "response.csv"),
}


def materialize(name: str, target: Path, marker: str) -> None:
    if (target / marker).exists():
        print(f"ready: {target.relative_to(ROOT)}")
        return
    archive_dir = ROOT / "data_archives"
    complete = archive_dir / name
    parts = sorted(archive_dir.glob(f"{name}.part-*"))
    target.mkdir(parents=True, exist_ok=True)
    if complete.exists():
        archive_path = complete
        temporary = False
    elif parts:
        handle = tempfile.NamedTemporaryFile(prefix=name, suffix=".zip", delete=False)
        with handle:
            for part in parts:
                handle.write(part.read_bytes())
        archive_path = Path(handle.name)
        temporary = True
    else:
        raise FileNotFoundError(f"Neither {complete} nor archive parts were found")

    try:
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(target)
    finally:
        if temporary:
            archive_path.unlink(missing_ok=True)
    if not (target / marker).exists():
        raise RuntimeError(f"Archive {name} did not create expected marker {marker}")
    print(f"extracted: {target.relative_to(ROOT)}")


def main() -> None:
    for name, (target, marker) in ARCHIVES.items():
        materialize(name, target, marker)


if __name__ == "__main__":
    main()
