"""Zip the code (no data, no venv) for notebooks/colab_himawari.ipynb.

    python scripts/package_for_colab.py      ->  dist/stormcast-code.zip  (upload to MyDrive/stormcast/)
"""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["pyproject.toml", "README.md", "configs", "src"]


def main() -> None:
    out = ROOT / "dist" / "stormcast-code.zip"
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for item in INCLUDE:
            path = ROOT / item
            files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
            for f in files:
                if "__pycache__" in f.parts or f.suffix == ".pyc" or ".egg-info" in str(f):
                    continue
                z.write(f, f.relative_to(ROOT).as_posix())
    print(f"{out} ({out.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
