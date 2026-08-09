#!/usr/bin/env python3
"""Validate staging artifacts and build the official submission ZIP."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import zipfile


TITLE = "面向智能驾驶的大模型应用场景研究-南京大学"
TECHNICAL_PLAN = f"{TITLE}_技术方案.pdf"
REQUIRED_FILES = ("image.tar", "README.md", TECHNICAL_PLAN, "metrics.zip")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(staging: Path, weights_in_image: bool) -> list[Path]:
    missing = [name for name in REQUIRED_FILES if not (staging / name).is_file()]
    if missing:
        raise SystemExit(f"missing required artifacts: {', '.join(missing)}")
    if (staging / "image.tar").stat().st_size < 1024 * 1024:
        raise SystemExit("image.tar is too small; placeholder files are forbidden")
    if (staging / TECHNICAL_PLAN).read_bytes()[:5] != b"%PDF-":
        raise SystemExit("technical plan is not a valid PDF")
    with zipfile.ZipFile(staging / "metrics.zip") as archive:
        bad = archive.testzip()
        if bad:
            raise SystemExit(f"metrics.zip CRC failure: {bad}")
        names = set(archive.namelist())
        for scene in ("scene1/", "scene2/", "scene3/"):
            if not any(name.startswith(scene) for name in names):
                raise SystemExit(f"metrics.zip lacks {scene}")
    weights = staging / "weights"
    if not weights_in_image:
        files = [path for path in weights.rglob("*") if path.is_file()]
        files = [path for path in files if path.name != "README.md"]
        if not files:
            raise SystemExit("external weights selected but weights/ has no model files")
    output_files = [staging / name for name in REQUIRED_FILES]
    if not weights_in_image:
        output_files.extend(path for path in weights.rglob("*") if path.is_file())
    return output_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--staging",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--weights-in-image", action="store_true")
    args = parser.parse_args()
    staging = args.staging.resolve()
    output = staging.parent / f"{TITLE}.zip"
    files = validate(staging, args.weights_in_image)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(staging).as_posix())
    sidecar = output.with_suffix(output.suffix + ".sha256")
    sidecar.write_text(f"{sha256(output)}  {output.name}\n", encoding="utf-8")
    print(output)
    print(sidecar)


if __name__ == "__main__":
    main()
