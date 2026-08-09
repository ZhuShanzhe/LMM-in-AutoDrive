"""Download an explicit, licensed Waymo v2 modular perception subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gcloud-bin", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    spec = json.loads(args.config.read_text(encoding="utf-8"))
    gcloud = (
        str(args.gcloud_bin.resolve())
        if args.gcloud_bin is not None
        else shutil.which("gcloud")
    )
    if not gcloud:
        raise FileNotFoundError("gcloud was not found; pass --gcloud-bin")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    files = []
    for split, segments in spec["splits"].items():
        for component in spec["components"]:
            component_dir = output / split / component
            component_dir.mkdir(parents=True, exist_ok=True)
            for segment in segments:
                filename = f"{segment}.parquet"
                destination = component_dir / filename
                uri = (
                    f"gs://{spec['bucket']}/{split}/{component}/{filename}"
                )
                if not destination.is_file() or destination.stat().st_size == 0:
                    subprocess.run(
                        [gcloud, "storage", "cp", uri, str(destination)],
                        check=True,
                    )
                files.append(
                    {
                        "split": split,
                        "component": component,
                        "segment": segment,
                        "path": destination.relative_to(output).as_posix(),
                        "bytes": destination.stat().st_size,
                        "sha256": sha256(destination),
                    }
                )
    manifest = {
        "schema_version": "waymo_v2_subset_download/1.0",
        "source_config": args.config.name,
        "bucket": spec["bucket"],
        "files": files,
        "total_bytes": sum(item["bytes"] for item in files),
        "license": (
            "Downloaded with a user-authorized Waymo Open Dataset account; "
            "redistribution remains subject to the Waymo license."
        ),
    }
    (output / "download_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
