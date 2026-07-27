from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify configured offline teacher assets")
    parser.add_argument(
        "--config",
        default="lightweight_vla_adapter/configs/teachers.json",
    )
    parser.add_argument(
        "--hash",
        action="store_true",
        help="Compute SHA256 for each checkpoint; this reads every weight byte.",
    )
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    results = []
    missing = False
    for name, spec in config["teachers"].items():
        path = Path(spec["local_checkpoint"])
        exists = path.is_file()
        missing |= not exists
        record = {
            "teacher": name,
            "exists": exists,
            "path": str(path),
            "size_bytes": path.stat().st_size if exists else None,
        }
        if exists and args.hash:
            record["sha256"] = sha256(path)
        results.append(record)

    print(json.dumps(results, indent=2))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
