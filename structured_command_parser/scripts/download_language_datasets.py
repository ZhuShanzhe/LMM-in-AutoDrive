from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import sleep

from huggingface_hub import HfApi, hf_hub_download


MODULE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = MODULE_ROOT / "data" / "corpus" / "raw" / "simlingo"
REPO_ID = "RenzKa/simlingo"
PREFIXES = {
    "dreamer_": "dreamer_archives",
    "commentary_": "commentary_archives",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    remote: list[dict[str, object]] = []
    for entry in api.list_repo_tree(REPO_ID, repo_type="dataset", recursive=True):
        path = getattr(entry, "path", "")
        for prefix, directory in PREFIXES.items():
            if path.startswith(prefix) and path.endswith(".tar.gz"):
                remote.append(
                    {
                        "filename": path,
                        "size": int(getattr(entry, "size", 0) or 0),
                        "directory": directory,
                    }
                )
                break

    for metadata_name in ("README.md", "LICENSE"):
        hf_hub_download(
            REPO_ID,
            metadata_name,
            repo_type="dataset",
            local_dir=output / "metadata",
        )

    def download(row: dict[str, object]) -> dict[str, object]:
        filename = str(row["filename"])
        target_dir = output / str(row["directory"])
        target = target_dir / filename
        expected_size = int(row["size"])
        if target.is_file() and target.stat().st_size == expected_size:
            return {**row, "status": "REUSED", "local_path": str(target)}
        target_dir.mkdir(parents=True, exist_ok=True)
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                local_path = Path(
                    hf_hub_download(
                        REPO_ID,
                        filename,
                        repo_type="dataset",
                        local_dir=target_dir,
                    )
                )
                if local_path.stat().st_size != expected_size:
                    raise RuntimeError(
                        f"size mismatch for {filename}: "
                        f"{local_path.stat().st_size} != {expected_size}"
                    )
                return {**row, "status": "DOWNLOADED", "local_path": str(local_path)}
            except Exception as error:  # network retries are intentionally broad
                last_error = error
                sleep(attempt * 3)
        raise RuntimeError(f"failed to download {filename}") from last_error

    completed: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download, row): row for row in remote}
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            completed.append(result)
            print(
                f"[{index}/{len(remote)}] {result['status']} "
                f"{result['filename']}"
            )

    completed.sort(key=lambda row: str(row["filename"]))
    manifest = {
        "schema": "simlingo-language-download-v1",
        "repo_id": REPO_ID,
        "file_count": len(completed),
        "total_bytes": sum(int(row["size"]) for row in completed),
        "groups": {
            directory: {
                "files": sum(row["directory"] == directory for row in completed),
                "bytes": sum(
                    int(row["size"])
                    for row in completed
                    if row["directory"] == directory
                ),
            }
            for directory in PREFIXES.values()
        },
        "files": completed,
    }
    manifest_path = output.parent.parent / "manifests" / "simlingo_download.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["groups"], ensure_ascii=False, indent=2))
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
