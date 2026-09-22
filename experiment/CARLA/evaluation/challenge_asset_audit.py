"""Audit every pinned runtime asset without importing or loading any model."""

import argparse
import hashlib
import json
from pathlib import Path


def audit(manifest, roots, parser_dir=None):
    if manifest.get("schema_version") != "challenge_runtime_assets/1.0":
        raise ValueError("unsupported challenge asset manifest")
    roots = [Path(root).resolve() for root in roots]
    rows = []
    for item in manifest["files"]:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe asset path")
        candidates = [root / relative for root in roots]
        if parser_dir is not None and relative.parts[0] == "modernbert-drive-command-compositional":
            candidates.append(Path(parser_dir).resolve().joinpath(*relative.parts[1:]))
        checked = []
        match = None
        for path in dict.fromkeys(candidates):
            record = {"path": str(path), "status": "MISSING"}
            if path.is_file():
                try:
                    record["bytes"] = path.stat().st_size
                    digest = hashlib.sha256()
                    with path.open("rb") as handle:
                        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                            digest.update(chunk)
                    record["sha256"] = digest.hexdigest()
                    record["status"] = "VERIFIED" if record["bytes"] == item["bytes"] and record["sha256"] == item["sha256"] else "MISMATCH"
                    if record["status"] == "VERIFIED" and match is None:
                        match = str(path)
                except OSError as error:
                    record.update(status="UNREADABLE", error=str(error))
            checked.append(record)
        rows.append(dict(asset=item["path"], expected_bytes=item["bytes"], expected_sha256=item["sha256"],
                         status="VERIFIED" if match else "UNAVAILABLE", verified_path=match, candidates=checked))
    verified = sum(row["status"] == "VERIFIED" for row in rows)
    return dict(schema_version="challenge_asset_audit/1.0", version=manifest.get("version"),
                status="READY" if verified == len(rows) and rows else "NOT_READY",
                verified_count=verified, required_count=len(rows), files=rows,
                scope="file identity only; no model loading or driving validation")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[3]
    parser.add_argument("--manifest", type=Path, default=repo / "lightweight_vla_adapter/configs/challenge_assets.json")
    parser.add_argument("--model-root", action="append", type=Path, required=True)
    parser.add_argument("--parser-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = audit(json.loads(args.manifest.read_text(encoding="utf-8")), args.model_root, args.parser_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({key: value for key, value in report.items() if key != "files"}, ensure_ascii=False))
    return 0 if report["status"] == "READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
