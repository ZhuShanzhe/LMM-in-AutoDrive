"""Fuse validated Qwen/MiniCPM scene outputs with matching CARLA captures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scene_understanding.core.visual_semantic_fusion import fuse_visual_semantics


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            yield record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-index", type=Path, required=True)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--min-iou", type=float, default=0.05)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    args = parser.parse_args()

    captures = {str(record["frame_id"]): record for record in _read_jsonl(args.capture_index)}
    if args.output.exists() or args.audit_output.exists():
        raise SystemExit("output paths must not already exist")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)

    accepted = skipped = failed = 0
    with args.output.open("x", encoding="utf-8") as worlds, args.audit_output.open(
        "x", encoding="utf-8"
    ) as audits:
        for inference in _read_jsonl(args.inference):
            frame_id = str(inference.get("frame_id", ""))
            capture = captures.get(frame_id)
            if inference.get("status") != "valid" or capture is None:
                skipped += 1
                continue
            try:
                world_state = json.loads(Path(capture["world_state_path"]).read_text(encoding="utf-8"))
                projection = json.loads(Path(capture["projection_path"]).read_text(encoding="utf-8"))
                enriched, audit = fuse_visual_semantics(
                    world_state,
                    inference,
                    projection,
                    min_iou=args.min_iou,
                    min_confidence=args.min_confidence,
                )
                worlds.write(json.dumps(enriched, ensure_ascii=False) + "\n")
                audits.write(json.dumps(audit, ensure_ascii=False) + "\n")
                accepted += 1
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                audits.write(json.dumps({
                    "frame_id": frame_id,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }, ensure_ascii=False) + "\n")
                failed += 1
    print(json.dumps({"accepted": accepted, "skipped": skipped, "failed": failed}))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
