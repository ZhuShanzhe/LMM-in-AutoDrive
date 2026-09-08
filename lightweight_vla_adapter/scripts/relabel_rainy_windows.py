"""Relabel rainy-night event-window training rows to high risk (stage-9)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


SOURCE_CAPTURE = Path(
    "/root/autodl-tmp/LMM-in-AutoDrive/experiment/CARLA/outputs/"
    "scene3_cf_rainy_night_seed101_20260805"
)
STAGE8 = Path(
    "/root/autodl-tmp/datasets/training/"
    "universal_three_scene_v6_finetune_stage8"
)
STAGE9 = Path(
    "/root/autodl-tmp/datasets/training/"
    "universal_three_scene_v6_finetune_stage9"
)


def load_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    windows: list[tuple[int, int]] = []
    active_start: dict[str, int] = {}
    with (SOURCE_CAPTURE / "event_timeline.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            frame = int(row["simulation_frame"])
            if row["state"] == "ACTIVE":
                active_start[row["event_id"]] = frame
            elif row["state"] == "RESOLVED":
                start = active_start.pop(row["event_id"], frame)
                windows.append((start, frame))

    rows = load_manifest(STAGE8 / "manifest.jsonl")
    relabeled = 0
    for row in rows:
        frame = row.get("source_frame")
        if (
            row.get("split") != "train"
            or frame is None
            or not str(row.get("weather_profile", "")).startswith(
                "official-rainy-night"
            )
        ):
            continue
        in_window = any(
            start - 10 <= int(frame) <= end + 10 for start, end in windows
        )
        if in_window and row.get("risk_level") in {"low", "medium"}:
            row["risk_level"] = "high"
            row["sampling_weight"] = 250.0
            row["risk_reason_codes"] = ["rainy_event_window_relabel"]
            row["label"]["action"] = "decelerate"
            row["label"]["target_speed_kmh"] = 20.0
            relabeled += 1

    STAGE9.mkdir(parents=True, exist_ok=True)
    for name in ("images", "tensors", "intents"):
        target = STAGE9 / name
        target.mkdir(exist_ok=True)
        for source in (STAGE8 / name).glob("*.pt"):
            shutil.copy2(source, target / source.name)
    with (STAGE9 / "manifest.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("relabeled rows:", relabeled)
    from collections import Counter

    print(
        "train risk:",
        dict(
            Counter(
                r["risk_level"]
                for r in rows
                if r.get("split") == "train"
            )
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
