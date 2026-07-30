"""Add scheduled voice clips to a completed CARLA demonstration video."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


VOICE_FILES = {
    "c01_depart_45": "保持当前车道提速至45.mp3",
    "c02_cruise_60": "保持当前车道提速至 60.mp3",
    "c03_reduce_for_right_turn": "减速至 30 公里每小时.mp3",
    "c04_change_left": "确认左侧安全后向左变.mp3",
    "c05_change_right": "确认右侧安全后向右变.mp3",
    "c06_hold_45": "保持当前车道提速至45.mp3",
    "c07_turn_left": "前方路口左转.mp3",
    "c08_keep_50": "保持当前车道以 50.mp3",
    "c09_reduce_for_turn": "减速至 30 公里每小时.mp3",
    "c10_keep_35": "完成左转后保持当前车道以.mp3",
    "c11_accelerate_50": "道路通畅提速至 50.mp3",
    "c12_keep_50": "保持当前车道以 50.mp3",
    "c13_reduce_45": "减速至 45 公里每小时.mp3",
    "c14_reduce_40_final": "减速至 40 公里每小时并保持当前车道.mp3",
    "c15_keep_to_goal": "保持当前车道行驶至终.mp3",
}


def _triggers(frames_path: Path, audio_dir: Path):
    seen: set[str] = set()
    result = []
    for line in frames_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        intent = record.get("intent", {})
        command_id = intent.get("command_id")
        if (
            not isinstance(command_id, str)
            or command_id in seen
            or str(intent.get("command_phase", "")).upper() != "EXECUTING"
        ):
            continue
        filename = VOICE_FILES.get(command_id)
        path = audio_dir / filename if filename else None
        if path is not None and path.is_file():
            seen.add(command_id)
            result.append((float(record.get("sim_time_s", 0.0)), command_id, path))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--frames", required=True, type=Path)
    parser.add_argument("--audio-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()

    clips = _triggers(args.frames, args.audio_dir)
    if not clips:
        raise RuntimeError("no EXECUTING command frames with matching audio clips")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [str(args.ffmpeg), "-y", "-i", str(args.video)]
    for _, _, path in clips:
        command.extend(["-i", str(path)])
    filters = []
    labels = []
    for index, (time_s, _, _) in enumerate(clips, start=1):
        label = f"voice{index}"
        delay_ms = max(0, round(time_s * 1000))
        filters.append(f"[{index}:a]adelay={delay_ms}|{delay_ms}[{label}]")
        labels.append(f"[{label}]")
    filters.append(
        "".join(labels)
        + f"amix=inputs={len(labels)}:duration=longest:dropout_transition=1000000,"
        + f"volume={len(labels)}[audio]"
    )
    command.extend([
        "-filter_complex", ";".join(filters),
        "-map", "0:v:0", "-map", "[audio]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", str(args.output),
    ])
    subprocess.run(command, check=True)
    print(json.dumps({
        "output": str(args.output),
        "clips": [{"time_s": time_s, "command_id": command_id, "audio": str(path)}
                  for time_s, command_id, path in clips],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
