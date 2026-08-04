"""Complete visual preview entry for the Town05_Opt Scene 3 rebuild.

This wrapper runs the full 6 km route, records the required four low-signal
RGB views plus a third-person H.264 preview, and enables strict completion
checks. Importing the module does not require the CARLA Python package.
"""

from __future__ import annotations

# ============================================================
# CARLA 0.9.16 PythonAPI path fix
# 必须放在 import run_emergency_response_6km 之前
# ============================================================

import sys
from pathlib import Path

CARLA_PYTHON_API = Path(
    r"D:\CARLA_0.9.16\PythonAPI\carla"
)

if CARLA_PYTHON_API.exists():
    sys.path.insert(
        0,
        str(CARLA_PYTHON_API)
    )
else:
    raise RuntimeError(
        f"CARLA PythonAPI not found: {CARLA_PYTHON_API}"
    )


# ============================================================
# Normal imports
# ============================================================

import argparse
import bisect
import json
from typing import Any, Sequence

import run_emergency_response_6km as runner


EVENT_LABELS = {
    "scene3_cut_in": "CUT-IN VEHICLE",
    "scene3_advance_warning": "WORK-ZONE WARNING",
    "scene3_cone_taper": "LANE NARROWING",
    "scene3_work_zone": "ACTIVE WORK ZONE",
    "scene3_temporary_pedestrian": "WORKER CROSSING",
    "scene3_blocked_lane": "BLOCKED LANE",
    "scene3_work_zone_exit": "WORK-ZONE EXIT",
}


def route_anchors(
    camera_frames: Sequence[int],
    ground_truth_rows: Sequence[dict[str, Any]],
    *,
    route_completed: bool,
    finish_progress_m: float = 5990.0,
) -> tuple[list[int], list[float]]:

    """Build monotonic frame/progress anchors for offline HUD rendering."""

    anchors = {
        int(row["simulation_frame"]): float(
            row.get("route_progress_m", row.get("route_s_m", 0.0))
        )
        for row in ground_truth_rows
        if "simulation_frame" in row
    }

    if camera_frames:
        anchors.setdefault(int(min(camera_frames)), 0.0)

        if route_completed:
            anchors[int(max(camera_frames))] = float(
                finish_progress_m
            )

    ordered = sorted(anchors.items())

    return (
        [item[0] for item in ordered],
        [item[1] for item in ordered],
    )


def interpolate_route_s(
    frame: int,
    anchor_frames: Sequence[int],
    route_progress: Sequence[float],
) -> float:

    if not anchor_frames:
        return 0.0

    if len(anchor_frames) != len(route_progress):
        return 0.0

    index = bisect.bisect_left(
        anchor_frames,
        int(frame)
    )

    if index <= 0:
        return float(route_progress[0])

    if index >= len(anchor_frames):
        return float(route_progress[-1])

    left_frame = int(anchor_frames[index - 1])
    right_frame = int(anchor_frames[index])

    ratio = (
        int(frame) - left_frame
    ) / max(
        right_frame - left_frame,
        1
    )

    return (
        float(route_progress[index - 1])
        +
        ratio *
        (
            float(route_progress[index])
            -
            float(route_progress[index - 1])
        )
    )


def active_event_label(
    frame: int,
    timeline: Sequence[dict[str, Any]],
) -> str:

    active: set[str] = set()

    for row in sorted(
        timeline,
        key=lambda item: int(item["simulation_frame"])
    ):

        if int(row["simulation_frame"]) > int(frame):
            break

        event_id = str(row["event_id"])

        if row["state"] == "ACTIVE":
            active.add(event_id)

        elif row["state"] == "RESOLVED":
            active.discard(event_id)

    if active:
        return " + ".join(
            EVENT_LABELS.get(
                item,
                item.upper()
            )
            for item in sorted(active)
        )

    if timeline and int(frame) >= int(
        timeline[-1]["simulation_frame"]
    ):
        return "ALL EVENTS RESOLVED"

    return "RAINY-NIGHT URBAN EXPRESSWAY"


def build_parser():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=2000
    )

    parser.add_argument(
        "--traffic-manager-port",
        type=int,
        default=8000
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            /
            "outputs"
            /
            "scene3_town05_preview"
        )
    )

    parser.add_argument(
        "--video-output",
        type=Path
    )

    parser.add_argument(
        "--ego-speed-kmh",
        type=float,
        default=40.0
    )

    parser.add_argument(
        "--fixed-delta-seconds",
        type=float,
        default=0.05
    )

    parser.add_argument(
        "--camera-tick",
        type=float,
        default=0.2
    )

    parser.add_argument(
        "--validate-only",
        action="store_true"
    )

    parser.add_argument(
        "--no-ground-truth",
        action="store_true"
    )

    parser.add_argument(
        "--no-strict-completion",
        action="store_true"
    )

    return parser


def main(argv=None):

    args = build_parser().parse_args(argv)

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    video_output = (
        args.video_output.expanduser().resolve()
        if args.video_output
        else
        output_dir /
        "scene3_town05_complete_preview.mp4"
    )


    runner_args = [
        "--host",
        args.host,

        "--port",
        str(args.port),

        "--traffic-manager-port",
        str(args.traffic_manager_port),

        "--output-dir",
        str(output_dir),

        "--duration",
        "0",

        "--fixed-delta-seconds",
        str(args.fixed_delta_seconds),

        "--camera-tick",
        str(args.camera_tick),

        "--camera-mode",
        "four-view-plus-chase",

        "--presentation-lighting",
        "official-rainy-night",

        "--ego-speed-kmh",
        str(args.ego_speed_kmh),

        "--video-output",
        str(video_output),

        "--video-overlay",

        "--video-fps",
        "20",
    ]


    if args.validate_only:
        runner_args.append(
            "--validate-config-only"
        )


    if not args.no_ground_truth:
        runner_args.extend(
            [
                "--record-ground-truth",
                "--ground-truth-every-n",
                "1",
            ]
        )


    if not args.no_strict_completion:
        runner_args.append(
            "--require-complete-scene"
        )


    print(
        json.dumps(
            {
                "preview":
                    "Town05_Opt Scene 3 complete 6 km",

                "four_view_output":
                    str(output_dir / "rgb"),

                "h264_output":
                    str(video_output),

                "strict_completion":
                    not args.no_strict_completion,
            },

            ensure_ascii=False,
            indent=2,
        )
    )


    return int(
        runner.main(runner_args) or 0
    )


if __name__ == "__main__":
    raise SystemExit(main())