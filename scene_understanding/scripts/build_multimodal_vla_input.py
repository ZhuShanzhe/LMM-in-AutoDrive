"""Build a synchronized multimodal input bundle for VLA inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from scene_understanding.core.multimodal_frame_bundle import (
    validate_multimodal_frame_bundle,
)
from scene_understanding.core.multimodal_input_adapter import (
    assemble_vla_multimodal_input,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "--request-id",
        required=True,
        help="Unique request identifier",
    )
    parser.add_argument(
        "--asr-result",
        required=True,
        type=Path,
        help="ASR result JSON file",
    )
    parser.add_argument(
        "--instruction-timestamp-s",
        required=True,
        type=float,
        help=(
            "Voice instruction acquisition timestamp "
            "in seconds; ASR processing duration must "
            "not be used"
        ),
    )
    parser.add_argument(
        "--instruction-confidence",
        type=float,
        default=None,
        help=(
            "ASR confidence in [0, 1]; may be omitted "
            "only when the ASR JSON already contains "
            "a confidence field"
        ),
    )
    parser.add_argument(
        "--language",
        default="zh-CN",
        help="BCP-47 instruction language tag",
    )
    parser.add_argument(
        "--sensor-snapshot",
        required=True,
        type=Path,
        help=(
            "Exact-frame camera and LiDAR snapshot "
            "JSON file"
        ),
    )
    parser.add_argument(
        "--world-state",
        required=True,
        type=Path,
        help="WorldState JSON file",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help=(
            "Artifact directory used for the saved "
            "WorldState reference"
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="MultimodalFrameBundle JSON output path",
    )
    parser.add_argument(
        "--tolerance-ms",
        type=float,
        default=50.0,
        help=(
            "Maximum sensor timestamp skew in "
            "milliseconds (default: 50)"
        ),
    )
    parser.add_argument(
        "--bundle-id",
        default=None,
        help=(
            "Optional explicit bundle identifier; "
            "otherwise derived from frame and request"
        ),
    )
    parser.add_argument(
        "--required-modality",
        action="append",
        dest="required_modalities",
        default=None,
        help=(
            "Required modality name; repeat this "
            "option to override the default set"
        ),
    )
    return parser


def _read_json_object(
    path: Path,
    *,
    label: str,
) -> dict[str, Any]:
    document = json.loads(
        path.read_text(encoding="utf-8")
    )
    if not isinstance(document, dict):
        raise ValueError(
            f"{label} must contain a JSON object"
        )
    return document


def _write_json(
    path: Path,
    document: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(
            document,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main(
    argv: list[str] | None = None,
) -> int:
    args = build_parser().parse_args(argv)

    try:
        asr_result = _read_json_object(
            args.asr_result,
            label="ASR result",
        )
        sensor_snapshot = _read_json_object(
            args.sensor_snapshot,
            label="sensor snapshot",
        )
        world_state = _read_json_object(
            args.world_state,
            label="WorldState",
        )

        assemble_arguments: dict[str, Any] = {
            "request_id": args.request_id,
            "asr_result": asr_result,
            "instruction_timestamp_s": (
                args.instruction_timestamp_s
            ),
            "instruction_confidence": (
                args.instruction_confidence
            ),
            "language": args.language,
            "sensor_snapshot": sensor_snapshot,
            "world_state": world_state,
            "output_dir": args.output_dir,
            "tolerance_ms": args.tolerance_ms,
            "bundle_id": args.bundle_id,
        }

        if args.required_modalities is not None:
            assemble_arguments[
                "required_modalities"
            ] = args.required_modalities

        bundle = assemble_vla_multimodal_input(
            **assemble_arguments
        )

        errors = validate_multimodal_frame_bundle(
            bundle
        )
        if errors:
            raise ValueError(
                "invalid MultimodalFrameBundle: "
                + "; ".join(errors)
            )

        _write_json(
            args.output,
            bundle,
        )
    except (
        OSError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        print(
            f"ERROR: {error}",
            file=sys.stderr,
        )
        return 1

    synchronization = bundle["synchronization"]
    lidar_status = (
        "available"
        if bundle["lidar"] is not None
        else "missing"
    )

    print(
        "Wrote multimodal VLA input to "
        f"{args.output}"
    )
    print(
        "Synchronization status: "
        f"{synchronization['status']}"
    )
    print(
        "Simulation frame: "
        f"{bundle['simulation_frame']}"
    )
    print(
        "Cameras: "
        f"{len(bundle['cameras'])}; "
        f"LiDAR: {lidar_status}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
