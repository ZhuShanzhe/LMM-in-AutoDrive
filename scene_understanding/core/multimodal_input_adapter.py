"""Adapters from team ASR and CARLA WorldState to VLA input bundles."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from scene_understanding.core.multimodal_frame_bundle import (
    build_multimodal_frame_bundle,
)
from scene_understanding.core.world_state import (
    validate_world_state,
)


DEFAULT_REQUIRED_MODALITIES = (
    "instruction",
    "front_rgb",
    "left_rgb",
    "right_rgb",
    "rear_rgb",
    "lidar",
    "world_state",
)


def _finite_nonnegative_number(
    value: Any,
    field_name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(
            f"{field_name} must be a finite "
            "non-negative number"
        )
    return float(value)


def _confidence(
    value: Any,
) -> float:
    confidence = _finite_nonnegative_number(
        value,
        "confidence",
    )

    if confidence > 1.0:
        raise ValueError(
            "confidence must be between 0 and 1"
        )

    return confidence


def normalize_asr_instruction(
    asr_result: Mapping[str, Any],
    *,
    timestamp_s: float,
    language: str,
    confidence: float | None = None,
) -> dict[str, Any]:
    """Normalize both team ASR output variants.

    ``timestamp_s`` is supplied by the caller because ASR processing
    duration is not the audio acquisition time.
    """

    if not isinstance(asr_result, Mapping):
        raise TypeError(
            "asr_result must be an object"
        )

    text = ""

    # Full ASR pipelines use chinese_text. Direct ASR services and
    # Qwen3ASRPipeline may use text.
    for key in ("chinese_text", "text"):
        candidate = asr_result.get(key)
        if isinstance(candidate, str) and candidate.strip():
            text = candidate.strip()
            break

    if not text:
        raise ValueError(
            "ASR text/chinese_text must be non-empty"
        )

    if not isinstance(language, str) or not language.strip():
        raise ValueError(
            "language must be a non-empty string"
        )

    if confidence is None:
        confidence = asr_result.get("confidence")

    if confidence is None:
        raise ValueError(
            "confidence must be supplied because the "
            "ASR result does not provide it"
        )

    return {
        "source": "asr",
        "text": text,
        "language": language.strip(),
        "confidence": _confidence(confidence),
        "timestamp_s": _finite_nonnegative_number(
            timestamp_s,
            "timestamp_s",
        ),
    }


def _safe_frame_id(frame_id: Any) -> str:
    if not isinstance(frame_id, str) or not frame_id:
        raise ValueError(
            "WorldState frame_id must be a "
            "non-empty string"
        )

    if (
        Path(frame_id).name != frame_id
        or frame_id in {".", ".."}
    ):
        raise ValueError(
            "WorldState frame_id must not contain "
            "path separators"
        )

    return frame_id


def write_world_state_reference(
    world_state: Mapping[str, Any],
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Validate and atomically persist one full WorldState."""

    if not isinstance(world_state, Mapping):
        raise TypeError(
            "world_state must be an object"
        )

    document = dict(world_state)
    errors = validate_world_state(document)

    if errors:
        raise ValueError(
            "invalid WorldState: " + "; ".join(errors)
        )

    frame_id = _safe_frame_id(
        document.get("frame_id")
    )

    root = Path(output_dir).resolve()
    world_dir = root / "world_states"
    world_dir.mkdir(parents=True, exist_ok=True)

    target = world_dir / f"{frame_id}.json"
    temporary = world_dir / f".{frame_id}.json.tmp"

    temporary.write_text(
        json.dumps(
            document,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)

    return {
        "frame_id": frame_id,
        "simulation_frame": int(
            document["simulation_frame"]
        ),
        "timestamp_s": float(
            document["timestamp_s"]
        ),
        "path": str(target.resolve()),
    }


def _bundle_identifier(
    frame_id: str,
    request_id: str,
) -> str:
    safe_request = re.sub(
        r"[^A-Za-z0-9_.]+",
        "_",
        request_id,
    ).strip("_")

    if not safe_request:
        safe_request = "request"

    return f"bundle_{frame_id}_{safe_request}"


def _metric_source(
    source: str,
) -> str:
    if source == "carla":
        return "carla_actor_api"
    if source == "nuscenes":
        return "dataset_annotation"
    return "unavailable"


def assemble_vla_multimodal_input(
    *,
    request_id: str,
    asr_result: Mapping[str, Any],
    instruction_timestamp_s: float,
    instruction_confidence: float | None,
    language: str,
    sensor_snapshot: Mapping[str, Any],
    world_state: Mapping[str, Any],
    output_dir: str | Path,
    tolerance_ms: float = 50.0,
    required_modalities: Sequence[str] = (
        DEFAULT_REQUIRED_MODALITIES
    ),
    bundle_id: str | None = None,
) -> dict[str, Any]:
    """Assemble ASR, synchronized sensors and WorldState for VLA."""

    if not isinstance(request_id, str) or not request_id:
        raise ValueError(
            "request_id must be a non-empty string"
        )

    if not isinstance(world_state, Mapping):
        raise TypeError(
            "world_state must be an object"
        )

    instruction = normalize_asr_instruction(
        asr_result,
        timestamp_s=instruction_timestamp_s,
        confidence=instruction_confidence,
        language=language,
    )

    world_reference = write_world_state_reference(
        world_state,
        output_dir=output_dir,
    )

    frame_id = world_reference["frame_id"]
    source = world_state.get("source")

    if not isinstance(source, str) or not source:
        raise ValueError(
            "WorldState source must be a "
            "non-empty string"
        )

    resolved_bundle_id = (
        bundle_id
        if bundle_id is not None
        else _bundle_identifier(
            frame_id,
            request_id,
        )
    )

    return build_multimodal_frame_bundle(
        bundle_id=resolved_bundle_id,
        request_id=request_id,
        source=source,
        frame_id=frame_id,
        simulation_frame=world_reference[
            "simulation_frame"
        ],
        timestamp_s=world_reference["timestamp_s"],
        instruction=instruction,
        sensor_snapshot=dict(sensor_snapshot),
        world_state=world_reference,
        provenance={
            "capture_module": (
                "carla_multimodal_sensor_manager"
            ),
            "metric_source": _metric_source(source),
        },
        required_modalities=list(
            required_modalities
        ),
        tolerance_ms=tolerance_ms,
    )
