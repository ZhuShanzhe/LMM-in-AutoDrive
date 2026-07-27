"""Stable multimodal frame-bundle contract for VLA integration.

The contract aligns an instruction, multi-view camera frames, LiDAR data and
one WorldState snapshot without importing CARLA or model runtimes.
"""

from __future__ import annotations

import math
from typing import Any


SCHEMA_VERSION = "1.0.0"

TOP_LEVEL_KEYS = {
    "schema_version",
    "bundle_id",
    "request_id",
    "source",
    "frame_id",
    "simulation_frame",
    "timestamp_s",
    "synchronization",
    "instruction",
    "cameras",
    "lidar",
    "world_state",
    "provenance",
}

SYNCHRONIZATION_KEYS = {
    "status",
    "reference_frame",
    "reference_timestamp_s",
    "tolerance_ms",
    "max_skew_ms",
    "required_modalities",
    "missing_modalities",
}

INSTRUCTION_KEYS = {
    "source",
    "text",
    "language",
    "confidence",
    "timestamp_s",
}

CAMERA_KEYS = {
    "sensor_name",
    "frame",
    "timestamp_s",
    "image_path",
    "image_size",
    "intrinsic_matrix",
    "sensor_to_ego",
}

IMAGE_SIZE_KEYS = {
    "width",
    "height",
}

LIDAR_KEYS = {
    "sensor_name",
    "frame",
    "timestamp_s",
    "point_cloud_path",
    "point_count",
    "coordinate_frame",
    "sensor_to_ego",
}

WORLD_STATE_KEYS = {
    "frame_id",
    "simulation_frame",
    "timestamp_s",
    "path",
}

PROVENANCE_KEYS = {
    "capture_module",
    "metric_source",
}

SOURCES = {
    "carla",
    "nuscenes",
    "waymo",
    "other",
}

SYNCHRONIZATION_STATUSES = {
    "EXACT",
    "WITHIN_TOLERANCE",
    "INCOMPLETE",
}

INSTRUCTION_SOURCES = {
    "asr",
    "text",
    "driving_intent",
}

LIDAR_COORDINATE_FRAMES = {
    "sensor",
    "ego",
    "carla_world",
}

METRIC_SOURCES = {
    "carla_actor_api",
    "dataset_annotation",
    "unavailable",
}

TIMESTAMP_EPSILON_MS = 1e-3


def _is_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _exact_keys(
    value: Any,
    required: set[str],
    path: str,
    errors: list[str],
) -> bool:
    if not isinstance(value, dict):
        errors.append(f"{path}: expected an object")
        return False

    actual = set(value)
    missing = sorted(required - actual)
    extra = sorted(actual - required)

    if missing:
        errors.append(
            f"{path}: missing fields: {', '.join(missing)}"
        )
    if extra:
        errors.append(
            f"{path}: unexpected fields: {', '.join(extra)}"
        )

    return not missing and not extra


def _nonempty_string(
    value: Any,
    path: str,
    errors: list[str],
) -> None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{path}: expected a non-empty string")


def _number(
    value: Any,
    path: str,
    errors: list[str],
    *,
    allow_null: bool = False,
    minimum: float | None = None,
    maximum: float | None = None,
) -> None:
    if allow_null and value is None:
        return

    if not _is_number(value):
        suffix = " or null" if allow_null else ""
        errors.append(
            f"{path}: expected a finite number{suffix}"
        )
        return

    number = float(value)

    if minimum is not None and number < minimum:
        errors.append(
            f"{path}: must be at least {minimum}"
        )
    if maximum is not None and number > maximum:
        errors.append(
            f"{path}: must be at most {maximum}"
        )


def _integer(
    value: Any,
    path: str,
    errors: list[str],
    *,
    minimum: int = 0,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        errors.append(f"{path}: expected an integer")
        return

    if value < minimum:
        errors.append(
            f"{path}: must be at least {minimum}"
        )


def _enum(
    value: Any,
    allowed: set[str],
    path: str,
    errors: list[str],
) -> None:
    if value not in allowed:
        errors.append(
            f"{path}: invalid value {value!r}; "
            f"allowed: {', '.join(sorted(allowed))}"
        )


def _string_list(
    value: Any,
    path: str,
    errors: list[str],
) -> list[str]:
    if not isinstance(value, list):
        errors.append(f"{path}: expected an array")
        return []

    result: list[str] = []

    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            errors.append(
                f"{path}[{index}]: expected a "
                "non-empty string"
            )
            continue
        result.append(item)

    if len(result) != len(set(result)):
        errors.append(f"{path}: values must be unique")

    return result


def _matrix(
    value: Any,
    length: int,
    path: str,
    errors: list[str],
) -> None:
    if not isinstance(value, list):
        errors.append(f"{path}: expected an array")
        return

    if len(value) != length:
        errors.append(
            f"{path}: expected exactly {length} values"
        )

    for index, item in enumerate(value):
        _number(
            item,
            f"{path}[{index}]",
            errors,
        )


def _validate_instruction(
    value: Any,
    errors: list[str],
) -> None:
    path = "$.instruction"

    if not _exact_keys(
        value,
        INSTRUCTION_KEYS,
        path,
        errors,
    ):
        return

    _enum(
        value["source"],
        INSTRUCTION_SOURCES,
        f"{path}.source",
        errors,
    )
    _nonempty_string(
        value["text"],
        f"{path}.text",
        errors,
    )
    _nonempty_string(
        value["language"],
        f"{path}.language",
        errors,
    )
    _number(
        value["confidence"],
        f"{path}.confidence",
        errors,
        allow_null=True,
        minimum=0.0,
        maximum=1.0,
    )
    _number(
        value["timestamp_s"],
        f"{path}.timestamp_s",
        errors,
        allow_null=True,
        minimum=0.0,
    )


def _validate_camera(
    value: Any,
    index: int,
    errors: list[str],
) -> None:
    path = f"$.cameras[{index}]"

    if not _exact_keys(
        value,
        CAMERA_KEYS,
        path,
        errors,
    ):
        return

    _nonempty_string(
        value["sensor_name"],
        f"{path}.sensor_name",
        errors,
    )
    _integer(
        value["frame"],
        f"{path}.frame",
        errors,
    )
    _number(
        value["timestamp_s"],
        f"{path}.timestamp_s",
        errors,
        minimum=0.0,
    )
    _nonempty_string(
        value["image_path"],
        f"{path}.image_path",
        errors,
    )

    image_size = value["image_size"]
    image_size_path = f"{path}.image_size"

    if _exact_keys(
        image_size,
        IMAGE_SIZE_KEYS,
        image_size_path,
        errors,
    ):
        _integer(
            image_size["width"],
            f"{image_size_path}.width",
            errors,
            minimum=1,
        )
        _integer(
            image_size["height"],
            f"{image_size_path}.height",
            errors,
            minimum=1,
        )

    _matrix(
        value["intrinsic_matrix"],
        9,
        f"{path}.intrinsic_matrix",
        errors,
    )
    _matrix(
        value["sensor_to_ego"],
        16,
        f"{path}.sensor_to_ego",
        errors,
    )


def _validate_lidar(
    value: Any,
    errors: list[str],
) -> None:
    if value is None:
        return

    path = "$.lidar"

    if not _exact_keys(
        value,
        LIDAR_KEYS,
        path,
        errors,
    ):
        return

    _nonempty_string(
        value["sensor_name"],
        f"{path}.sensor_name",
        errors,
    )
    _integer(
        value["frame"],
        f"{path}.frame",
        errors,
    )
    _number(
        value["timestamp_s"],
        f"{path}.timestamp_s",
        errors,
        minimum=0.0,
    )
    _nonempty_string(
        value["point_cloud_path"],
        f"{path}.point_cloud_path",
        errors,
    )
    _integer(
        value["point_count"],
        f"{path}.point_count",
        errors,
    )
    _enum(
        value["coordinate_frame"],
        LIDAR_COORDINATE_FRAMES,
        f"{path}.coordinate_frame",
        errors,
    )
    _matrix(
        value["sensor_to_ego"],
        16,
        f"{path}.sensor_to_ego",
        errors,
    )


def _validate_world_state_reference(
    value: Any,
    errors: list[str],
) -> None:
    path = "$.world_state"

    if not _exact_keys(
        value,
        WORLD_STATE_KEYS,
        path,
        errors,
    ):
        return

    _nonempty_string(
        value["frame_id"],
        f"{path}.frame_id",
        errors,
    )
    _integer(
        value["simulation_frame"],
        f"{path}.simulation_frame",
        errors,
    )
    _number(
        value["timestamp_s"],
        f"{path}.timestamp_s",
        errors,
        minimum=0.0,
    )
    _nonempty_string(
        value["path"],
        f"{path}.path",
        errors,
    )


def _validate_provenance(
    value: Any,
    errors: list[str],
) -> None:
    path = "$.provenance"

    if not _exact_keys(
        value,
        PROVENANCE_KEYS,
        path,
        errors,
    ):
        return

    _nonempty_string(
        value["capture_module"],
        f"{path}.capture_module",
        errors,
    )
    _enum(
        value["metric_source"],
        METRIC_SOURCES,
        f"{path}.metric_source",
        errors,
    )


def _same_number(
    first: Any,
    second: Any,
) -> bool:
    return (
        _is_number(first)
        and _is_number(second)
        and abs(float(first) - float(second)) <= 1e-9
    )


def validate_multimodal_frame_bundle(
    data: Any,
) -> list[str]:
    """Return structural and synchronization errors."""

    errors: list[str] = []

    if not _exact_keys(
        data,
        TOP_LEVEL_KEYS,
        "$",
        errors,
    ):
        return errors

    if data["schema_version"] != SCHEMA_VERSION:
        errors.append(
            "$.schema_version: expected "
            f"{SCHEMA_VERSION!r}"
        )

    for key in (
        "bundle_id",
        "request_id",
        "frame_id",
    ):
        _nonempty_string(
            data[key],
            f"$.{key}",
            errors,
        )

    _enum(
        data["source"],
        SOURCES,
        "$.source",
        errors,
    )
    _integer(
        data["simulation_frame"],
        "$.simulation_frame",
        errors,
    )
    _number(
        data["timestamp_s"],
        "$.timestamp_s",
        errors,
        minimum=0.0,
    )

    synchronization = data["synchronization"]

    if not _exact_keys(
        synchronization,
        SYNCHRONIZATION_KEYS,
        "$.synchronization",
        errors,
    ):
        return errors

    status = synchronization["status"]

    _enum(
        status,
        SYNCHRONIZATION_STATUSES,
        "$.synchronization.status",
        errors,
    )
    _integer(
        synchronization["reference_frame"],
        "$.synchronization.reference_frame",
        errors,
    )
    _number(
        synchronization["reference_timestamp_s"],
        "$.synchronization.reference_timestamp_s",
        errors,
        minimum=0.0,
    )
    _number(
        synchronization["tolerance_ms"],
        "$.synchronization.tolerance_ms",
        errors,
        minimum=0.0,
    )
    _number(
        synchronization["max_skew_ms"],
        "$.synchronization.max_skew_ms",
        errors,
        minimum=0.0,
    )

    required_modalities = _string_list(
        synchronization["required_modalities"],
        "$.synchronization.required_modalities",
        errors,
    )
    missing_modalities = _string_list(
        synchronization["missing_modalities"],
        "$.synchronization.missing_modalities",
        errors,
    )

    _validate_instruction(
        data["instruction"],
        errors,
    )

    cameras = data["cameras"]

    if not isinstance(cameras, list):
        errors.append("$.cameras: expected an array")
        cameras = []
    elif not cameras:
        errors.append(
            "$.cameras: at least one camera is required"
        )

    camera_names: list[str] = []

    for index, camera in enumerate(cameras):
        _validate_camera(
            camera,
            index,
            errors,
        )
        if isinstance(camera, dict):
            name = camera.get("sensor_name")
            if isinstance(name, str) and name:
                if name in camera_names:
                    errors.append(
                        "$.cameras: duplicate sensor_name "
                        f"{name!r}"
                    )
                camera_names.append(name)

    _validate_lidar(
        data["lidar"],
        errors,
    )
    _validate_world_state_reference(
        data["world_state"],
        errors,
    )
    _validate_provenance(
        data["provenance"],
        errors,
    )

    reference_frame = synchronization[
        "reference_frame"
    ]
    reference_timestamp = synchronization[
        "reference_timestamp_s"
    ]

    if (
        isinstance(reference_frame, int)
        and not isinstance(reference_frame, bool)
        and data["simulation_frame"]
        != reference_frame
    ):
        errors.append(
            "$.simulation_frame: must equal "
            "synchronization.reference_frame"
        )

    if not _same_number(
        data["timestamp_s"],
        reference_timestamp,
    ):
        errors.append(
            "$.timestamp_s: must equal "
            "synchronization.reference_timestamp_s"
        )

    world_state = data["world_state"]

    if isinstance(world_state, dict):
        if (
            world_state.get("frame_id")
            != data["frame_id"]
        ):
            errors.append(
                "$.world_state.frame_id: must equal "
                "$.frame_id"
            )
        if (
            world_state.get("simulation_frame")
            != reference_frame
        ):
            errors.append(
                "$.world_state.simulation_frame: "
                "must equal "
                "synchronization.reference_frame"
            )

    if status == "EXACT":
        for index, camera in enumerate(cameras):
            if (
                isinstance(camera, dict)
                and camera.get("frame")
                != reference_frame
            ):
                errors.append(
                    f"$.cameras[{index}].frame: "
                    "must equal "
                    "synchronization.reference_frame"
                )

        lidar = data["lidar"]
        if (
            isinstance(lidar, dict)
            and lidar.get("frame") != reference_frame
        ):
            errors.append(
                "$.lidar.frame: must equal "
                "synchronization.reference_frame"
            )

    present_modalities = {
        "instruction",
        "world_state",
        *camera_names,
    }

    if data["lidar"] is not None:
        present_modalities.add("lidar")

    required_set = set(required_modalities)
    missing_set = set(missing_modalities)

    undeclared_missing = (
        required_set - present_modalities - missing_set
    )

    for modality in sorted(undeclared_missing):
        errors.append(
            "$.synchronization.missing_modalities: "
            f"must contain {modality!r} when the "
            "required modality is unavailable"
        )

    for modality in sorted(
        missing_set - required_set
    ):
        errors.append(
            "$.synchronization.missing_modalities: "
            f"{modality!r} is not a required modality"
        )

    for modality in sorted(
        missing_set & present_modalities
    ):
        errors.append(
            "$.synchronization.missing_modalities: "
            f"{modality!r} is present and must not "
            "be marked missing"
        )

    if status == "INCOMPLETE":
        if not missing_set:
            errors.append(
                "$.synchronization.status: "
                "INCOMPLETE requires at least one "
                "missing modality"
            )
    elif missing_set:
        errors.append(
            "$.synchronization.status: only "
            "INCOMPLETE may contain missing modalities"
        )

    timestamps: list[float] = []

    for camera in cameras:
        if (
            isinstance(camera, dict)
            and _is_number(
                camera.get("timestamp_s")
            )
        ):
            timestamps.append(
                float(camera["timestamp_s"])
            )

    lidar = data["lidar"]
    if (
        isinstance(lidar, dict)
        and _is_number(lidar.get("timestamp_s"))
    ):
        timestamps.append(
            float(lidar["timestamp_s"])
        )

    if (
        isinstance(world_state, dict)
        and _is_number(
            world_state.get("timestamp_s")
        )
    ):
        timestamps.append(
            float(world_state["timestamp_s"])
        )

    if _is_number(reference_timestamp):
        actual_skew_ms = max(
            (
                abs(
                    timestamp
                    - float(reference_timestamp)
                )
                * 1000.0
                for timestamp in timestamps
            ),
            default=0.0,
        )

        declared_skew = synchronization[
            "max_skew_ms"
        ]

        if (
            _is_number(declared_skew)
            and abs(
                float(declared_skew)
                - actual_skew_ms
            )
            > TIMESTAMP_EPSILON_MS
        ):
            errors.append(
                "$.synchronization.max_skew_ms: "
                f"declared {float(declared_skew):.6f} "
                "does not match computed "
                f"{actual_skew_ms:.6f}"
            )

        tolerance = synchronization[
            "tolerance_ms"
        ]

        if (
            _is_number(tolerance)
            and actual_skew_ms
            > float(tolerance)
            + TIMESTAMP_EPSILON_MS
        ):
            errors.append(
                "$.synchronization.max_skew_ms: "
                f"computed {actual_skew_ms:.6f} "
                "exceeds tolerance "
                f"{float(tolerance):.6f}"
            )

        if (
            status == "EXACT"
            and actual_skew_ms
            > TIMESTAMP_EPSILON_MS
        ):
            errors.append(
                "$.synchronization.status: "
                "EXACT requires zero sensor "
                "timestamp skew"
            )

    return errors


def _builder_frame(
    value: Any,
    field_name: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(
            f"{field_name} must be a non-negative integer"
        )
    return value


def _builder_number(
    value: Any,
    field_name: str,
) -> float:
    if not _is_number(value) or float(value) < 0.0:
        raise ValueError(
            f"{field_name} must be a finite "
            "non-negative number"
        )
    return float(value)


def build_multimodal_frame_bundle(
    *,
    bundle_id: str,
    request_id: str,
    source: str,
    frame_id: str,
    simulation_frame: int,
    timestamp_s: float,
    instruction: Any,
    sensor_snapshot: Any,
    world_state: Any,
    provenance: Any,
    required_modalities: Any,
    tolerance_ms: float = 50.0,
) -> dict[str, Any]:
    """Build and validate one synchronized VLA input bundle.

    Sensor records must use the same CARLA frame. Timestamp skew is
    permitted only within ``tolerance_ms``. Instruction time is not
    included in sensor skew because a voice command may legitimately
    precede the scene frame to which it applies.
    """

    from copy import deepcopy

    simulation_frame = _builder_frame(
        simulation_frame,
        "simulation_frame",
    )
    reference_timestamp = _builder_number(
        timestamp_s,
        "timestamp_s",
    )
    tolerance = _builder_number(
        tolerance_ms,
        "tolerance_ms",
    )

    for name, value in (
        ("bundle_id", bundle_id),
        ("request_id", request_id),
        ("frame_id", frame_id),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"{name} must be a non-empty string"
            )

    if not isinstance(sensor_snapshot, dict):
        raise TypeError(
            "sensor_snapshot must be an object"
        )

    snapshot_frame = sensor_snapshot.get(
        "simulation_frame"
    )
    if snapshot_frame != simulation_frame:
        raise ValueError(
            "sensor snapshot frame must equal "
            "simulation_frame"
        )

    cameras_value = sensor_snapshot.get("cameras")
    if not isinstance(cameras_value, list):
        raise TypeError(
            "sensor_snapshot.cameras must be an array"
        )

    cameras = deepcopy(cameras_value)
    lidar = deepcopy(sensor_snapshot.get("lidar"))

    camera_names: list[str] = []
    sensor_timestamps: list[float] = []

    for index, camera in enumerate(cameras):
        if not isinstance(camera, dict):
            raise TypeError(
                "camera record must be an object"
            )

        camera_frame = camera.get("frame")
        if camera_frame != simulation_frame:
            raise ValueError(
                f"camera frame at index {index} "
                "must equal simulation_frame"
            )

        sensor_name = camera.get("sensor_name")
        if not isinstance(sensor_name, str) or not sensor_name:
            raise ValueError(
                f"camera at index {index} has invalid "
                "sensor_name"
            )
        if sensor_name in camera_names:
            raise ValueError(
                f"duplicate camera sensor_name: "
                f"{sensor_name!r}"
            )

        camera_names.append(sensor_name)
        sensor_timestamps.append(
            _builder_number(
                camera.get("timestamp_s"),
                (
                    "camera "
                    f"{sensor_name!r} timestamp_s"
                ),
            )
        )

    if lidar is not None:
        if not isinstance(lidar, dict):
            raise TypeError(
                "sensor_snapshot.lidar must be an "
                "object or null"
            )
        if lidar.get("frame") != simulation_frame:
            raise ValueError(
                "LiDAR frame must equal simulation_frame"
            )

        sensor_timestamps.append(
            _builder_number(
                lidar.get("timestamp_s"),
                "LiDAR timestamp_s",
            )
        )

    if not isinstance(world_state, dict):
        raise TypeError(
            "world_state must be an object"
        )

    world_state_copy = deepcopy(world_state)

    if (
        world_state_copy.get("simulation_frame")
        != simulation_frame
    ):
        raise ValueError(
            "WorldState frame must equal "
            "simulation_frame"
        )

    if world_state_copy.get("frame_id") != frame_id:
        raise ValueError(
            "WorldState frame_id must equal frame_id"
        )

    sensor_timestamps.append(
        _builder_number(
            world_state_copy.get("timestamp_s"),
            "WorldState timestamp_s",
        )
    )

    if not isinstance(required_modalities, list):
        raise TypeError(
            "required_modalities must be an array"
        )

    required = list(required_modalities)

    if any(
        not isinstance(item, str) or not item
        for item in required
    ):
        raise ValueError(
            "required modalities must be "
            "non-empty strings"
        )

    if len(set(required)) != len(required):
        raise ValueError(
            "required modalities must be unique"
        )

    present_modalities = {
        "instruction",
        "world_state",
        *camera_names,
    }
    if lidar is not None:
        present_modalities.add("lidar")

    missing_modalities = [
        modality
        for modality in required
        if modality not in present_modalities
    ]

    max_skew_ms = max(
        (
            abs(value - reference_timestamp)
            * 1000.0
            for value in sensor_timestamps
        ),
        default=0.0,
    )

    if (
        max_skew_ms
        > tolerance + TIMESTAMP_EPSILON_MS
    ):
        raise ValueError(
            "sensor timestamp skew "
            f"{max_skew_ms:.6f} ms exceeds "
            f"tolerance {tolerance:.6f} ms"
        )

    if missing_modalities:
        synchronization_status = "INCOMPLETE"
    elif max_skew_ms <= TIMESTAMP_EPSILON_MS:
        synchronization_status = "EXACT"
        max_skew_ms = 0.0
    else:
        synchronization_status = "WITHIN_TOLERANCE"

    bundle = {
        "schema_version": SCHEMA_VERSION,
        "bundle_id": bundle_id,
        "request_id": request_id,
        "source": source,
        "frame_id": frame_id,
        "simulation_frame": simulation_frame,
        "timestamp_s": reference_timestamp,
        "synchronization": {
            "status": synchronization_status,
            "reference_frame": simulation_frame,
            "reference_timestamp_s": (
                reference_timestamp
            ),
            "tolerance_ms": tolerance,
            "max_skew_ms": max_skew_ms,
            "required_modalities": required,
            "missing_modalities": missing_modalities,
        },
        "instruction": deepcopy(instruction),
        "cameras": cameras,
        "lidar": lidar,
        "world_state": world_state_copy,
        "provenance": deepcopy(provenance),
    }

    errors = validate_multimodal_frame_bundle(bundle)

    if errors:
        raise ValueError(
            "built multimodal frame bundle is invalid: "
            + "; ".join(errors)
        )

    return bundle
