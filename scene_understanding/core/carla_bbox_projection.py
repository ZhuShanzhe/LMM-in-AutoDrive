"""Project CARLA actor bounding boxes into a normalized camera image plane.

The numerical helpers are independent of the CARLA package. Production code
may pass CARLA vectors, transforms, actors, and sensors directly, while unit
tests can use small stand-ins.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence


def camera_intrinsics(
    image_width: int,
    image_height: int,
    fov_deg: float,
) -> tuple[float, float, float, float]:
    """Return ``fx, fy, cx, cy`` for CARLA's horizontal field of view."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if not 0 < fov_deg < 180:
        raise ValueError("fov_deg must be between 0 and 180")
    focal = image_width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    return focal, focal, image_width / 2.0, image_height / 2.0


def _xyz(point: Any) -> tuple[float, float, float]:
    if isinstance(point, Mapping):
        values = (point.get("x"), point.get("y"), point.get("z"))
    else:
        values = (
            getattr(point, "x", None),
            getattr(point, "y", None),
            getattr(point, "z", None),
        )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in values
    ):
        raise ValueError("point must contain finite x, y and z coordinates")
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def _matrix4(matrix: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
        raise ValueError("world_to_camera_matrix must be 4x4")
    normalized: list[tuple[float, ...]] = []
    for row in matrix:
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in row
        ):
            raise ValueError("world_to_camera_matrix must contain finite numbers")
        normalized.append(tuple(float(value) for value in row))
    return tuple(normalized)


def world_to_sensor(
    point: Any,
    world_to_camera_matrix: Sequence[Sequence[float]],
) -> tuple[float, float, float]:
    """Transform one world point into CARLA sensor coordinates.

    CARLA sensor coordinates use x forward, y right and z up.
    """

    x, y, z = _xyz(point)
    matrix = _matrix4(world_to_camera_matrix)
    vector = (x, y, z, 1.0)
    transformed = tuple(
        sum(matrix[row][column] * vector[column] for column in range(4))
        for row in range(4)
    )
    w = transformed[3]
    if abs(w) > 1e-12 and abs(w - 1.0) > 1e-12:
        return transformed[0] / w, transformed[1] / w, transformed[2] / w
    return transformed[0], transformed[1], transformed[2]


def project_world_vertices(
    vertices: Iterable[Any],
    *,
    world_to_camera_matrix: Sequence[Sequence[float]],
    image_width: int,
    image_height: int,
    fov_deg: float,
    near_clip_m: float = 0.1,
) -> list[float] | None:
    """Return a clipped normalized ``xyxy`` box, or ``None`` if not visible."""

    if near_clip_m <= 0:
        raise ValueError("near_clip_m must be positive")
    fx, fy, cx, cy = camera_intrinsics(image_width, image_height, fov_deg)
    pixels: list[tuple[float, float]] = []
    for vertex in vertices:
        forward, right, up = world_to_sensor(vertex, world_to_camera_matrix)
        if forward <= near_clip_m:
            continue
        pixels.append(
            (
                fx * right / forward + cx,
                fy * -up / forward + cy,
            )
        )
    if not pixels:
        return None

    x_min = min(point[0] for point in pixels)
    y_min = min(point[1] for point in pixels)
    x_max = max(point[0] for point in pixels)
    y_max = max(point[1] for point in pixels)
    if x_max <= 0 or y_max <= 0 or x_min >= image_width or y_min >= image_height:
        return None

    x_min = min(max(x_min, 0.0), float(image_width))
    y_min = min(max(y_min, 0.0), float(image_height))
    x_max = min(max(x_max, 0.0), float(image_width))
    y_max = min(max(y_max, 0.0), float(image_height))
    if x_min >= x_max or y_min >= y_max:
        return None
    return [
        round(x_min / image_width, 6),
        round(y_min / image_height, 6),
        round(x_max / image_width, 6),
        round(y_max / image_height, 6),
    ]


def project_actor_bboxes(
    actor: Any,
    camera_sensor: Any,
    *,
    image_width: int,
    image_height: int,
    fov_deg: float,
    near_clip_m: float = 0.1,
) -> list[list[float]]:
    """Project one live actor into one or more visible geometry boxes."""

    actor_transform = actor.get_transform()
    light_boxes = []
    if str(getattr(actor, "type_id", "")).startswith("traffic.traffic_light"):
        get_light_boxes = getattr(actor, "get_light_boxes", None)
        if callable(get_light_boxes):
            light_boxes = list(get_light_boxes())
    inverse_matrix = camera_sensor.get_transform().get_inverse_matrix()
    vertex_groups = (
        [box.get_local_vertices() for box in light_boxes]
        if light_boxes
        else [actor.bounding_box.get_world_vertices(actor_transform)]
    )
    projections = []
    for vertices in vertex_groups:
        projection = project_world_vertices(
            vertices,
            world_to_camera_matrix=inverse_matrix,
            image_width=image_width,
            image_height=image_height,
            fov_deg=fov_deg,
            near_clip_m=near_clip_m,
        )
        if projection is not None:
            projections.append(projection)
    return projections


def project_actor_bbox(
    actor: Any,
    camera_sensor: Any,
    *,
    image_width: int,
    image_height: int,
    fov_deg: float,
    near_clip_m: float = 0.1,
) -> list[float] | None:
    """Project one actor, returning the union for backward compatibility."""

    boxes = project_actor_bboxes(
        actor,
        camera_sensor,
        image_width=image_width,
        image_height=image_height,
        fov_deg=fov_deg,
        near_clip_m=near_clip_m,
    )
    if not boxes:
        return None
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def project_world_state_objects(
    world_state: Mapping[str, Any],
    actors: Iterable[Any],
    camera_sensor: Any,
    *,
    camera_name: str,
    image_width: int,
    image_height: int,
    fov_deg: float,
) -> dict[str, Any]:
    """Build frame-aligned 2D truth boxes for actors already in WorldState."""

    actor_index = {str(actor.id): actor for actor in actors}
    projections: list[dict[str, Any]] = []
    for obj in world_state.get("objects", []):
        if not isinstance(obj, Mapping):
            continue
        source_id = obj.get("source_object_id")
        actor = actor_index.get(str(source_id)) if source_id is not None else None
        if actor is None or not getattr(actor, "is_alive", True):
            continue
        bboxes = project_actor_bboxes(
            actor,
            camera_sensor,
            image_width=image_width,
            image_height=image_height,
            fov_deg=fov_deg,
        )
        base_object_id = str(obj["object_id"])
        for component_index, bbox in enumerate(bboxes):
            projection = {
                "world_object_id": base_object_id,
                "source_object_id": str(source_id),
                "category": str(obj["category"]),
                "bbox_2d": bbox,
            }
            if len(bboxes) > 1:
                projection["world_object_id"] = (
                    f"{base_object_id}_component_{component_index}"
                )
                projection["parent_world_object_id"] = base_object_id
                projection["component_index"] = component_index
            projections.append(projection)
    projections.sort(key=lambda item: item["world_object_id"])
    return {
        "schema_version": "1.0",
        "frame_id": str(world_state["frame_id"]),
        "camera_name": camera_name,
        "image_width": image_width,
        "image_height": image_height,
        "objects": projections,
    }
