"""Capture sensor calibration from instantiated CARLA actors."""

import math
from typing import Any


def sensor_record(actor: Any, mounting_transform: Any) -> dict:
    attributes = dict(actor.attributes)
    record = {
        "blueprint": str(actor.type_id),
        "attributes": attributes,
        "sensor_to_ego": mounting_transform.get_matrix(),
    }
    if actor.type_id == "sensor.camera.rgb":
        width = int(attributes["image_size_x"])
        height = int(attributes["image_size_y"])
        fov = float(attributes["fov"])
        if width <= 0 or height <= 0 or not 0 < fov < 180:
            raise ValueError("invalid camera intrinsics")
        focal = width / (2 * math.tan(math.radians(fov) / 2))
        record.update(
            width=width, height=height, horizontal_fov_deg=fov,
            intrinsic_matrix=[[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]],
            intrinsic_model="ideal_pinhole; lens/postprocessing attributes recorded separately",
            camera_optical_from_sensor=[[0, 1, 0], [0, 0, -1], [1, 0, 0]],
            artifact_format="PNG",
        )
    else:
        record["artifact_format"] = "CARLA save_to_disk PLY; inspect PLY header for available properties"
    return record


def calibration_document(sensors: dict) -> dict:
    return {
        "schema_version": "carla_sensor_calibration/1.0",
        "length_unit": "metre",
        "sensor_and_ego_axes": "CARLA: x forward, y right, z up",
        "matrix_convention": "column vectors; p_ego = sensor_to_ego @ p_sensor",
        "camera_optical_axes": "x right, y down, z forward",
        "sensors": sensors,
    }
