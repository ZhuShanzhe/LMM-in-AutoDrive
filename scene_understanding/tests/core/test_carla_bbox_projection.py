from __future__ import annotations

import unittest

from scene_understanding.core.carla_bbox_projection import (
    camera_intrinsics,
    project_actor_bbox,
    project_actor_bboxes,
    project_world_state_objects,
    project_world_vertices,
)


IDENTITY = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


class _Transform:
    def get_inverse_matrix(self):
        return IDENTITY


class _BoundingBox:
    def __init__(self, vertices):
        self.vertices = vertices

    def get_world_vertices(self, _transform):
        return self.vertices

    def get_local_vertices(self):
        return self.vertices


class _Actor:
    def __init__(self, actor_id, vertices, *, type_id="vehicle.test", light_boxes=None):
        self.id = actor_id
        self.bounding_box = _BoundingBox(vertices)
        self.type_id = type_id
        self._light_boxes = light_boxes or []
        self.is_alive = True

    def get_transform(self):
        return _Transform()

    def get_light_boxes(self):
        return self._light_boxes


class _Camera:
    def get_transform(self):
        return _Transform()


def _box_vertices(x=10.0, half_y=1.0, half_z=1.0):
    return [
        {"x": x, "y": y, "z": z}
        for y in (-half_y, half_y)
        for z in (-half_z, half_z)
    ]


class CarlaBboxProjectionTests(unittest.TestCase):
    def test_camera_intrinsics_use_horizontal_fov(self):
        actual = camera_intrinsics(800, 600, 90)
        for value, expected in zip(actual, (400.0, 400.0, 400.0, 300.0)):
            self.assertAlmostEqual(value, expected)

    def test_projects_normalized_visible_box(self):
        bbox = project_world_vertices(
            _box_vertices(),
            world_to_camera_matrix=IDENTITY,
            image_width=800,
            image_height=600,
            fov_deg=90,
        )
        self.assertEqual(bbox, [0.45, 0.433333, 0.55, 0.566667])

    def test_rejects_vertices_behind_camera(self):
        bbox = project_world_vertices(
            _box_vertices(x=-2.0),
            world_to_camera_matrix=IDENTITY,
            image_width=800,
            image_height=600,
            fov_deg=90,
        )
        self.assertIsNone(bbox)

    def test_project_actor_uses_world_vertices_and_camera_inverse(self):
        bbox = project_actor_bbox(
            _Actor(42, _box_vertices()),
            _Camera(),
            image_width=800,
            image_height=600,
            fov_deg=90,
        )
        self.assertEqual(bbox, [0.45, 0.433333, 0.55, 0.566667])

    def test_traffic_light_uses_light_boxes_instead_of_actor_box(self):
        actor = _Actor(
            7,
            _box_vertices(half_y=5.0, half_z=5.0),
            type_id="traffic.traffic_light",
            light_boxes=[_BoundingBox(_box_vertices(half_y=0.5, half_z=0.5))],
        )
        bbox = project_actor_bbox(
            actor,
            _Camera(),
            image_width=800,
            image_height=600,
            fov_deg=90,
        )
        self.assertEqual(bbox, [0.475, 0.466667, 0.525, 0.533333])

    def test_traffic_light_components_stay_separate(self):
        actor = _Actor(
            7,
            _box_vertices(half_y=5.0, half_z=5.0),
            type_id="traffic.traffic_light",
            light_boxes=[
                _BoundingBox(_box_vertices(half_y=0.5, half_z=0.5)),
                _BoundingBox(_box_vertices(x=20.0, half_y=0.5, half_z=0.5)),
            ],
        )
        boxes = project_actor_bboxes(
            actor,
            _Camera(),
            image_width=800,
            image_height=600,
            fov_deg=90,
        )
        self.assertEqual(len(boxes), 2)
        self.assertEqual(boxes[0], [0.475, 0.466667, 0.525, 0.533333])
        self.assertEqual(boxes[1], [0.4875, 0.483333, 0.5125, 0.516667])

    def test_world_state_projection_keeps_world_object_identity(self):
        state = {
            "frame_id": "carla_000123",
            "objects": [
                {
                    "object_id": "carla_actor_42",
                    "source_object_id": "42",
                    "category": "vehicle",
                }
            ],
        }
        result = project_world_state_objects(
            state,
            [_Actor(42, _box_vertices())],
            _Camera(),
            camera_name="front_rgb",
            image_width=800,
            image_height=600,
            fov_deg=90,
        )
        self.assertEqual(result["frame_id"], "carla_000123")
        self.assertEqual(result["objects"][0]["world_object_id"], "carla_actor_42")
        self.assertEqual(result["objects"][0]["category"], "vehicle")

    def test_world_state_projection_assigns_unique_component_ids(self):
        state = {
            "frame_id": "carla_000123",
            "objects": [
                {
                    "object_id": "carla_actor_7",
                    "source_object_id": "7",
                    "category": "traffic_light",
                }
            ],
        }
        actor = _Actor(
            7,
            _box_vertices(),
            type_id="traffic.traffic_light",
            light_boxes=[
                _BoundingBox(_box_vertices()),
                _BoundingBox(_box_vertices(x=20.0)),
            ],
        )
        result = project_world_state_objects(
            state,
            [actor],
            _Camera(),
            camera_name="front_rgb",
            image_width=800,
            image_height=600,
            fov_deg=90,
        )
        self.assertEqual(
            ["carla_actor_7_component_0", "carla_actor_7_component_1"],
            [item["world_object_id"] for item in result["objects"]],
        )
        self.assertTrue(
            all(
                item["parent_world_object_id"] == "carla_actor_7"
                for item in result["objects"]
            )
        )


if __name__ == "__main__":
    unittest.main()
