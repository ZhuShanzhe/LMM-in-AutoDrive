from __future__ import annotations

import json
import unittest
from copy import deepcopy
from pathlib import Path

from scene_understanding.core.visual_semantic_fusion import fuse_visual_semantics
from scene_understanding.core.world_state import validate_world_state


EXAMPLE = Path("scene_understanding/schemas/examples/world_state.example.json")


def _world_state():
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def _visual_object(object_id="vlm_obj_001", bbox=None, confidence=0.9):
    return {
        "object_id": object_id,
        "category": "vehicle",
        "subtype": "car",
        "color": "red",
        "bbox_2d": bbox or [0.41, 0.36, 0.59, 0.69],
        "relative_position": "front",
        "lane_relation": "ego_lane",
        "motion_state": "unknown",
        "distance_level": "medium",
        "occlusion": "none",
        "confidence": confidence,
    }


def _inference(objects=None, frame_id="carla_000123"):
    output = {
        "schema_version": "1.0",
        "frame_id": frame_id,
        "source": "carla",
        "camera_name": "front_rgb",
        "scene": {
            "summary": "A red vehicle is visible ahead.",
            "road_type": "urban",
            "is_intersection": False,
            "weather": "clear",
            "visibility": "good",
            "traffic_light_state": "not_visible",
            "left_lane_marking": "dashed",
            "right_lane_marking": "solid",
        },
        "objects": objects if objects is not None else [_visual_object()],
        "potential_hazards": [],
    }
    return {
        "frame_id": frame_id,
        "source": "carla",
        "camera_name": "front_rgb",
        "status": "valid",
        "inference_config": {"model_path": "/models/Qwen2.5-VL-3B-Instruct"},
        "parsed_output": output,
    }


def _projection(frame_id="carla_000123"):
    return {
        "schema_version": "1.0",
        "frame_id": frame_id,
        "camera_name": "front_rgb",
        "image_width": 800,
        "image_height": 600,
        "objects": [
            {
                "world_object_id": "carla_actor_42",
                "source_object_id": "42",
                "category": "vehicle",
                "bbox_2d": [0.4, 0.35, 0.6, 0.7],
            }
        ],
    }


class VisualSemanticFusionTests(unittest.TestCase):
    def test_fuses_match_without_changing_metric_truth(self):
        world = _world_state()
        original_distance = world["objects"][0]["distance_m"]
        enriched, audit = fuse_visual_semantics(world, _inference(), _projection())
        self.assertEqual(audit["matched_count"], 1)
        self.assertEqual(enriched["objects"][0]["distance_m"], original_distance)
        matches = enriched["objects"][0]["semantic_matches"]
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["visual_object_id"], "vlm_obj_001")
        self.assertIn("red", matches[0]["description"])
        self.assertEqual(
            enriched["provenance"]["semantic_source"],
            "Qwen2.5-VL-3B-Instruct",
        )
        self.assertEqual(validate_world_state(enriched), [])

    def test_replaces_same_camera_semantics_idempotently(self):
        enriched, _ = fuse_visual_semantics(_world_state(), _inference(), _projection())
        enriched_again, _ = fuse_visual_semantics(enriched, _inference(), _projection())
        self.assertEqual(len(enriched_again["objects"][0]["semantic_matches"]), 1)

    def test_component_projection_maps_back_to_parent_world_object(self):
        projection = _projection()
        projection["objects"][0].update(
            {
                "world_object_id": "carla_actor_42_component_0",
                "parent_world_object_id": "carla_actor_42",
                "component_index": 0,
            }
        )
        enriched, audit = fuse_visual_semantics(
            _world_state(), _inference(), projection
        )
        self.assertEqual(audit["matched_count"], 1)
        self.assertEqual(audit["matches"][0]["world_object_id"], "carla_actor_42")
        self.assertEqual(
            audit["matches"][0]["projection_object_id"],
            "carla_actor_42_component_0",
        )
        self.assertEqual(len(enriched["objects"][0]["semantic_matches"]), 1)

    def test_one_truth_object_accepts_only_one_prediction(self):
        objects = [
            _visual_object("vlm_obj_001", [0.41, 0.36, 0.59, 0.69], 0.8),
            _visual_object("vlm_obj_002", [0.42, 0.37, 0.58, 0.68], 0.9),
        ]
        _, audit = fuse_visual_semantics(_world_state(), _inference(objects), _projection())
        self.assertEqual(audit["matched_count"], 1)
        self.assertEqual(len(audit["unmatched_visual_object_ids"]), 1)

    def test_low_confidence_prediction_is_not_fused(self):
        inference = _inference([_visual_object(confidence=0.2)])
        enriched, audit = fuse_visual_semantics(
            _world_state(), inference, _projection(), min_confidence=0.5
        )
        self.assertEqual(audit["matched_count"], 0)
        self.assertEqual(enriched["objects"][0]["semantic_matches"], [])

    def test_rejects_cross_frame_fusion(self):
        with self.assertRaisesRegex(ValueError, "frame_id"):
            fuse_visual_semantics(
                _world_state(),
                _inference(frame_id="carla_000124"),
                _projection(),
            )

    def test_rejects_invalid_inference_status(self):
        inference = deepcopy(_inference())
        inference["status"] = "invalid"
        with self.assertRaisesRegex(ValueError, "status='valid'"):
            fuse_visual_semantics(_world_state(), inference, _projection())

    def test_rejects_projection_for_unknown_world_object(self):
        projection = _projection()
        projection["objects"][0]["world_object_id"] = "carla_actor_999"
        with self.assertRaisesRegex(ValueError, "unknown WorldState object"):
            fuse_visual_semantics(_world_state(), _inference(), projection)

    def test_rejects_projection_category_mismatch(self):
        projection = _projection()
        projection["objects"][0]["category"] = "pedestrian"
        with self.assertRaisesRegex(ValueError, "category mismatch"):
            fuse_visual_semantics(_world_state(), _inference(), projection)


if __name__ == "__main__":
    unittest.main()
