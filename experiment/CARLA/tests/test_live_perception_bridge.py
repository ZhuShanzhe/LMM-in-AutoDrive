import unittest

from control.live_perception_bridge import _scene_output
from scene_understanding.core.validate_scene_output import validate_scene_output


class LivePerceptionBridgeTests(unittest.TestCase):
    def test_detector_tracks_adapt_to_valid_scene_contract(self):
        perception = {
            "frame_id": "carla_10",
            "camera_name": "front_rgb",
            "tracks": [{
                "category": "vehicle",
                "subtype": "car",
                "bbox_2d": [0.2, 0.3, 0.5, 0.7],
                "confidence": 0.9,
            }],
        }
        world_state = {
            "environment": {
                "weather": "clear",
                "road_type": "urban",
                "is_intersection": False,
                "visibility": "good",
            }
        }
        output = _scene_output(perception, world_state)
        self.assertEqual(
            [],
            validate_scene_output(
                output,
                expected_frame_id="carla_10",
                expected_source="carla",
                expected_camera_name="front_rgb",
            ),
        )


if __name__ == "__main__":
    unittest.main()
