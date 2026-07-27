import threading
import time
import unittest

from scene_understanding.core.carla_multimodal_sensor_manager import (
    MultimodalSensorFrameBuffer,
    camera_intrinsic_matrix,
    lidar_point_count,
)


def _identity_4x4():
    return [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]


def _camera_record(name, frame, timestamp_s):
    return {
        "sensor_name": name,
        "frame": frame,
        "timestamp_s": timestamp_s,
        "image_path": f"/tmp/{name}/{frame:08d}.png",
        "image_size": {
            "width": 800,
            "height": 600,
        },
        "intrinsic_matrix": [
            400.0, 0.0, 400.0,
            0.0, 400.0, 300.0,
            0.0, 0.0, 1.0,
        ],
        "sensor_to_ego": _identity_4x4(),
    }


def _lidar_record(frame, timestamp_s):
    return {
        "sensor_name": "roof_lidar",
        "frame": frame,
        "timestamp_s": timestamp_s,
        "point_cloud_path": (
            f"/tmp/roof_lidar/{frame:08d}.bin"
        ),
        "point_count": 128,
        "coordinate_frame": "sensor",
        "sensor_to_ego": _identity_4x4(),
    }


class MultimodalSensorFrameBufferTests(unittest.TestCase):
    def test_never_mixes_lidar_from_adjacent_frame(self):
        buffer = MultimodalSensorFrameBuffer(
            required_camera_names=(
                "front_rgb",
                "left_rgb",
            ),
            require_lidar=True,
        )

        buffer.add_camera(
            _camera_record("front_rgb", 100, 5.00)
        )
        buffer.add_camera(
            _camera_record("left_rgb", 100, 5.01)
        )

        # 第 101 帧雷达不能补到第 100 帧。
        buffer.add_lidar(
            _lidar_record(101, 5.05)
        )

        snapshot = buffer.snapshot(100)

        self.assertFalse(snapshot["complete"])
        self.assertEqual(
            snapshot["missing_modalities"],
            ["lidar"],
        )
        self.assertIsNone(snapshot["lidar"])
        self.assertEqual(
            [
                item["frame"]
                for item in snapshot["cameras"]
            ],
            [100, 100],
        )

        buffer.add_lidar(
            _lidar_record(100, 5.02)
        )

        snapshot = buffer.snapshot(100)

        self.assertTrue(snapshot["complete"])
        self.assertEqual(
            snapshot["missing_modalities"],
            [],
        )
        self.assertEqual(
            snapshot["lidar"]["frame"],
            100,
        )

    def test_never_uses_adjacent_camera_frame(self):
        buffer = MultimodalSensorFrameBuffer(
            required_camera_names=("front_rgb",),
            require_lidar=True,
        )

        buffer.add_camera(
            _camera_record("front_rgb", 199, 9.95)
        )
        buffer.add_camera(
            _camera_record("front_rgb", 201, 10.05)
        )
        buffer.add_lidar(
            _lidar_record(200, 10.00)
        )

        snapshot = buffer.snapshot(200)

        self.assertFalse(snapshot["complete"])
        self.assertEqual(
            snapshot["missing_modalities"],
            ["front_rgb"],
        )
        self.assertEqual(snapshot["cameras"], [])
        self.assertEqual(
            snapshot["lidar"]["frame"],
            200,
        )

    def test_camera_output_order_is_deterministic(self):
        buffer = MultimodalSensorFrameBuffer(
            required_camera_names=(
                "front_rgb",
                "left_rgb",
                "right_rgb",
            ),
            require_lidar=False,
        )

        # 故意以相反顺序写入。
        buffer.add_camera(
            _camera_record("right_rgb", 300, 15.02)
        )
        buffer.add_camera(
            _camera_record("left_rgb", 300, 15.01)
        )
        buffer.add_camera(
            _camera_record("front_rgb", 300, 15.00)
        )

        snapshot = buffer.snapshot(300)

        self.assertTrue(snapshot["complete"])
        self.assertEqual(
            [
                item["sensor_name"]
                for item in snapshot["cameras"]
            ],
            [
                "front_rgb",
                "left_rgb",
                "right_rgb",
            ],
        )

    def test_rejects_undeclared_camera(self):
        buffer = MultimodalSensorFrameBuffer(
            required_camera_names=("front_rgb",),
            require_lidar=False,
        )

        with self.assertRaises(ValueError):
            buffer.add_camera(
                _camera_record("rear_rgb", 400, 20.0)
            )

    def test_camera_intrinsic_matrix_for_90_degree_fov(self):
        matrix = camera_intrinsic_matrix(
            width_px=800,
            height_px=600,
            horizontal_fov_degrees=90.0,
        )

        self.assertEqual(len(matrix), 9)
        self.assertAlmostEqual(matrix[0], 400.0)
        self.assertAlmostEqual(matrix[2], 400.0)
        self.assertAlmostEqual(matrix[4], 400.0)
        self.assertAlmostEqual(matrix[5], 300.0)
        self.assertEqual(
            matrix[6:],
            [0.0, 0.0, 1.0],
        )

    def test_lidar_point_count_uses_xyzi_float32_layout(self):
        # CARLA 每个 LiDAR 点为 x、y、z、intensity，
        # 共 4 个 float32，即每点 16 字节。
        self.assertEqual(
            lidar_point_count(bytes(16 * 25)),
            25,
        )

        with self.assertRaises(ValueError):
            lidar_point_count(bytes(17))

    def test_waits_for_async_exact_frame(self):
        buffer = MultimodalSensorFrameBuffer(
            required_camera_names=("front_rgb",),
            require_lidar=True,
        )

        def producer():
            time.sleep(0.02)
            buffer.add_camera(
                _camera_record("front_rgb", 500, 25.00)
            )
            buffer.add_lidar(
                _lidar_record(500, 25.01)
            )

        thread = threading.Thread(
            target=producer,
            daemon=True,
        )
        thread.start()

        snapshot = buffer.wait_for_snapshot(
            500,
            timeout_s=1.0,
        )

        thread.join(timeout=1.0)

        self.assertFalse(thread.is_alive())
        self.assertTrue(snapshot["complete"])
        self.assertEqual(
            snapshot["missing_modalities"],
            [],
        )
        self.assertEqual(
            snapshot["cameras"][0]["frame"],
            500,
        )
        self.assertEqual(
            snapshot["lidar"]["frame"],
            500,
        )

    def test_evicts_oldest_frame_when_capacity_is_exceeded(self):
        buffer = MultimodalSensorFrameBuffer(
            required_camera_names=("front_rgb",),
            require_lidar=False,
            max_frames=2,
        )

        for frame in (600, 601, 602):
            buffer.add_camera(
                _camera_record(
                    "front_rgb",
                    frame,
                    frame / 20.0,
                )
            )

        self.assertEqual(
            buffer.available_frames(),
            [601, 602],
        )

        evicted = buffer.snapshot(600)

        self.assertFalse(evicted["complete"])
        self.assertEqual(
            evicted["missing_modalities"],
            ["front_rgb"],
        )
        self.assertEqual(evicted["cameras"], [])


if __name__ == "__main__":
    unittest.main()
