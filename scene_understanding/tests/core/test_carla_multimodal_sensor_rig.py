import json
import tempfile
import unittest
from pathlib import Path

from scene_understanding.core.carla_multimodal_sensor_manager import (
    CarlaMultimodalSensorManager,
    transform_to_matrix,
)

from scene_understanding.core.multimodal_frame_bundle import (
    validate_multimodal_frame_bundle,
)


class FakeBlueprint:
    def __init__(self, type_id):
        self.id = type_id
        self.attributes = {}

    def set_attribute(self, key, value):
        self.attributes[key] = value


class FakeBlueprintLibrary:
    def __init__(self):
        self.blueprints = {}

    def find(self, type_id):
        self.blueprints.setdefault(
            type_id,
            FakeBlueprint(type_id),
        )
        return self.blueprints[type_id]


class FakeSensor:
    def __init__(self, blueprint, transform):
        self.blueprint = blueprint
        self.transform = transform
        self.callback = None
        self.is_alive = True
        self.stop_count = 0
        self.destroy_count = 0

    def listen(self, callback):
        self.callback = callback

    def emit(self, data):
        self.callback(data)

    def stop(self):
        self.stop_count += 1

    def destroy(self):
        self.destroy_count += 1
        self.is_alive = False


class FakeWorld:
    def __init__(self):
        self.library = FakeBlueprintLibrary()
        self.spawned = []

    def get_blueprint_library(self):
        return self.library

    def spawn_actor(
        self,
        blueprint,
        transform,
        attach_to=None,
    ):
        sensor = FakeSensor(blueprint, transform)
        self.spawned.append(
            (sensor, transform, attach_to)
        )
        return sensor


class FakeLocation:
    def __init__(
        self,
        x=0.0,
        y=0.0,
        z=0.0,
    ):
        self.x = x
        self.y = y
        self.z = z


class FakeRotation:
    def __init__(
        self,
        pitch=0.0,
        yaw=0.0,
        roll=0.0,
    ):
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll


class FakeTransform:
    def __init__(
        self,
        location=None,
        rotation=None,
    ):
        self.location = location or FakeLocation()
        self.rotation = rotation or FakeRotation()


class FakeCarla:
    Location = FakeLocation
    Rotation = FakeRotation
    Transform = FakeTransform


class FakeImage:
    def __init__(
        self,
        frame,
        timestamp_s,
        width=800,
        height=600,
    ):
        self.frame = frame
        self.timestamp = timestamp_s
        self.width = width
        self.height = height
        self.saved_path = None

    def save_to_disk(self, path):
        self.saved_path = path
        Path(path).touch()


class FakeLidarMeasurement:
    def __init__(
        self,
        frame,
        timestamp_s,
        point_count=5,
    ):
        self.frame = frame
        self.timestamp = timestamp_s
        self.raw_data = bytes(16 * point_count)


class CarlaMultimodalSensorManagerTests(
    unittest.TestCase
):
    CAMERA_NAMES = (
        "front_rgb",
        "left_rgb",
        "right_rgb",
        "rear_rgb",
    )

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.world = FakeWorld()
        self.ego_vehicle = object()
        self.manager = CarlaMultimodalSensorManager(
            self.world,
            self.ego_vehicle,
            output_dir=self.temp_dir.name,
            carla_module=FakeCarla,
        )

    def tearDown(self):
        self.manager.destroy()
        self.temp_dir.cleanup()

    def test_setup_spawns_four_cameras_and_lidar(self):
        self.manager.setup()

        self.assertEqual(len(self.world.spawned), 5)

        for name in self.CAMERA_NAMES:
            self.assertIsNotNone(
                self.manager.sensor_actor(name)
            )

        self.assertIsNotNone(
            self.manager.sensor_actor("roof_lidar")
        )

        camera_bp = self.world.library.find(
            "sensor.camera.rgb"
        )
        self.assertEqual(
            camera_bp.attributes["image_size_x"],
            "800",
        )
        self.assertEqual(
            camera_bp.attributes["image_size_y"],
            "600",
        )
        self.assertEqual(
            camera_bp.attributes["fov"],
            "90.0",
        )

        lidar_bp = self.world.library.find(
            "sensor.lidar.ray_cast"
        )
        self.assertEqual(
            lidar_bp.attributes["channels"],
            "32",
        )
        self.assertEqual(
            lidar_bp.attributes["range"],
            "100.0",
        )

    def test_callbacks_create_complete_contract_records(self):
        self.manager.setup()

        frame = 700
        timestamp_s = 35.0

        for name in self.CAMERA_NAMES:
            self.manager.sensor_actor(name).emit(
                FakeImage(frame, timestamp_s)
            )

        self.manager.sensor_actor(
            "roof_lidar"
        ).emit(
            FakeLidarMeasurement(
                frame,
                timestamp_s,
                point_count=5,
            )
        )

        snapshot = self.manager.snapshot(frame)

        self.assertTrue(snapshot["complete"])
        self.assertEqual(
            snapshot["missing_modalities"],
            [],
        )
        self.assertEqual(
            [
                record["sensor_name"]
                for record in snapshot["cameras"]
            ],
            list(self.CAMERA_NAMES),
        )

        camera = snapshot["cameras"][0]
        self.assertEqual(
            set(camera),
            {
                "sensor_name",
                "frame",
                "timestamp_s",
                "image_path",
                "image_size",
                "intrinsic_matrix",
                "sensor_to_ego",
            },
        )
        self.assertEqual(camera["frame"], frame)
        self.assertEqual(
            camera["image_size"],
            {"width": 800, "height": 600},
        )
        self.assertEqual(
            len(camera["intrinsic_matrix"]),
            9,
        )
        self.assertEqual(
            len(camera["sensor_to_ego"]),
            16,
        )
        self.assertTrue(
            Path(camera["image_path"]).exists()
        )

        lidar = snapshot["lidar"]
        self.assertEqual(
            set(lidar),
            {
                "sensor_name",
                "frame",
                "timestamp_s",
                "point_cloud_path",
                "point_count",
                "coordinate_frame",
                "sensor_to_ego",
            },
        )
        self.assertEqual(lidar["frame"], frame)
        self.assertEqual(lidar["point_count"], 5)
        self.assertEqual(
            lidar["coordinate_frame"],
            "sensor",
        )
        self.assertTrue(
            Path(lidar["point_cloud_path"]).exists()
        )
        self.assertEqual(
            Path(
                lidar["point_cloud_path"]
            ).stat().st_size,
            16 * 5,
        )

    def test_manager_never_substitutes_adjacent_lidar(self):
        self.manager.setup()

        for name in self.CAMERA_NAMES:
            self.manager.sensor_actor(name).emit(
                FakeImage(800, 40.0)
            )

        self.manager.sensor_actor(
            "roof_lidar"
        ).emit(
            FakeLidarMeasurement(801, 40.05)
        )

        snapshot = self.manager.snapshot(800)

        self.assertFalse(snapshot["complete"])
        self.assertEqual(
            snapshot["missing_modalities"],
            ["lidar"],
        )
        self.assertIsNone(snapshot["lidar"])

    def test_sensor_records_validate_in_bundle_contract(
        self,
    ):
        self.manager.setup()

        frame = 123
        timestamp_s = 6.15

        for name in self.CAMERA_NAMES:
            self.manager.sensor_actor(name).emit(
                FakeImage(frame, timestamp_s)
            )

        self.manager.sensor_actor(
            "roof_lidar"
        ).emit(
            FakeLidarMeasurement(
                frame,
                timestamp_s,
                point_count=5,
            )
        )

        snapshot = self.manager.snapshot(frame)
        self.assertTrue(snapshot["complete"])

        scene_root = Path(__file__).resolve().parents[2]
        example_path = (
            scene_root
            / "schemas"
            / "examples"
            / "multimodal_frame_bundle.example.json"
        )

        bundle = json.loads(
            example_path.read_text(encoding="utf-8")
        )
        bundle["cameras"] = snapshot["cameras"]
        bundle["lidar"] = snapshot["lidar"]

        errors = validate_multimodal_frame_bundle(
            bundle
        )

        self.assertEqual(errors, [])

    def test_transform_matrix_contains_pose(self):
        transform = FakeTransform(
            FakeLocation(x=1.5, y=-0.5, z=2.4),
            FakeRotation(yaw=90.0),
        )

        matrix = transform_to_matrix(transform)

        self.assertEqual(len(matrix), 16)
        self.assertAlmostEqual(matrix[0], 0.0)
        self.assertAlmostEqual(matrix[1], -1.0)
        self.assertAlmostEqual(matrix[3], 1.5)
        self.assertAlmostEqual(matrix[4], 1.0)
        self.assertAlmostEqual(matrix[5], 0.0)
        self.assertAlmostEqual(matrix[7], -0.5)
        self.assertAlmostEqual(matrix[11], 2.4)
        self.assertEqual(
            matrix[12:],
            [0.0, 0.0, 0.0, 1.0],
        )


if __name__ == "__main__":
    unittest.main()
