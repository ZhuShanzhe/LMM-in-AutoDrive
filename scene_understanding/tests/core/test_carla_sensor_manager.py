import tempfile
import threading
import time
import unittest
from pathlib import Path

from scene_understanding.core.carla_sensor_manager import CarlaSensorManager, SensorEventBuffer


class FakeVector:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z


class FakeOtherActor:
    id = 99


class FakeCollisionEvent:
    frame = 12
    timestamp = 0.6
    other_actor = FakeOtherActor()
    normal_impulse = FakeVector(3, 4, 0)


class FakeMarking:
    def __init__(self, marking_type):
        self.type = marking_type


class FakeLaneEvent:
    frame = 13
    timestamp = 0.65
    crossed_lane_markings = [FakeMarking("LaneMarkingType.Solid")]


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
        self.blueprints.setdefault(type_id, FakeBlueprint(type_id))
        return self.blueprints[type_id]


class FakeSensor:
    def __init__(self, blueprint):
        self.blueprint = blueprint
        self.callback = None
        self.is_alive = True
        self.stop_count = 0
        self.destroy_count = 0

    def listen(self, callback):
        self.callback = callback

    def emit(self, event):
        self.callback(event)

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

    def spawn_actor(self, blueprint, transform, attach_to=None):
        sensor = FakeSensor(blueprint)
        self.spawned.append((sensor, transform, attach_to))
        return sensor


class FakeLocation:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z


class FakeTransform:
    def __init__(self, location=None):
        self.location = location or FakeLocation()


class FakeCarla:
    Location = FakeLocation
    Transform = FakeTransform


class FakeImage:
    frame = 42
    timestamp = 2.1
    width = 800
    height = 600

    def __init__(self, frame=42):
        self.frame = frame
        self.saved_path = None

    def save_to_disk(self, path):
        self.saved_path = path
        Path(path).touch()


class SensorEventBufferTests(unittest.TestCase):
    def test_normalizes_collision_impulse(self):
        buffer = SensorEventBuffer()
        record = buffer.record_collision(FakeCollisionEvent())
        self.assertEqual(record["other_actor_id"], "99")
        self.assertEqual(record["impulse_magnitude_ns"], 5.0)

    def test_normalizes_lane_marking(self):
        buffer = SensorEventBuffer()
        record = buffer.record_lane_invasion(FakeLaneEvent())
        self.assertEqual(record["crossed_lane_markings"], ["solid"])

    def test_drain_keeps_future_events(self):
        buffer = SensorEventBuffer()
        buffer.record_collision(FakeCollisionEvent())
        buffer.record_lane_invasion(FakeLaneEvent())
        first = buffer.drain_through(12)
        second = buffer.drain_through(13)
        self.assertEqual(len(first["collisions"]), 1)
        self.assertEqual(first["lane_invasions"], [])
        self.assertEqual(len(second["lane_invasions"]), 1)


class CarlaSensorManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.world = FakeWorld()
        self.manager = CarlaSensorManager(
            self.world,
            ego_vehicle=object(),
            output_dir=self.temp_dir.name,
            carla_module=FakeCarla,
        )

    def tearDown(self):
        self.manager.destroy()
        self.temp_dir.cleanup()

    def test_setup_spawns_three_sensors(self):
        self.manager.setup()
        self.assertEqual(len(self.world.spawned), 3)
        camera = self.world.library.find("sensor.camera.rgb")
        self.assertEqual(camera.attributes["image_size_x"], "800")
        self.assertEqual(camera.attributes["image_size_y"], "600")

    def test_camera_uses_carla_frame_number(self):
        self.manager.setup()
        image = FakeImage()
        camera_sensor = self.world.spawned[2][0]
        camera_sensor.emit(image)
        latest = self.manager.latest_camera_frame()
        self.assertEqual(latest["frame"], 42)
        self.assertTrue(latest["image_path"].endswith("00000042.png"))
        self.assertTrue(Path(latest["image_path"]).exists())
        self.assertEqual(self.manager.camera_frame(42), latest)
        self.assertIs(self.manager.front_camera_sensor, camera_sensor)

    def test_camera_history_prunes_old_frames(self):
        self.manager.camera_history_size = 2
        self.manager.setup()
        camera_sensor = self.world.spawned[2][0]
        for frame in (41, 42, 43):
            camera_sensor.emit(FakeImage(frame))
        self.assertIsNone(self.manager.camera_frame(41))
        self.assertEqual(self.manager.camera_frame(42)["frame"], 42)
        self.assertEqual(self.manager.camera_frame(43)["frame"], 43)

    def test_camera_frame_filter_skips_unselected_frames(self):
        self.manager.camera_frame_filter = lambda frame: frame % 10 == 0
        self.manager.setup()
        camera_sensor = self.world.spawned[2][0]
        skipped = FakeImage(41)
        selected = FakeImage(50)

        camera_sensor.emit(skipped)
        camera_sensor.emit(selected)

        self.assertIsNone(skipped.saved_path)
        self.assertIsNone(self.manager.camera_frame(41))
        self.assertEqual(self.manager.camera_frame(50)["frame"], 50)
        self.assertIsNotNone(selected.saved_path)

    def test_wait_for_camera_frame_returns_exact_delayed_frame(self):
        self.manager.setup()
        camera_sensor = self.world.spawned[2][0]

        def emit_later():
            time.sleep(0.02)
            camera_sensor.emit(FakeImage(44))

        thread = threading.Thread(target=emit_later)
        thread.start()
        record = self.manager.wait_for_camera_frame(44, timeout_s=0.5)
        thread.join()
        self.assertEqual(record["frame"], 44)

    def test_wait_for_camera_frame_times_out(self):
        self.manager.setup()
        self.assertIsNone(self.manager.wait_for_camera_frame(99, timeout_s=0.01))

    def test_wait_for_camera_frame_rejects_negative_timeout(self):
        with self.assertRaisesRegex(ValueError, "timeout_s"):
            self.manager.wait_for_camera_frame(1, timeout_s=-0.1)

    def test_destroy_is_idempotent(self):
        self.manager.setup()
        sensors = [item[0] for item in self.world.spawned]
        self.manager.destroy()
        self.manager.destroy()
        self.assertTrue(all(sensor.destroy_count == 1 for sensor in sensors))


if __name__ == "__main__":
    unittest.main()
