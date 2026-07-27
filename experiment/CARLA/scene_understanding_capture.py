"""Frame-aligned bridge from the lx CARLA scenarios to scene_understanding."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene_understanding.core.carla_bbox_projection import project_world_state_objects
from scene_understanding.core.carla_sensor_manager import CarlaSensorManager
from scene_understanding.core.carla_world_state import CarlaWorldStateCollector
from scene_understanding.core.prepare_carla_samples import append_jsonl, write_capture_bundle


class SceneUnderstandingCapture:
    """Capture synchronized images, WorldState records, and actor projections."""

    def __init__(
        self,
        world: Any,
        ego_vehicle: Any,
        *,
        output_dir: str | Path,
        every_n_frames: int = 10,
        image_width: int = 800,
        image_height: int = 600,
        fov_deg: float = 90.0,
        camera_timeout_s: float = 1.0,
    ) -> None:
        if every_n_frames <= 0:
            raise ValueError("every_n_frames must be positive")
        if camera_timeout_s < 0:
            raise ValueError("camera_timeout_s must be non-negative")
        self.world = world
        self.output_dir = Path(output_dir).resolve()
        self.every_n_frames = int(every_n_frames)
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.fov_deg = float(fov_deg)
        self.camera_timeout_s = float(camera_timeout_s)
        self.capture_index = self.output_dir / "capture_index.jsonl"
        self.sensors = CarlaSensorManager(
            world,
            ego_vehicle,
            output_dir=self.output_dir / "sensors",
            image_width=self.image_width,
            image_height=self.image_height,
            camera_fov_deg=self.fov_deg,
            camera_history_size=max(32, self.every_n_frames * 4),
            camera_frame_filter=lambda frame: frame % self.every_n_frames == 0,
        )
        self.collector = CarlaWorldStateCollector(world, ego_vehicle)
        self.captured = 0
        self.camera_timeouts = 0

    def setup(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sensors.setup()

    def capture_current_frame(self) -> dict[str, Any] | None:
        """Capture the latest completed CARLA tick when it is a selected keyframe."""

        snapshot = self.world.get_snapshot()
        frame = int(snapshot.frame)
        if frame % self.every_n_frames != 0:
            return None

        camera_record = self.sensors.wait_for_camera_frame(
            frame, timeout_s=self.camera_timeout_s
        )
        if camera_record is None:
            self.camera_timeouts += 1
            return {
                "status": "camera_timeout",
                "simulation_frame": frame,
            }

        events = self.sensors.drain_events_through(frame)
        world_state = self.collector.collect(sensor_events=events)
        projection = project_world_state_objects(
            world_state,
            self.world.get_actors(),
            self.sensors.front_camera_sensor,
            camera_name=camera_record["camera_name"],
            image_width=self.image_width,
            image_height=self.image_height,
            fov_deg=self.fov_deg,
        )
        index_record = write_capture_bundle(
            self.output_dir,
            camera_record=camera_record,
            world_state=world_state,
            projection_record=projection,
        )
        append_jsonl(self.capture_index, index_record)
        self.captured += 1
        return {"status": "captured", **index_record}

    def stats(self) -> dict[str, int]:
        return {
            "captured": self.captured,
            "camera_timeouts": self.camera_timeouts,
        }

    def destroy(self) -> None:
        self.sensors.destroy()
