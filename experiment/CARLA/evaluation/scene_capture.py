"""Optional exact-frame RGB/LiDAR recording for formal CARLA runners."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evaluation.multimodal import ExactFrameSensorSuite, REQUIRED_SENSOR_NAMES
from scene2_runtime_interface import build_multimodal_frame_bundle


class _SensorRegistry:
    def __init__(self) -> None:
        self.actors: list[Any] = []

    def add(self, actor: Any) -> None:
        self.actors.append(actor)

    def close(self) -> list[str]:
        errors = []
        for actor in reversed(self.actors):
            try:
                if getattr(actor, "is_listening", False):
                    actor.stop()
            except (RuntimeError, OSError) as error:
                errors.append(str(error))
            try:
                actor.destroy()
            except (RuntimeError, OSError) as error:
                errors.append(str(error))
        self.actors.clear()
        return errors


class SceneCaptureSession:
    """Write replay inputs; independent task truth remains under benchmark/."""

    def __init__(self, world: Any, ego: Any, output_dir: Path, scene_id: str,
                 fixed_delta_s: float, stride: int = 10, timeout_s: float = 2.0,
                 image_width: int = 960, image_height: int = 540) -> None:
        if stride < 1 or fixed_delta_s <= 0 or timeout_s <= 0:
            raise ValueError("invalid synchronized capture cadence or timeout")
        self.world = world
        self.ego = ego
        self.root = Path(output_dir)
        self.scene_id = scene_id
        self.stride = int(stride)
        self.timeout_s = float(timeout_s)
        self.registry = _SensorRegistry()
        self.sensors = ExactFrameSensorSuite(
            world, ego, self.registry, self.root, fixed_delta_s * stride,
            image_width=image_width, image_height=image_height,
        )
        self.phase_frame: int | None = None
        self._states = None
        self._bundles = None
        self.recorded = 0
        self._failed = False
        self._closed = False

    def start(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for name in ("world_state.jsonl", "multimodal_frame_bundle.jsonl",
                     "sensor_calibration.json", "interface_manifest.json", "capture_summary.json"):
            if (self.root / name).exists():
                raise FileExistsError(f"capture output already exists: {self.root / name}")
        try:
            self.sensors.start()
            warmup = [int(self.world.tick()) for _ in range(self.stride * 3 + 2)]
            self.phase_frame, observed = self.sensors.wait_for_stable_phase(
                warmup, self.stride, self.timeout_s,
            )
            if self.phase_frame is None:
                raise RuntimeError(f"{self.scene_id}: sensor cadence did not stabilize: {observed}")
            self._states = (self.root / "world_state.jsonl").open("x", encoding="utf-8")
            self._bundles = (self.root / "multimodal_frame_bundle.jsonl").open("x", encoding="utf-8")
            (self.root / "interface_manifest.json").write_text(json.dumps({
                "scene_id": self.scene_id,
                "schema_version": "carla_exact_frame_capture/1.0",
                "sensor_stride_frames": self.stride,
                "phase_frame": self.phase_frame,
                "driving_intent_source": "not_recorded; command schedule is separate",
                "evaluator_truth_location": "separate run benchmark/episode_truth.jsonl",
                "policy_state_contract": "ego_telemetry_allowlist/1.0",
            }, indent=2) + "\n", encoding="utf-8")
        except Exception:
            self.close()
            raise

    def observe(self, snapshot: Any, route_s_m: float) -> bool:
        if self.phase_frame is None or self._states is None or self._bundles is None:
            raise RuntimeError("capture session is not started")
        frame = int(snapshot.frame)
        if frame < self.phase_frame or (frame - self.phase_frame) % self.stride:
            return False
        complete, sensor_frames = self.sensors.wait_for_frame(frame, self.timeout_s)
        if not complete:
            self._failed = True
            raise RuntimeError(f"{self.scene_id}: incomplete synchronized sensors at frame {frame}: {sensor_frames}")
        transform = self.ego.get_transform()
        velocity = self.ego.get_velocity()
        acceleration = self.ego.get_acceleration()
        angular = self.ego.get_angular_velocity()
        control = self.ego.get_control()

        def xyz(value: Any) -> dict[str, float]:
            return {name: float(getattr(value, name)) for name in ("x", "y", "z")}

        state = {
            "schema_version": "1.0.0", "scene_id": self.scene_id,
            "simulation_frame": frame,
            "timestamp_s": float(snapshot.timestamp.elapsed_seconds),
            "route_s_m": float(route_s_m),
            "ego": {
                "actor_id": int(self.ego.id),
                "speed_kmh": 3.6 * sum(value * value for value in xyz(velocity).values()) ** .5,
                "location": xyz(transform.location),
                "rotation": {name: float(getattr(transform.rotation, name))
                             for name in ("pitch", "yaw", "roll")},
                "velocity_mps": xyz(velocity), "acceleration_mps2": xyz(acceleration),
                "angular_velocity_deg_s": xyz(angular),
                "control": {"throttle": float(control.throttle), "brake": float(control.brake),
                            "steer": float(control.steer), "reverse": bool(control.reverse),
                            "hand_brake": bool(control.hand_brake), "gear": int(control.gear)},
            },
        }
        bundle = build_multimodal_frame_bundle(self.scene_id, frame, frame, sensor_frames, None)
        if bundle["status"] != "COMPLETE":
            self._failed = True
            raise RuntimeError(f"{self.scene_id}: sensor barrier disagrees with bundle at frame {frame}")
        bundle["timestamp_s"] = state["timestamp_s"]
        bundle["artifacts"] = {
            name: {"path": (f"rgb/{name}/{frame:08d}.png" if name != "lidar"
                            else f"lidar/{frame:08d}.ply")}
            for name in REQUIRED_SENSOR_NAMES
        }
        bundle["artifacts"]["lidar_raw"] = {
            "path": f"lidar/{frame:08d}.xyzi.bin", "encoding": "float32_le_xyzi",
        }
        try:
            self._states.write(json.dumps(state, allow_nan=False) + "\n")
            self._bundles.write(json.dumps(bundle, allow_nan=False) + "\n")
            self._states.flush()
            self._bundles.flush()
        except Exception:
            self._failed = True
            raise
        self.recorded += 1
        return True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._states is not None:
            self._states.close()
            self._states = None
        if self._bundles is not None:
            self._bundles.close()
            self._bundles = None
        errors = self.registry.close()
        if self.phase_frame is not None:
            (self.root / "capture_summary.json").write_text(json.dumps({
                "schema_version": "carla_exact_frame_capture_summary/1.0",
                "scene_id": self.scene_id, "recorded_frames": self.recorded,
                "status": "RECORDED" if self.recorded and not self._failed and not errors else "INVALID",
                "errors": errors,
                "synchronization": self.sensors.summary(),
                "policy_truth_separated": True,
            }, indent=2) + "\n", encoding="utf-8")
