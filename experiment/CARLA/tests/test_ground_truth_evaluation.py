"""Offline tests for exact-frame truth and shadow evaluation."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


CARLA_ROOT = Path(__file__).resolve().parents[1]
if str(CARLA_ROOT) not in sys.path:
    sys.path.insert(0, str(CARLA_ROOT))

from evaluation.ground_truth import (  # noqa: E402
    FRAME_GROUND_TRUTH_SCHEMA,
    FrameGroundTruthRecorder,
    unique_frames,
    validate_event_ground_truth_contracts,
)
from evaluation.shadow_evaluation import (  # noqa: E402
    evaluate_shadow_records,
)
from emergency_scene_3_events import (  # noqa: E402
    EmergencySceneActorRuntime,
)


class Vector:
    def __init__(
        self,
        x: float,
        y: float,
        z: float = 0.0,
    ) -> None:
        self.x = x
        self.y = y
        self.z = z


class Rotation:
    def __init__(
        self,
        yaw: float = 0.0,
    ) -> None:
        self.pitch = 0.0
        self.yaw = yaw
        self.roll = 0.0


class Transform:
    def __init__(
        self,
        location: Vector,
        yaw: float = 0.0,
    ) -> None:
        self.location = location
        self.rotation = Rotation(yaw)


class Actor:
    def __init__(
        self,
        actor_id: int,
        x: float,
        y: float,
        speed_mps: float,
        lane_id: int,
        role_name: str,
    ) -> None:
        self.id = actor_id
        self.type_id = "vehicle.test"
        self.is_alive = True
        self.attributes = {
            "role_name": role_name,
        }
        self._transform = Transform(
            Vector(x, y),
        )
        self._velocity = Vector(speed_mps, 0.0)
        self._transform.location.lane_id = lane_id

    def get_transform(self) -> Transform:
        return self._transform

    def get_velocity(self) -> Vector:
        return self._velocity

    def get_location(self) -> Vector:
        return self._transform.location


class Waypoint:
    def __init__(
        self,
        lane_id: int,
    ) -> None:
        self.road_id = 1
        self.section_id = 0
        self.lane_id = lane_id
        self.is_junction = False


class CarlaMap:
    def get_waypoint(
        self,
        location: Vector,
        project_to_road: bool = True,
    ) -> Waypoint:
        del project_to_road
        return Waypoint(location.lane_id)


class World:
    def __init__(self) -> None:
        self._map = CarlaMap()

    def get_map(self) -> CarlaMap:
        return self._map


def observed_truth_record(
    frame: int,
    *,
    event_id: str = "event_a",
    allowed: list[str] | None = None,
    forbidden: list[str] | None = None,
) -> dict:
    return {
        "schema_version": FRAME_GROUND_TRUTH_SCHEMA,
        "scene_id": "test_scene",
        "simulation_frame": frame,
        "frame_truth_quality": "OBSERVED",
        "claim_eligible": True,
        "provenance": {
            "model_output_used": False,
            "adjacent_frame_fill_used": False,
        },
        "active_events": [
            {
                "event_id": event_id,
                "truth_quality": "OBSERVED",
                "risk_labels": [
                    "object_ahead",
                ],
                "expected_control": {
                    "allowed_actions": (
                        allowed or ["decelerate"]
                    ),
                    "forbidden_actions": (
                        forbidden or ["accelerate"]
                    ),
                },
            }
        ],
    }


class GroundTruthRecorderTests(unittest.TestCase):
    def test_scene_contracts_expose_truth_limitations(
        self,
    ) -> None:
        scene2 = json.loads(
            (
                CARLA_ROOT
                / "configs"
                / "scene_2_complex_avoidance_8km_runtime.json"
            ).read_text(encoding="utf-8")
        )
        scene3 = json.loads(
            (
                CARLA_ROOT
                / "configs"
                / "scene_3_emergency_6km_runtime.json"
            ).read_text(encoding="utf-8")
        )
        validate_event_ground_truth_contracts(
            scene2["events"]
        )
        validate_event_ground_truth_contracts(
            scene3["events"]
        )
        self.assertEqual(
            [
                event["ground_truth"][
                    "evidence_mode"
                ]
                for event in scene2["events"]
            ].count("observed"),
            0,
        )
        self.assertEqual(
            [
                event["ground_truth"][
                    "evidence_mode"
                ]
                for event in scene3["events"]
            ],
            ["observed"] * 7,
        )

    def test_observed_actor_relation_and_ttc(self) -> None:
        event = {
            "id": "event_a",
            "scenario": "cut_in",
            "ground_truth": {
                "evidence_mode": "observed",
                "actor_roles": ["hazard_vehicle"],
                "risk_labels": ["cut_in_vehicle"],
                "allowed_control_actions": [
                    "decelerate",
                ],
                "forbidden_control_actions": [
                    "accelerate",
                ],
            },
        }
        ego = Actor(
            1,
            0.0,
            0.0,
            10.0,
            -2,
            "hero",
        )
        hazard = Actor(
            2,
            20.0,
            0.0,
            5.0,
            -2,
            "hazard_vehicle",
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "truth.jsonl"
            recorder = FrameGroundTruthRecorder(
                path,
                scene_id="test_scene",
                events=[event],
            )
            record = recorder.record(
                world=World(),
                ego=ego,
                simulation_frame=100,
                timestamp_s=5.0,
                route_s_m=50.0,
                event_states={
                    "event_a": "ACTIVE",
                },
                actor_bindings={
                    "hazard_vehicle": hazard,
                },
            )
            recorder.close()

            self.assertIsNotNone(record)
            self.assertEqual(
                record["frame_truth_quality"],
                "OBSERVED",
            )
            self.assertTrue(record["claim_eligible"])
            relation = record["actors"][
                "hazard_vehicle"
            ]["relation_to_ego"]
            self.assertAlmostEqual(
                relation["longitudinal_m"],
                20.0,
            )
            self.assertAlmostEqual(
                relation["relative_closing_speed_mps"],
                5.0,
            )
            self.assertAlmostEqual(
                relation["time_to_collision_s"],
                4.0,
            )
            persisted = json.loads(
                path.read_text(
                    encoding="utf-8",
                ).strip()
            )
            self.assertFalse(
                persisted["provenance"][
                    "model_output_used"
                ]
            )

    def test_missing_actor_is_not_claim_eligible(self) -> None:
        event = {
            "id": "event_a",
            "ground_truth": {
                "evidence_mode": "observed",
                "actor_roles": ["missing_actor"],
            },
        }
        ego = Actor(
            1,
            0.0,
            0.0,
            0.0,
            -2,
            "hero",
        )
        with tempfile.TemporaryDirectory() as temporary:
            recorder = FrameGroundTruthRecorder(
                Path(temporary) / "truth.jsonl",
                scene_id="test_scene",
                events=[event],
            )
            record = recorder.record(
                world=World(),
                ego=ego,
                simulation_frame=1,
                timestamp_s=0.05,
                route_s_m=1.0,
                event_states={
                    "event_a": "ACTIVE",
                },
                actor_bindings={},
            )
            recorder.close()
        self.assertEqual(
            record["frame_truth_quality"],
            "SCHEDULE_ONLY",
        )
        self.assertFalse(record["claim_eligible"])

    def test_duplicate_frame_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "duplicate simulation_frame",
        ):
            unique_frames(
                [
                    {"simulation_frame": 1},
                    {"simulation_frame": 1},
                ]
            )

    def test_proxy_truth_never_becomes_claim_eligible(
        self,
    ) -> None:
        event = {
            "id": "event_a",
            "ground_truth": {
                "evidence_mode": "proxy",
                "actor_roles": ["proxy_actor"],
                "risk_labels": ["proxy_hazard"],
                "allowed_control_actions": [
                    "decelerate",
                ],
                "forbidden_control_actions": [],
            },
        }
        ego = Actor(
            1,
            0.0,
            0.0,
            5.0,
            -2,
            "hero",
        )
        proxy = Actor(
            2,
            15.0,
            0.0,
            3.0,
            -2,
            "proxy_actor",
        )
        with tempfile.TemporaryDirectory() as temporary:
            recorder = FrameGroundTruthRecorder(
                Path(temporary) / "truth.jsonl",
                scene_id="test_scene",
                events=[event],
            )
            record = recorder.record(
                world=World(),
                ego=ego,
                simulation_frame=2,
                timestamp_s=0.1,
                route_s_m=2.0,
                event_states={
                    "event_a": "ACTIVE",
                },
                actor_bindings={
                    "proxy_actor": proxy,
                },
            )
            recorder.close()
        self.assertEqual(
            record["frame_truth_quality"],
            "PROXY",
        )
        self.assertFalse(record["claim_eligible"])

    def test_truth_contract_requires_physical_evidence(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "requires actor or runtime evidence",
        ):
            validate_event_ground_truth_contracts(
                [
                    {
                        "id": "event_a",
                        "ground_truth": {
                            "evidence_mode": "observed",
                            "risk_labels": ["hazard"],
                            "allowed_control_actions": [
                                "decelerate",
                            ],
                            "forbidden_control_actions": [],
                        },
                    }
                ]
            )

    def test_scene3_runtime_exposes_stable_truth_roles(
        self,
    ) -> None:
        runtime = EmergencySceneActorRuntime.__new__(
            EmergencySceneActorRuntime
        )
        cut_in = object()
        cone = object()
        truck = object()
        crossing = object()
        static = object()
        maintenance = object()
        front = object()
        rear = object()
        runtime._cut_in_actor = cut_in
        runtime._warning_sign = None
        runtime._cone_actors = [cone]
        runtime._work_vehicles = [truck]
        runtime._crossing_worker = crossing
        runtime._worker_actors = [
            static,
            crossing,
        ]
        runtime._maintenance_vehicle = maintenance
        runtime._gap_control_vehicles = {
            "front": front,
            "rear": rear,
        }
        runtime._cut_in_phase = "MERGED"
        runtime._worker_phase = "CROSSING"
        runtime._target_lane_released = True
        runtime._blocked_lane_change_commanded = True
        runtime._gap_release_commanded = True
        runtime._work_zone_exited = False

        bindings = (
            runtime.ground_truth_actor_bindings()
        )
        self.assertIs(
            bindings["scene3_cut_in_vehicle"],
            cut_in,
        )
        self.assertIs(
            bindings["scene3_crossing_worker"],
            crossing,
        )
        self.assertIs(
            bindings["scene3_static_worker"],
            static,
        )
        self.assertIs(
            bindings["scene3_gap_front_vehicle"],
            front,
        )
        state = runtime.ground_truth_runtime_state()
        self.assertEqual(
            state["cut_in_phase"],
            "MERGED",
        )
        self.assertTrue(
            state["target_lane_released"]
        )


class ShadowEvaluationTests(unittest.TestCase):
    def test_model_derived_ground_truth_is_rejected(
        self,
    ) -> None:
        truth = observed_truth_record(1)
        truth["provenance"][
            "model_output_used"
        ] = True
        with self.assertRaisesRegex(
            ValueError,
            "independent from model output",
        ):
            evaluate_shadow_records(
                ground_truth_records=[truth],
                semantic_predictions=[],
                minimum_observed_frames=1,
                minimum_observed_events=1,
            )

    def test_perfect_exact_frame_predictions_are_measured(
        self,
    ) -> None:
        truth = [
            observed_truth_record(frame)
            for frame in (10, 11, 12)
        ]
        semantics = [
            {
                "schema_version": (
                    "SemanticPrediction/1.0.0"
                ),
                "scene_id": "test_scene",
                "simulation_frame": frame,
                "active_event_ids": ["event_a"],
                "risk_labels": ["object_ahead"],
            }
            for frame in (10, 11, 12)
        ]
        controls = [
            {
                "schema_version": (
                    "ControlDecisionShadow/1.0.0"
                ),
                "scene_id": "test_scene",
                "simulation_frame": frame,
                "action": "decelerate",
                "safety_gate_status": "APPROVED",
                "latency_ms": 50.0 + frame,
            }
            for frame in (10, 11, 12)
        ]
        report = evaluate_shadow_records(
            ground_truth_records=truth,
            semantic_predictions=semantics,
            control_decisions=controls,
            minimum_observed_frames=1,
            minimum_observed_events=1,
        )
        semantic = report["semantic_alignment"]
        control = report["control"]
        self.assertEqual(
            semantic["status"],
            "MEASURED",
        )
        self.assertEqual(
            semantic["event_detection"][
                "exact_match_accuracy"
            ],
            1.0,
        )
        self.assertEqual(
            semantic["risk_label_alignment"][
                "micro_f1"
            ],
            1.0,
        )
        self.assertEqual(
            control["action_compatibility_rate"],
            1.0,
        )
        self.assertEqual(
            control[
                "unsafe_action_false_approvals"
            ],
            0,
        )
        self.assertFalse(
            report[
                "closed_loop_actuation_performed"
            ]
        )

    def test_insufficient_coverage_is_explicit(self) -> None:
        truth = [
            observed_truth_record(frame)
            for frame in (1, 2, 3, 4)
        ]
        report = evaluate_shadow_records(
            ground_truth_records=truth,
            semantic_predictions=[
                {
                    "schema_version": (
                        "SemanticPrediction/1.0.0"
                    ),
                    "scene_id": "test_scene",
                    "simulation_frame": 1,
                    "active_event_ids": ["event_a"],
                    "risk_labels": ["object_ahead"],
                }
            ],
            minimum_observed_frames=1,
            minimum_observed_events=1,
            minimum_prediction_coverage=0.95,
        )
        semantic = report["semantic_alignment"]
        self.assertEqual(
            semantic["status"],
            "INSUFFICIENT_EVIDENCE",
        )
        self.assertAlmostEqual(
            semantic["exact_frame_coverage"],
            0.25,
        )

    def test_unsafe_approved_action_is_counted(self) -> None:
        report = evaluate_shadow_records(
            ground_truth_records=[
                observed_truth_record(7),
            ],
            control_decisions=[
                {
                    "schema_version": (
                        "ControlDecisionShadow/1.0.0"
                    ),
                    "scene_id": "test_scene",
                    "simulation_frame": 7,
                    "action": "accelerate",
                    "safety_gate_status": "APPROVED",
                    "latency_ms": 20.0,
                }
            ],
            minimum_observed_frames=1,
            minimum_observed_events=1,
        )
        control = report["control"]
        self.assertEqual(
            control[
                "unsafe_action_false_approvals"
            ],
            1,
        )
        self.assertEqual(
            control[
                "unsafe_action_false_approval_rate"
            ],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
