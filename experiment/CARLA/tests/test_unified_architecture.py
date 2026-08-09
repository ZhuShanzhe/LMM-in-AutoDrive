"""Unified sensor-VLA architecture tests across the three CARLA scenes."""

from __future__ import annotations

import json
import math
import pathlib
import sys
from types import SimpleNamespace

import pytest
import torch

import carla


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
EXPERIMENT_CARLA = REPO_ROOT / "experiment" / "CARLA"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EXPERIMENT_CARLA) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_CARLA))

from lightweight_vla_adapter.scripts.run_offline_inference import build_model
from lightweight_vla_adapter.src.contracts import SensorTensorBatch
from lightweight_vla_adapter.src.pipeline import LightweightVLAPipeline
from lightweight_vla_adapter.src.unified_sensor_batch import (
    CAMERA_VIEW_NAMES,
    UNIFIED_SENSOR_BATCH_SCHEMA_VERSION,
    UnifiedSensorBatch,
    default_modality_mask,
)


def make_batch(
    *,
    front: bool,
    left: bool,
    right: bool,
    rear: bool,
    lidar: bool,
    schema_version: str = UNIFIED_SENSOR_BATCH_SCHEMA_VERSION,
) -> UnifiedSensorBatch:
    batch = 1
    height, width = 16, 16
    mask = default_modality_mask(
        text=True,
        front_rgb=front,
        left_rgb=left,
        right_rgb=right,
        rear_rgb=rear,
        lidar_bev=lidar,
        vehicle_state=True,
        environment_state=True,
    )
    camera_view_mask = torch.tensor(
        [[front, left, right, rear]], dtype=torch.bool
    )
    rgb = {
        name: (
            torch.randint(0, 256, (batch, 3, height, width), dtype=torch.uint8)
            if available
            else torch.zeros(batch, 3, height, width, dtype=torch.uint8)
        )
        for name, available in zip(CAMERA_VIEW_NAMES, (front, left, right, rear))
    }
    return UnifiedSensorBatch(
        schema_version=schema_version,
        text_tokens=torch.randn(batch, 8, 768),
        text_mask=torch.ones(batch, 8, dtype=torch.bool),
        front_rgb=rgb["front"],
        left_rgb=rgb["left"],
        right_rgb=rgb["right"],
        rear_rgb=rgb["rear"],
        lidar_bev=(
            torch.rand(batch, 4, 64, 64)
            if lidar
            else torch.zeros(batch, 4, 64, 64)
        ),
        vehicle_state=torch.randn(batch, 8),
        environment_state=torch.randn(batch, 14),
        camera_view_mask=camera_view_mask,
        modality_mask=mask,
        frame_id="test_frame",
        timestamp_s=1.0,
    )


SCENE_BATCH_KWARGS = {
    "scene1": dict(front=True, left=False, right=False, rear=False, lidar=False),
    "scene2": dict(front=True, left=True, right=True, rear=True, lidar=True),
    "scene3": dict(front=True, left=True, right=True, rear=True, lidar=False),
}


def test_three_scenes_construct_same_unified_batch_fields():
    batches = {
        name: make_batch(**kwargs) for name, kwargs in SCENE_BATCH_KWARGS.items()
    }
    field_names = {
        name: set(batch.__dataclass_fields__)
        for name, batch in batches.items()
    }
    assert field_names["scene1"] == field_names["scene2"] == field_names["scene3"]
    for name, batch in batches.items():
        batch.validate()
        tensor_batch = batch.to_sensor_batch()
        assert isinstance(tensor_batch, SensorTensorBatch)


def test_three_scenes_share_schema_version():
    versions = {
        name: make_batch(**kwargs).schema_version
        for name, kwargs in SCENE_BATCH_KWARGS.items()
    }
    assert len(set(versions.values())) == 1
    assert versions["scene1"] == UNIFIED_SENSOR_BATCH_SCHEMA_VERSION


def test_missing_rgb_uses_zero_tensor_and_mask():
    batch = make_batch(front=True, left=False, right=False, rear=False, lidar=False)
    tensor_batch = batch.to_sensor_batch()
    assert tensor_batch.camera_view_mask[0].tolist() == [True, False, False, False]
    assert torch.equal(
        tensor_batch.camera_images[0, 1],
        torch.zeros_like(tensor_batch.camera_images[0, 1]),
    )
    assert batch.modality("left_rgb") is False


def test_missing_lidar_uses_zero_tensor_and_mask():
    batch = make_batch(front=True, left=True, right=True, rear=True, lidar=False)
    assert batch.modality("lidar_bev") is False
    tensor_batch = batch.to_sensor_batch()
    assert torch.equal(tensor_batch.lidar_bev, torch.zeros_like(tensor_batch.lidar_bev))


def test_three_scenes_load_same_checkpoint():
    manifest_path = (
        REPO_ROOT
        / "lightweight_vla_adapter"
        / "configs"
        / "universal_three_scene_v6_sensor_policy.manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(manifest["shared_across_scenes"]) == {"scene1", "scene2", "scene3"}
    for runner in (
        "run_control_experiment.py",
        "run_complex_avoidance_town05.py",
        "run_emergency_response_6km.py",
    ):
        source = (EXPERIMENT_CARLA / runner).read_text(encoding="utf-8")
        assert "UniversalVLAController" in source


def test_same_model_instance_processes_three_scene_samples():
    config = json.loads(
        (
            REPO_ROOT
            / "lightweight_vla_adapter"
            / "configs"
            / "universal_three_scene_v6_sensor_policy.json"
        ).read_text(encoding="utf-8")
    )
    model = build_model(config)
    pipeline = LightweightVLAPipeline(
        model,
        model_name=config["model_name"],
        device="cpu",
        dtype=torch.float32,
        checkpoint_loaded=True,
    )
    outputs = []
    for name, kwargs in SCENE_BATCH_KWARGS.items():
        batch = make_batch(**kwargs).to_sensor_batch()
        proposal = pipeline.predict_proposal(
            batch,
            request_id=f"unified-{name}",
            frame_id=f"carla-{name}",
            candidate_entity_ids=[[]],
            world_state={"frame_id": f"carla-{name}", "ego": {"speed_mps": 5.0}, "objects": []},
            stream_id=name,
            use_model_risk_assessment=True,
        )
        outputs.append(proposal)
    assert all(item["action"] in {
        "keep_lane",
        "accelerate",
        "decelerate",
        "stop",
        "emergency_brake",
        "lane_change_left",
        "lane_change_right",
        "turn_left",
        "turn_right",
    } for item in outputs)
    assert all(0.0 <= item["target_speed_kmh"] <= 100.0 for item in outputs)


def test_universal_controller_has_no_scene_branch():
    controller_path = EXPERIMENT_CARLA / "universal_vla_controller.py"
    source = controller_path.read_text(encoding="utf-8")
    for forbidden in (
        "scene_id",
        "scene_1",
        "scene_2",
        "scene_3",
        "scene3_",
        "event_id",
        "command_id",
    ):
        assert forbidden not in source


def test_instruction_fsm_supports_announce_activate_windows():
    from control.generic_instruction_fsm import GenericInstructionFSM

    fsm = GenericInstructionFSM(default_speed_kmh=25.0)
    commands = [
        {
            "id": "c01",
            "announce_at_m": 0,
            "activate_at_m": 0,
            "voice_text": "保持当前车道，提速至45公里每小时。",
            "structured_command": {
                "action": "SET_SPEED",
                "target_speed_kmh": 45,
            },
        },
        {
            "id": "c07",
            "announce_at_m": 2000,
            "activate_at_m": 2100,
            "voice_text": "前方路口左转。",
            "structured_command": {"action": "TURN", "direction": "LEFT"},
        },
    ]
    active = fsm.active_command(commands, 5.0)
    assert active["id"] == "c01"
    parsed = fsm.parse(active, use_parser_model=False)
    assert parsed.parsed_intent == "SET_SPEED"
    assert parsed.target_speed_kmh == 45.0
    active2 = fsm.active_command(commands, 2100.0)
    assert active2["id"] == "c07"
    parsed2 = fsm.parse(active2, use_parser_model=False)
    assert parsed2.parsed_intent == "TURN_LEFT"


def test_generic_yield_resume_does_not_depend_on_command_id():
    from control.generic_temporal_risk_supervisor import (
        GenericTemporalRiskSupervisor,
        TemporalRiskSupervisorConfig,
    )

    supervisor = GenericTemporalRiskSupervisor(
        TemporalRiskSupervisorConfig(hold_seconds=1.0, min_samples=3)
    )
    for frame in range(4):
        supervisor.observe(
            frame=frame,
            timestamp_s=float(frame) * 0.05,
            parsed_intent="YIELD",
            risk_level="low",
            target_lane_risk_level=None,
            ego_speed_kmh=0.0,
            requested_lane_direction=None,
        )
    decision = {
        "action": "emergency_brake",
        "target_speed_kmh": 0.0,
        "emergency": True,
    }
    canonical = dict(decision)
    risk = {"risk_level": "low", "recommended_action": "keep_lane"}
    final, override = supervisor.apply(
        decision,
        canonical,
        risk,
        parsed_intent="YIELD",
        requested_lane_direction=None,
        target_lane_risk=None,
        stationary_elapsed_s=3.0,
        resume_active=False,
        resume_speed_kmh=20.0,
        hold_seconds=1.0,
    )
    assert override == "temporal_hazard_clearance"
    assert final["target_speed_kmh"] >= 10.0


def test_fsm_semantic_goal_lane_change_takes_priority():
    from control.generic_instruction_fsm import GenericInstructionFSM

    fsm = GenericInstructionFSM(default_speed_kmh=32.0)
    parsed = fsm.parse(
        {
            "text": "施工路段，减速并道至左侧车道",
            "semantic_goal": ["decelerate", "lane_change_left"],
        },
        use_parser_model=False,
    )
    assert parsed.parsed_intent == "CHANGE_LANE_LEFT"
    assert parsed.requested_lane_direction == "left"


def test_generic_left_lane_clearance_does_not_depend_on_event_id():
    from control.generic_temporal_risk_supervisor import (
        GenericTemporalRiskSupervisor,
        TemporalRiskSupervisorConfig,
    )

    supervisor = GenericTemporalRiskSupervisor(
        TemporalRiskSupervisorConfig(hold_seconds=1.0, min_samples=3)
    )
    for frame in range(4):
        supervisor.observe(
            frame=frame,
            timestamp_s=float(frame) * 0.05,
            parsed_intent="CHANGE_LANE_LEFT",
            risk_level="high",
            target_lane_risk_level="low",
            ego_speed_kmh=0.0,
            requested_lane_direction="left",
        )
    decision = {
        "action": "decelerate",
        "target_speed_kmh": 4.0,
        "emergency": False,
    }
    canonical = dict(decision)
    risk = {"risk_level": "high", "recommended_action": "emergency_brake"}
    target_lane_risk = {"risk_level": "low", "recommended_action": "keep_lane"}
    final, override = supervisor.apply(
        decision,
        canonical,
        risk,
        parsed_intent="CHANGE_LANE_LEFT",
        requested_lane_direction="left",
        target_lane_risk=target_lane_risk,
        stationary_elapsed_s=3.0,
        resume_active=False,
        resume_speed_kmh=20.0,
        hold_seconds=1.0,
    )
    assert override == "target_lane_visual_clearance"
    assert final["action"] == "lane_change_left"


def test_generic_right_lane_clearance_does_not_depend_on_event_id():
    from control.generic_temporal_risk_supervisor import (
        GenericTemporalRiskSupervisor,
        TemporalRiskSupervisorConfig,
    )

    supervisor = GenericTemporalRiskSupervisor(
        TemporalRiskSupervisorConfig(hold_seconds=1.0, min_samples=3)
    )
    for frame in range(4):
        supervisor.observe(
            frame=frame,
            timestamp_s=float(frame) * 0.05,
            parsed_intent="CHANGE_LANE_RIGHT",
            risk_level="high",
            target_lane_risk_level="low",
            ego_speed_kmh=0.0,
            requested_lane_direction="right",
        )
    decision = {
        "action": "decelerate",
        "target_speed_kmh": 4.0,
        "emergency": False,
    }
    canonical = dict(decision)
    risk = {"risk_level": "high", "recommended_action": "emergency_brake"}
    target_lane_risk = {"risk_level": "low", "recommended_action": "keep_lane"}
    final, override = supervisor.apply(
        decision,
        canonical,
        risk,
        parsed_intent="CHANGE_LANE_RIGHT",
        requested_lane_direction="right",
        target_lane_risk=target_lane_risk,
        stationary_elapsed_s=3.0,
        resume_active=False,
        resume_speed_kmh=20.0,
        hold_seconds=1.0,
    )
    assert override == "target_lane_visual_clearance"
    assert final["action"] == "lane_change_right"


def test_cautious_crawl_holds_single_high_frame_once():
    from control.generic_temporal_risk_supervisor import (
        GenericTemporalRiskSupervisor,
        TemporalRiskSupervisorConfig,
    )

    supervisor = GenericTemporalRiskSupervisor(
        TemporalRiskSupervisorConfig(hold_seconds=1.0, min_samples=3)
    )
    supervisor.observe(
        frame=10,
        timestamp_s=0.5,
        parsed_intent="KEEP_LANE",
        risk_level="medium",
        target_lane_risk_level=None,
        ego_speed_kmh=8.0,
        requested_lane_direction=None,
    )
    crawl = {
        "action": "decelerate",
        "target_speed_kmh": 4.0,
        "emergency": False,
    }
    emergency = {
        "action": "emergency_brake",
        "target_speed_kmh": 0.0,
        "emergency": True,
    }
    # Enter cautious crawl on a medium-risk decision.
    final, override = supervisor.apply(
        crawl,
        crawl,
        {"risk_level": "medium", "recommended_action": "decelerate"},
        parsed_intent="KEEP_LANE",
        requested_lane_direction=None,
        target_lane_risk=None,
        stationary_elapsed_s=0.0,
        resume_active=False,
        resume_speed_kmh=32.0,
    )
    assert override is None
    # One isolated high frame holds the crawl once.
    final, override = supervisor.apply(
        emergency,
        emergency,
        {"risk_level": "high", "recommended_action": "emergency_brake"},
        parsed_intent="KEEP_LANE",
        requested_lane_direction=None,
        target_lane_risk=None,
        stationary_elapsed_s=0.0,
        resume_active=False,
        resume_speed_kmh=32.0,
    )
    assert override == "temporal_risk_confirmation"
    assert final["action"] == "decelerate"
    # The second consecutive high frame still holds the crawl; only the
    # third consecutive high frame escalates to emergency.
    final, override = supervisor.apply(
        emergency,
        emergency,
        {"risk_level": "high", "recommended_action": "emergency_brake"},
        parsed_intent="KEEP_LANE",
        requested_lane_direction=None,
        target_lane_risk=None,
        stationary_elapsed_s=0.0,
        resume_active=False,
        resume_speed_kmh=32.0,
    )
    assert override == "temporal_risk_confirmation"
    assert final["action"] == "decelerate"
    final, override = supervisor.apply(
        emergency,
        emergency,
        {"risk_level": "high", "recommended_action": "emergency_brake"},
        parsed_intent="KEEP_LANE",
        requested_lane_direction=None,
        target_lane_risk=None,
        stationary_elapsed_s=0.0,
        resume_active=False,
        resume_speed_kmh=32.0,
    )
    assert override is None
    assert final["action"] == "emergency_brake"
    # A persistent high remains latched; it must not alternate back to crawl.
    final, override = supervisor.apply(
        emergency,
        emergency,
        {"risk_level": "high", "recommended_action": "emergency_brake"},
        parsed_intent="KEEP_LANE",
        requested_lane_direction=None,
        target_lane_risk=None,
        stationary_elapsed_s=0.0,
        resume_active=False,
        resume_speed_kmh=32.0,
    )
    assert override is None
    assert final["action"] == "emergency_brake"


def test_high_confidence_temporal_risk_brakes_without_confirmation_delay():
    from control.generic_temporal_risk_supervisor import (
        GenericTemporalRiskSupervisor,
        TemporalRiskSupervisorConfig,
    )
    supervisor = GenericTemporalRiskSupervisor(TemporalRiskSupervisorConfig())
    supervisor.observe(frame=10, timestamp_s=0.5, parsed_intent="KEEP_LANE", risk_level="high", target_lane_risk_level=None, ego_speed_kmh=30.0, requested_lane_direction=None)
    emergency = {"action": "emergency_brake", "target_speed_kmh": 0.0, "emergency": True}
    risk = {"risk_level": "high", "recommended_action": "emergency_brake", "probabilities": {"high": 0.91}}
    final, override = supervisor.apply(emergency, emergency, risk, parsed_intent="KEEP_LANE", requested_lane_direction=None, target_lane_risk=None, stationary_elapsed_s=0.0, resume_active=False, resume_speed_kmh=32.0)
    assert override is None and final["action"] == "emergency_brake"


def test_unconfirmed_stop_at_low_risk_uses_crawl_floor():
    from control.generic_temporal_risk_supervisor import (
        GenericTemporalRiskSupervisor,
        TemporalRiskSupervisorConfig,
    )

    supervisor = GenericTemporalRiskSupervisor(
        TemporalRiskSupervisorConfig(hold_seconds=1.0, min_samples=3)
    )
    supervisor.observe(
        frame=10,
        timestamp_s=0.5,
        parsed_intent="KEEP_LANE",
        risk_level="low",
        target_lane_risk_level=None,
        ego_speed_kmh=12.0,
        requested_lane_direction=None,
    )
    emergency = {
        "action": "emergency_brake",
        "target_speed_kmh": 0.0,
        "emergency": True,
    }
    cruise = {"action": "keep_lane", "target_speed_kmh": 32.0}
    final, override = supervisor.apply(
        emergency,
        cruise,
        {"risk_level": "low", "recommended_action": "keep_lane"},
        parsed_intent="KEEP_LANE",
        requested_lane_direction=None,
        target_lane_risk=None,
        stationary_elapsed_s=0.0,
        resume_active=False,
        resume_speed_kmh=32.0,
    )
    assert override == "unconfirmed_stop_crawl_floor"
    assert final["action"] == "decelerate"
    assert final["target_speed_kmh"] == 15.0
    # High risk while moving is held as crawl until confirmed.
    final, override = supervisor.apply(
        emergency,
        cruise,
        {"risk_level": "high", "recommended_action": "emergency_brake"},
        parsed_intent="KEEP_LANE",
        requested_lane_direction=None,
        target_lane_risk=None,
        stationary_elapsed_s=0.0,
        resume_active=False,
        resume_speed_kmh=32.0,
    )
    assert override == "temporal_risk_confirmation"
    assert final["action"] == "decelerate"
    final, override = supervisor.apply(
        emergency,
        cruise,
        {"risk_level": "high", "recommended_action": "emergency_brake"},
        parsed_intent="KEEP_LANE",
        requested_lane_direction=None,
        target_lane_risk=None,
        stationary_elapsed_s=0.0,
        resume_active=False,
        resume_speed_kmh=32.0,
    )
    assert override == "temporal_risk_confirmation"
    assert final["action"] == "decelerate"
    final, override = supervisor.apply(
        emergency,
        cruise,
        {"risk_level": "high", "recommended_action": "emergency_brake"},
        parsed_intent="KEEP_LANE",
        requested_lane_direction=None,
        target_lane_risk=None,
        stationary_elapsed_s=0.0,
        resume_active=False,
        resume_speed_kmh=32.0,
    )
    assert override is None
    assert final["action"] == "emergency_brake"


def test_low_risk_command_speed_floor_holds_commanded_envelope():
    from control.generic_temporal_risk_supervisor import (
        GenericTemporalRiskSupervisor,
        TemporalRiskSupervisorConfig,
    )

    supervisor = GenericTemporalRiskSupervisor(
        TemporalRiskSupervisorConfig(hold_seconds=1.0, min_samples=3)
    )
    model = {
        "action": "decelerate",
        "target_speed_kmh": 9.0,
        "emergency": False,
    }
    canonical = {"action": "decelerate", "target_speed_kmh": 45.0}
    final, override = supervisor.apply(
        model,
        canonical,
        {"risk_level": "low", "recommended_action": "keep_lane"},
        parsed_intent="DECELERATE",
        requested_lane_direction=None,
        target_lane_risk=None,
        stationary_elapsed_s=0.0,
        resume_active=False,
        resume_speed_kmh=45.0,
    )
    assert override == "low_risk_command_speed_floor"
    assert final["action"] == "decelerate"
    assert final["target_speed_kmh"] == 45.0

    # An explicit stop command is never floored upward.
    stop_canonical = {"action": "stop", "target_speed_kmh": 0.0}
    final, override = supervisor.apply(
        model,
        stop_canonical,
        {"risk_level": "low", "recommended_action": "keep_lane"},
        parsed_intent="STOP",
        requested_lane_direction=None,
        target_lane_risk=None,
        stationary_elapsed_s=0.0,
        resume_active=False,
        resume_speed_kmh=45.0,
    )
    assert override is None
    assert final["target_speed_kmh"] == 9.0


def test_just_resumed_single_high_frame_holds_once():
    from control.generic_temporal_risk_supervisor import (
        GenericTemporalRiskSupervisor,
        TemporalRiskSupervisorConfig,
    )

    supervisor = GenericTemporalRiskSupervisor(
        TemporalRiskSupervisorConfig(hold_seconds=1.0, min_samples=3)
    )
    supervisor.observe(
        frame=100,
        timestamp_s=5.0,
        parsed_intent="KEEP_LANE",
        risk_level="low",
        target_lane_risk_level=None,
        ego_speed_kmh=5.0,
        requested_lane_direction=None,
    )
    cruise = {"action": "keep_lane", "target_speed_kmh": 32.0}
    emergency = {
        "action": "emergency_brake",
        "target_speed_kmh": 0.0,
        "emergency": True,
    }
    final, override = supervisor.apply(
        emergency,
        cruise,
        {"risk_level": "high", "recommended_action": "emergency_brake"},
        parsed_intent="KEEP_LANE",
        requested_lane_direction=None,
        target_lane_risk=None,
        stationary_elapsed_s=0.0,
        resume_active=False,
        resume_speed_kmh=32.0,
    )
    assert override == "temporal_risk_confirmation"
    assert final["action"] == "decelerate"
    final, override = supervisor.apply(
        emergency,
        cruise,
        {"risk_level": "high", "recommended_action": "emergency_brake"},
        parsed_intent="KEEP_LANE",
        requested_lane_direction=None,
        target_lane_risk=None,
        stationary_elapsed_s=0.0,
        resume_active=False,
        resume_speed_kmh=32.0,
    )
    assert override == "temporal_risk_confirmation"
    assert final["action"] == "decelerate"
    final, override = supervisor.apply(
        emergency,
        cruise,
        {"risk_level": "high", "recommended_action": "emergency_brake"},
        parsed_intent="KEEP_LANE",
        requested_lane_direction=None,
        target_lane_risk=None,
        stationary_elapsed_s=0.0,
        resume_active=False,
        resume_speed_kmh=32.0,
    )
    assert override is None
    assert final["action"] == "emergency_brake"


def test_masked_risk_probe_does_not_overwrite_main_risk_state():
    config = json.loads(
        (
            REPO_ROOT
            / "lightweight_vla_adapter"
            / "configs"
            / "universal_three_scene_v6_sensor_policy.json"
        ).read_text(encoding="utf-8")
    )
    model = build_model(config)
    pipeline = LightweightVLAPipeline(
        model,
        model_name=config["model_name"],
        device="cpu",
        dtype=torch.float32,
        checkpoint_loaded=True,
    )
    batch = make_batch(front=True, left=True, right=True, rear=True, lidar=False)
    sensor_batch = batch.to_sensor_batch()
    pipeline.predict_proposal(
        sensor_batch,
        request_id="unified-main-risk",
        frame_id="carla-main-risk",
        candidate_entity_ids=[[]],
        world_state={
            "frame_id": "carla-main-risk",
            "ego": {"speed_mps": 5.0},
            "objects": [],
        },
        stream_id="main-risk",
        use_model_risk_assessment=True,
    )
    main_risk = pipeline.last_visual_risk_assessment
    masked = SensorTensorBatch(
        **{
            **sensor_batch.__dict__,
            "camera_view_mask": torch.tensor(
                [[False, True, False, False]], dtype=torch.bool
            ),
        }
    )
    pipeline.predict_visual_risk(masked)
    assert pipeline.last_visual_risk_assessment == main_risk


def test_actor_truth_cannot_enter_model():
    batch = make_batch(**SCENE_BATCH_KWARGS["scene3"])
    sensor_batch = batch.to_sensor_batch()
    assert not hasattr(batch, "objects")
    assert not hasattr(batch, "actor")
    assert torch.equal(
        sensor_batch.camera_bev,
        torch.zeros_like(sensor_batch.camera_bev),
    )


def test_candidate_entities_cannot_enter_model():
    batch = make_batch(**SCENE_BATCH_KWARGS["scene3"])
    sensor_batch = batch.to_sensor_batch()
    assert sensor_batch.candidate_mask.sum().item() == 0
    assert sensor_batch.candidate_features.abs().sum().item() == 0


def test_scene_scripts_do_not_directly_control_ego_lane_change():
    source = (
        EXPERIMENT_CARLA / "emergency_scene_3_events.py"
    ).read_text(encoding="utf-8")
    # Event actors may be force-lane-changed, but never the ego.
    for line in source.splitlines():
        if "force_lane_change" in line and "ego" in line.lower():
            assert "event_actor" in line or "actor" in line


def test_scene_scripts_do_not_directly_control_ego_brake():
    source = (
        EXPERIMENT_CARLA / "emergency_scene_3_events.py"
    ).read_text(encoding="utf-8")
    for line in source.splitlines():
        if "apply_control" in line:
            assert "ego" not in line


def test_model_output_structure_is_consistent_across_scenes():
    config = json.loads(
        (
            REPO_ROOT
            / "lightweight_vla_adapter"
            / "configs"
            / "universal_three_scene_v6_sensor_policy.json"
        ).read_text(encoding="utf-8")
    )
    model = build_model(config)
    pipeline = LightweightVLAPipeline(
        model,
        model_name=config["model_name"],
        device="cpu",
        dtype=torch.float32,
        checkpoint_loaded=True,
    )
    keys = None
    for name, kwargs in SCENE_BATCH_KWARGS.items():
        proposal = pipeline.predict_proposal(
            make_batch(**kwargs).to_sensor_batch(),
            request_id=f"unified-{name}",
            frame_id=f"carla-{name}",
            candidate_entity_ids=[[]],
            world_state={"frame_id": f"carla-{name}", "ego": {"speed_mps": 5.0}, "objects": []},
            stream_id=name,
            use_model_risk_assessment=True,
        )
        current = set(proposal.keys())
        if keys is None:
            keys = current
        assert current == keys
        assert {"action", "target_speed_kmh", "target_lane", "confidence"} <= current
        risk = pipeline.last_visual_risk_assessment
        assert set(risk) >= {
            "risk_level",
            "recommended_action",
            "probabilities",
        }


class _FakeVector:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z


class _FakeTransform:
    def __init__(self, x=0.0, y=0.0, yaw=0.0):
        self.location = _FakeVector(x, y, 0.0)
        self.rotation = SimpleNamespace(pitch=0.0, yaw=yaw, roll=0.0)

    def get_forward_vector(self):
        yaw = math.radians(self.rotation.yaw)
        return _FakeVector(math.cos(yaw), math.sin(yaw), 0.0)

    def get_right_vector(self):
        yaw = math.radians(self.rotation.yaw)
        return _FakeVector(-math.sin(yaw), math.cos(yaw), 0.0)


class _FakeWaypoint:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, lane_id=-1, road_id=1):
        self.transform = _FakeTransform(x, y, yaw)
        self.lane_id = lane_id
        self.road_id = road_id
        self.lane_type = carla.LaneType.Driving
        self.lane_width = 3.5
        self.is_junction = False

    def next(self, distance_m):
        yaw = math.radians(self.transform.rotation.yaw)
        return [
            _FakeWaypoint(
                self.transform.location.x + math.cos(yaw) * float(distance_m),
                self.transform.location.y + math.sin(yaw) * float(distance_m),
                self.transform.rotation.yaw,
                self.lane_id,
                self.road_id,
            )
        ]

    def get_left_lane(self):
        return _FakeWaypoint(
            self.transform.location.x,
            self.transform.location.y,
            self.transform.rotation.yaw,
            self.lane_id - 1,
            self.road_id,
        )

    def get_right_lane(self):
        return _FakeWaypoint(
            self.transform.location.x,
            self.transform.location.y,
            self.transform.rotation.yaw,
            self.lane_id + 1,
            self.road_id,
        )


class _FakeVehicle:
    def __init__(self):
        self._transform = _FakeTransform(0.0, 0.0, 0.0)
        self._velocity = _FakeVector(5.0, 0.0, 0.0)
        self._control = SimpleNamespace(steer=0.0, throttle=0.3, brake=0.0)

    def get_transform(self):
        return self._transform

    def get_location(self):
        return self._transform.location

    def get_velocity(self):
        return self._velocity

    def get_control(self):
        return self._control


class _FakeMap:
    def __init__(self):
        self._waypoint = _FakeWaypoint(0.0, 0.0, 0.0, -1, 1)

    def get_waypoint(self, *args, **kwargs):
        return self._waypoint


def test_same_output_action_executes_through_one_route_pid():
    from control.generic_route_pid import GenericRoutePID

    world = SimpleNamespace(get_map=lambda: _FakeMap())
    ego = _FakeVehicle()
    pid = GenericRoutePID(
        world,
        ego,
        target_speed_kmh=40.0,
        fixed_delta_seconds=0.05,
    )
    for action in ("keep_lane", "decelerate", "lane_change_left"):
        pid.set_high_level_decision(
            {
                "action": action,
                "target_speed_kmh": 20.0,
                "target_lane": ("left" if action == "lane_change_left" else None),
                "emergency": False,
            }
        )
        control = pid.run_step()
        assert control is not None


def test_generic_lane_change_wait_timeout_falls_back_to_crawl():
    from control.generic_temporal_risk_supervisor import (
        GenericTemporalRiskSupervisor,
        TemporalRiskSupervisorConfig,
    )

    supervisor = GenericTemporalRiskSupervisor(
        TemporalRiskSupervisorConfig(lane_change_wait_timeout_s=8.0)
    )
    decision = {
        "action": "decelerate",
        "target_speed_kmh": 5.0,
        "emergency": False,
    }
    canonical = {"action": "keep_lane", "target_speed_kmh": 45.0}
    risk = {"risk_level": "low", "recommended_action": "keep_lane"}

    final, override = supervisor.apply(
        decision,
        canonical,
        risk,
        parsed_intent="CHANGE_LANE_LEFT",
        requested_lane_direction="left",
        target_lane_risk=None,
        stationary_elapsed_s=9.0,
        resume_active=False,
        resume_speed_kmh=20.0,
    )

    assert override == "lane_change_wait_timeout"
    assert final["action"] == "keep_lane"
    assert final["target_speed_kmh"] == 15.0


def test_generic_command_speed_floor_only_applies_to_first_deceleration_frame():
    from control.generic_temporal_risk_supervisor import (
        GenericTemporalRiskSupervisor,
        TemporalRiskSupervisorConfig,
    )

    supervisor = GenericTemporalRiskSupervisor(
        TemporalRiskSupervisorConfig()
    )
    decision = {
        "action": "decelerate",
        "target_speed_kmh": 8.0,
        "emergency": False,
    }
    canonical = {"action": "keep_lane", "target_speed_kmh": 45.0}
    risk = {"risk_level": "low", "recommended_action": "keep_lane"}
    kwargs = {
        "parsed_intent": "DECELERATE",
        "requested_lane_direction": None,
        "target_lane_risk": None,
        "stationary_elapsed_s": 0.0,
        "resume_active": False,
        "resume_speed_kmh": 20.0,
    }

    first, override1 = supervisor.apply(decision, canonical, risk, **kwargs)
    assert override1 == "low_risk_command_speed_floor"
    assert first["target_speed_kmh"] == 45.0

    second, override2 = supervisor.apply(decision, canonical, risk, **kwargs)
    assert override2 != "low_risk_command_speed_floor"
    assert second["target_speed_kmh"] == 8.0


def test_generic_stationary_high_risk_liveness_prevents_deadlock():
    from control.generic_temporal_risk_supervisor import (
        GenericTemporalRiskSupervisor,
        TemporalRiskSupervisorConfig,
    )

    supervisor = GenericTemporalRiskSupervisor(
        TemporalRiskSupervisorConfig()
    )
    decision = {
        "action": "emergency_brake",
        "target_speed_kmh": 0.0,
        "emergency": True,
    }
    canonical = {"action": "keep_lane", "target_speed_kmh": 45.0}
    risk = {"risk_level": "high", "recommended_action": "emergency_brake"}

    final, override = supervisor.apply(
        decision,
        canonical,
        risk,
        parsed_intent="KEEP_LANE",
        requested_lane_direction=None,
        target_lane_risk=None,
        stationary_elapsed_s=3.0,
        resume_active=False,
        resume_speed_kmh=20.0,
    )

    assert override == "stationary_high_risk_liveness"
    assert final["action"] == "decelerate"
    assert final["target_speed_kmh"] == 12.0
    assert final["emergency"] is False


def test_physical_forward_radar_emergency_preempts_visual_miss():
    from universal_vla_controller import fuse_forward_radar_risk

    learned = {
        "risk_level": "low",
        "recommended_action": "keep_lane",
        "reason_codes": [],
        "probabilities": {"low": 0.96, "medium": 0.02, "high": 0.02},
        "raw_argmax_level": "low",
    }
    fused = fuse_forward_radar_risk(
        learned,
        {
            "schema_version": "physical_forward_radar/1.0",
            "sensor_frame": 100,
            "candidate_count": 4,
            "nearest_distance_m": 5.5,
        },
        ego_speed_kmh=5.8,
    )

    assert fused["risk_level"] == "high"
    assert fused["recommended_action"] == "emergency_brake"
    assert fused["probabilities"]["high"] == 1.0
    assert fused["learned_probabilities"]["low"] == pytest.approx(0.96)
    assert "physical_forward_radar_emergency_distance" in fused["reason_codes"]


def test_physical_forward_radar_uses_speed_dependent_stopping_distance():
    from universal_vla_controller import fuse_forward_radar_risk

    learned = {
        "risk_level": "low",
        "recommended_action": "keep_lane",
        "reason_codes": [],
        "probabilities": {"low": 0.9, "medium": 0.08, "high": 0.02},
    }
    fused = fuse_forward_radar_risk(
        learned,
        {"candidate_count": 1, "nearest_distance_m": 20.0},
        ego_speed_kmh=50.0,
    )

    assert fused["risk_level"] == "high"
    assert fused["physical_forward_radar"]["emergency_distance_m"] > 20.0


def test_physical_forward_radar_caution_and_clear_paths():
    from universal_vla_controller import fuse_forward_radar_risk

    learned = {
        "risk_level": "low",
        "recommended_action": "keep_lane",
        "reason_codes": [],
        "probabilities": {"low": 0.9, "medium": 0.08, "high": 0.02},
    }
    caution = fuse_forward_radar_risk(
        learned,
        {"candidate_count": 1, "nearest_distance_m": 10.0},
        ego_speed_kmh=0.0,
    )
    clear = fuse_forward_radar_risk(
        learned,
        {"candidate_count": 0, "nearest_distance_m": None},
        ego_speed_kmh=30.0,
    )

    assert caution["risk_level"] == "medium"
    assert caution["recommended_action"] == "decelerate"
    assert "physical_forward_radar_caution_distance" in caution["reason_codes"]
    assert clear == learned


def test_confirmed_physical_radar_high_is_not_downgraded_while_stopped():
    from control.generic_temporal_risk_supervisor import (
        GenericTemporalRiskSupervisor,
    )

    supervisor = GenericTemporalRiskSupervisor()
    emergency = {"action": "emergency_brake", "target_speed_kmh": 0.0, "emergency": True}
    risk = {
        "risk_level": "high",
        "recommended_action": "emergency_brake",
        "probabilities": {"low": 0.0, "medium": 0.0, "high": 1.0},
        "risk_score": 1.0,
    }
    final, override = supervisor.apply(
        emergency,
        {"action": "keep_lane", "target_speed_kmh": 45.0},
        risk,
        parsed_intent="KEEP_LANE",
        requested_lane_direction=None,
        target_lane_risk=None,
        stationary_elapsed_s=5.0,
        resume_active=False,
        resume_speed_kmh=20.0,
    )
    assert override is None
    assert final["action"] == "emergency_brake"
