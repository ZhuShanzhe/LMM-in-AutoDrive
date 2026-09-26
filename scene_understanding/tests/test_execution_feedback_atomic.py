from scene_understanding.src.execution_feedback import evaluate_execution_feedback


def test_action_reached_requires_stable_approved_frames():
    intent = {
        "request_id": "compound-01",
        "intent": {
            "steps": [
                {
                    "step_id": "step-01",
                    "action": "KEEP_LANE",
                    "parameters": {},
                    "completion": {"type": "ACTION_REACHED"},
                }
            ]
        },
    }
    plan = {
        "request_id": "compound-01",
        "plan_status": "ACTIVE",
        "active_step_id": "step-01",
    }
    tracker = None
    feedback = None
    for frame in range(1, 6):
        frame_id = f"frame-{frame}"
        decision = {
            "request_id": "compound-01",
            "frame_id": frame_id,
            "source_step_id": "step-01",
            "decision_status": "READY",
            "target_lane": None,
            "matched_entity_id": None,
        }
        world = {
            "frame_id": frame_id,
            "timestamp_s": frame * 0.05,
            "ego": {
                "speed_mps": 5.0,
                "lane_id": -1,
                "adjacent_lanes": {},
            },
            "sensor_events": {"collisions": []},
            "objects": [],
        }
        tracker, feedback = evaluate_execution_feedback(
            intent,
            plan,
            decision,
            world,
            tracker=tracker,
            required_stable_frames=5,
        )
        if frame < 5:
            assert feedback is None
    assert feedback is not None
    assert feedback["outcome"] == "COMPLETED"
    assert feedback["reason_codes"] == ["approved_action_stable"]


def test_relative_speed_completion_uses_latched_target_and_stability():
    intent = {
        "request_id": "relative-speed-01",
        "intent": {
            "steps": [
                {
                    "step_id": "step-01",
                    "action": "ADJUST_SPEED",
                    "parameters": {
                        "change": "DECREASE",
                        "speed_delta_mps": 2.0,
                    },
                    "completion": {"type": "TARGET_SPEED_REACHED"},
                }
            ]
        },
    }
    plan = {
        "request_id": "relative-speed-01",
        "plan_status": "ACTIVE",
        "active_step_id": "step-01",
        "step_states": [
            {
                "step_id": "step-01",
                "resolved_target_speed_kmh": 28.8,
            }
        ],
    }
    tracker = None
    feedback = None
    for frame in range(1, 6):
        frame_id = f"relative-frame-{frame}"
        decision = {
            "request_id": "relative-speed-01",
            "frame_id": frame_id,
            "source_step_id": "step-01",
            "decision_status": "READY",
            "target_lane": None,
            "matched_entity_id": None,
        }
        world = {
            "frame_id": frame_id,
            "timestamp_s": frame * 0.05,
            "ego": {
                "speed_mps": 8.0,
                "lane_id": -1,
                "adjacent_lanes": {},
            },
            "sensor_events": {"collisions": []},
            "objects": [],
        }
        tracker, feedback = evaluate_execution_feedback(
            intent,
            plan,
            decision,
            world,
            tracker=tracker,
            required_stable_frames=5,
        )
        if frame < 5:
            assert feedback is None
    assert feedback is not None
    assert feedback["reason_codes"] == [
        "target_speed_reached",
        "target_speed_stable",
    ]


def test_grounded_lane_change_target_loss_never_becomes_completion():
    intent = {
        "request_id": "lane-target-loss",
        "intent": {
            "steps": [
                {
                    "step_id": "lane-step",
                    "action": "CHANGE_LANE",
                    "parameters": {"direction": "LEFT"},
                    "completion": {"type": "LANE_CHANGE_COMPLETED"},
                }
            ]
        },
    }
    plan = {
        "request_id": "lane-target-loss",
        "plan_status": "ACTIVE",
        "active_step_id": "lane-step",
    }
    tracker = None
    feedback = None
    for frame in range(1, 9):
        frame_id = f"lane-loss-{frame}"
        decision = {
            "request_id": "lane-target-loss",
            "frame_id": frame_id,
            "source_step_id": "lane-step",
            "decision_status": "READY",
            "target_lane": "left",
            "matched_entity_id": "lead-vehicle",
        }
        world = {
            "frame_id": frame_id,
            "timestamp_s": frame * 0.05,
            "ego": {
                "speed_mps": 5.0,
                "lane_id": -1 if frame == 1 else -2,
                "adjacent_lanes": {"left": {"lane_id": -2}},
            },
            "sensor_events": {"collisions": []},
            "objects": (
                [{"object_id": "lead-vehicle", "category": "vehicle"}]
                if frame == 1
                else []
            ),
        }
        tracker, feedback = evaluate_execution_feedback(
            intent,
            plan,
            decision,
            world,
            tracker=tracker,
            required_stable_frames=3,
        )
        assert feedback is None
