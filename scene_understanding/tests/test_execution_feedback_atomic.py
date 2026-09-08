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
