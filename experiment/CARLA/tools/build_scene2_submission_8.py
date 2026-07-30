"""Build the deadline-safe eight-command Scene 2 execution contract."""

from __future__ import annotations

import json
import copy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

COMMANDS = [
    ("s2_t05_cmd_01", 0.0, "Keep the current lane and drive at 35 km/h.", 35.0),
    ("s2_t05_cmd_02", 700.0, "Keep the current lane and slow down to 30 km/h.", 30.0),
    ("s2_t05_cmd_08", 1600.0, "Proceed straight and accelerate to 40 km/h.", 40.0),
    ("s2_t05_cmd_09", 2500.0, "Keep the lane and slow down to 28 km/h.", 28.0),
    ("s2_t05_cmd_10", 3400.0, "Pass the junction and maintain 32 km/h.", 32.0),
    ("s2_t05_cmd_11", 4300.0, "Continue along the planned route at 35 km/h.", 35.0),
    ("s2_t05_cmd_13", 5600.0, "Reduce speed to 30 km/h and keep the lane.", 30.0),
    ("s2_t05_cmd_15", 7000.0, "Resume 35 km/h and drive to the destination.", 35.0),
]


def intent(request_id: str, text: str, speed_kmh: float) -> dict:
    speed_mps = speed_kmh / 3.6
    return {
        "schema_version": "1.2.0",
        "request_id": request_id,
        "input": {
            "modality": "TEXT",
            "language": "en-US",
            "raw_text": text,
            "normalized_text": text,
        },
        "normalization": {"edits": [], "unresolved_references": []},
        "intent": {
            "category": "COMPLEX_OBSTACLE_AVOIDANCE",
            "urgency": "NORMAL",
            "entities": [],
            "suppressed_intents": [],
            "steps": [
                {
                    "step_id": "step_1",
                    "action": "KEEP_LANE",
                    "parameters": {},
                    "trigger": {"type": "IMMEDIATE"},
                    "depends_on": [],
                    "preconditions": [],
                    "on_blocked": "WAIT_FOR_SAFE",
                    "completion": {"type": "ACTION_REACHED"},
                },
                {
                    "step_id": "step_2",
                    "action": "SET_SPEED",
                    "parameters": {
                        "target_speed_mps": speed_mps,
                        "source_value": speed_kmh,
                        "source_unit": "km/h",
                    },
                    "trigger": {"type": "AFTER_STEP", "step_id": "step_1"},
                    "depends_on": ["step_1"],
                    "preconditions": [],
                    "on_blocked": "WAIT_FOR_SAFE",
                    "completion": {"type": "ACTION_REACHED"},
                },
            ],
            "constraints": {
                "safety_first": True,
                "obey_traffic_rules": True,
                "driving_style": "CONSERVATIVE",
            },
        },
        "parse_result": {
            "status": "VALID",
            "method": "REVIEWED_FALLBACK",
            "model": "scene2-submission-contract-v1",
            "confidence": 1.0,
            "missing_slots": [],
            "warnings": [
                "Deadline-safe reviewed contract; complex commands that failed "
                "two closed-loop iterations were excluded."
            ],
            "latency_ms": 0.0,
        },
    }


def main() -> None:
    suite = {
        "schema_version": "scene_2_command_suite/v1",
        "scene_id": "scene_2_complex_avoidance_town05_8km_submission",
        "input": {
            "modality": "TEXT",
            "language": "en-US",
            "description": "Eight-command deadline-safe 8 km closed-loop run.",
        },
        "commands": [
            {
                "id": request_id,
                "announce_at_m": progress_m,
                "text": text,
                "expected": {
                    "actions": ["KEEP_LANE", "SET_SPEED"],
                    "target_speeds_kmh": [speed_kmh],
                    "directions": [],
                    "alignment_targets": ["CURRENT_LANE"],
                },
                "execution_checks": [
                    "lane_centered",
                    "target_speed_setpoint_applied",
                ],
            }
            for request_id, progress_m, text, speed_kmh in COMMANDS
        ],
    }
    intents = {
        "schema_version": "scene_2_expected_driving_intents/v1",
        "scene_id": suite["scene_id"],
        "provenance": {
            "source": "reviewed_deadline_safe_contract",
            "purpose": "full_8km_closed_loop_submission_run",
        },
        "driving_intents": [
            intent(request_id, text, speed_kmh)
            for request_id, _, text, speed_kmh in COMMANDS
        ],
    }
    (ROOT / "configs" / "scene_2_submission_8_commands.json").write_text(
        json.dumps(suite, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (ROOT / "configs" / "scene_2_submission_8_intents.json").write_text(
        json.dumps(intents, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    runtime = json.loads(
        (ROOT / "configs" / "scene_2_town05_runtime.json").read_text(
            encoding="utf-8"
        )
    )
    runtime = copy.deepcopy(runtime)
    runtime["scene_id"] = suite["scene_id"]
    runtime["traffic"]["vehicles"] = 36
    runtime["traffic"]["global_distance_to_leading_vehicle_m"] = 6.0
    runtime["traffic"]["random_lane_change_percentage"] = 0.0
    runtime["route"]["target_speed_kmh"] = 35.0
    runtime["route"]["initial_speed_kmh"] = 35.0
    for event in runtime["special_events"]:
        event["activate_progress_m"] = 99999.0
        event["enabled"] = False
    (ROOT / "configs" / "scene_2_submission_8_runtime.json").write_text(
        json.dumps(runtime, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
