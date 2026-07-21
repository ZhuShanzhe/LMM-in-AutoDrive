from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


MODULE_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = MODULE_ROOT / "schemas" / "driving_intent.schema.json"


class IntentValidationError(ValueError):
    pass


def load_schema() -> dict[str, Any]:
    with SCHEMA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


_VALIDATOR = Draft202012Validator(load_schema())


def schema_errors(document: dict[str, Any]) -> list[str]:
    return [
        f"{'/'.join(str(item) for item in error.path) or '<root>'}: {error.message}"
        for error in sorted(
            _VALIDATOR.iter_errors(document), key=lambda item: list(item.path)
        )
    ]


def semantic_errors(document: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    intent = document.get("intent", {})
    steps = intent.get("steps", [])
    parse_result = document.get("parse_result", {})
    step_ids = [step.get("step_id") for step in steps]

    if len(step_ids) != len(set(step_ids)):
        errors.append("step_id must be unique")

    seen: set[str] = set()
    for step in steps:
        step_id = step.get("step_id", "<unknown>")
        action = step.get("action")
        parameters = step.get("parameters", {})
        target = step.get("target")
        trigger = step.get("trigger", {})

        for dependency in step.get("depends_on", []):
            if dependency not in seen:
                errors.append(
                    f"{step_id} depends on unknown or later step {dependency}"
                )
        trigger_step = trigger.get("step_id")
        if trigger_step is not None and trigger_step not in seen:
            errors.append(
                f"{step_id} trigger references unknown or later step {trigger_step}"
            )

        if action == "SET_SPEED" and "target_speed_mps" not in parameters:
            errors.append(f"{step_id} SET_SPEED requires target_speed_mps")
        if action == "ADJUST_SPEED" and not (
            "change" in parameters or "speed_delta_mps" in parameters
        ):
            errors.append(f"{step_id} ADJUST_SPEED requires change or speed_delta_mps")
        if action == "CHANGE_LANE" and parameters.get("direction") not in {
            "LEFT",
            "RIGHT",
        }:
            errors.append(f"{step_id} CHANGE_LANE requires LEFT or RIGHT direction")
        if action == "TURN" and parameters.get("direction") not in {
            "LEFT",
            "RIGHT",
            "STRAIGHT",
        }:
            errors.append(f"{step_id} TURN requires a direction")
        if action in {"YIELD", "OVERTAKE", "AVOID"} and target is None:
            errors.append(f"{step_id} {action} requires a target")
        if action == "EMERGENCY_BRAKE" and intent.get("urgency") != "EMERGENCY":
            errors.append("EMERGENCY_BRAKE requires EMERGENCY urgency")

        trigger_type = trigger.get("type")
        if trigger_type == "AFTER_STEP" and "step_id" not in trigger:
            errors.append(f"{step_id} AFTER_STEP trigger requires step_id")
        if trigger_type == "AT_DISTANCE" and "distance_m" not in trigger:
            errors.append(f"{step_id} AT_DISTANCE trigger requires distance_m")
        if trigger_type == "CONDITION" and "description" not in trigger:
            errors.append(f"{step_id} CONDITION trigger requires description")
        seen.add(step_id)

    if parse_result.get("status") == "VALID" and not steps:
        errors.append("VALID result must contain at least one step")
    return errors


def validate_document(document: dict[str, Any]) -> None:
    errors = schema_errors(document) + semantic_errors(document)
    if errors:
        raise IntentValidationError("\n".join(errors))

