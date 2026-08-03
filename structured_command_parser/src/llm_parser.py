from __future__ import annotations

import json
import re
from pathlib import Path
from time import perf_counter
from typing import Any

from .factory import make_document, make_step
from .intent_boundaries import classify_chinese_braking
from .normalizer import normalize_text
from .schema_tools import IntentValidationError


MODULE_ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = MODULE_ROOT / "configs" / "parser_prompt.txt"


class QwenIntentParser:
    def __init__(self, model_path: str, *, max_new_tokens: int = 320) -> None:
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.model: Any = None
        self.tokenizer: Any = None

    def load(self) -> None:
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "Qwen dependencies are missing. Install requirements-model.txt first."
            ) from error

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.generation_config.do_sample = False
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None
        self.model.eval()

    def parse(
        self,
        raw_text: str,
        *,
        modality: str = "TEXT",
        request_id: str | None = None,
    ) -> dict[str, Any]:
        self.load()
        normalized_text = normalize_text(raw_text)
        prompt = PROMPT_PATH.read_text(encoding="utf-8")
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": normalized_text},
        ]
        started = perf_counter()
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        generated = outputs[0][inputs["input_ids"].shape[-1] :]
        response = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        latency_ms = (perf_counter() - started) * 1000

        try:
            payload = self._normalize_payload(
                self._decode_payload(response), normalized_text
            )
            status = payload.pop("status")
            missing_slots = payload.pop("missing_slots", [])
            warnings = payload.pop("warnings", [])
            clarification_question = payload.pop("clarification_question", None)
            steps = self._expand_commands(payload.get("commands", []))
            document = make_document(
                raw_text=raw_text,
                normalized_text=normalized_text,
                modality=modality,
                category=payload["category"],
                urgency=payload["urgency"],
                steps=steps,
                status=status,
                method="LLM",
                model=Path(self.model_path).name,
                confidence=self._heuristic_confidence(status, missing_slots),
                latency_ms=latency_ms,
                request_id=request_id,
                missing_slots=missing_slots,
                warnings=warnings,
                clarification_question=clarification_question,
                driving_style=payload["driving_style"],
                max_speed_mps=payload.get("max_speed_mps"),
            )
            return document
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, IntentValidationError) as error:
            return make_document(
                raw_text=raw_text,
                normalized_text=normalized_text,
                modality=modality,
                category="META_CONTROL",
                urgency="NORMAL",
                steps=[],
                status="INVALID",
                method="LLM",
                model=Path(self.model_path).name,
                confidence=0.0,
                latency_ms=latency_ms,
                request_id=request_id,
                warnings=[
                    f"模型输出无法通过接口校验: {error}",
                    f"模型原始输出: {response[:500]}",
                ],
            )

    @staticmethod
    def _expand_commands(commands: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized_commands: list[dict[str, Any]] = []
        for index, command in enumerate(commands):
            normalized_commands.append(command)
            next_action = commands[index + 1].get("action") if index + 1 < len(commands) else None
            if (
                command.get("action") == "CHANGE_LANE"
                and command.get("purpose") == "OVERTAKE"
                and next_action != "OVERTAKE"
            ):
                overtake = {"action": "OVERTAKE"}
                for source_key in (
                    "target_type",
                    "target_relation",
                    "target_description",
                    "target_ref",
                    "target_attributes",
                    "target_open_descriptors",
                    "goal_conditions",
                ):
                    if source_key in command:
                        overtake[source_key] = command[source_key]
                normalized_commands.append(overtake)

        steps: list[dict[str, Any]] = []
        for index, command in enumerate(normalized_commands, start=1):
            action = command["action"]
            step_id = f"step_{index}"
            previous_id = f"step_{index - 1}" if index > 1 else None
            parameters = {
                key: command[key]
                for key in (
                    "direction",
                    "change",
                    "target_speed_mps",
                    "speed_delta_mps",
                    "distance_m",
                    "start_distance_m",
                    "transition_distance_m",
                    "following_distance_m",
                    "duration_s",
                    "lane_count",
                    "lane_index",
                    "lane_reference",
                    "parking_maneuver",
                    "source_value",
                    "source_unit",
                )
                if key in command
            }
            target = None
            if "target_type" in command:
                target = {
                    "type": command["target_type"],
                    "relation": command.get("target_relation", "UNSPECIFIED"),
                }
                if command.get("target_description"):
                    target["description"] = command["target_description"]
                if isinstance(command.get("target_attributes"), dict):
                    target["canonical_attributes"] = command[
                        "target_attributes"
                    ]
                if isinstance(command.get("target_open_descriptors"), list):
                    target["open_descriptors"] = command[
                        "target_open_descriptors"
                    ]
                if isinstance(command.get("target_coordinates"), dict):
                    target["coordinates"] = command["target_coordinates"]

            trigger: dict[str, Any] = {"type": "IMMEDIATE"}
            condition_predicates = {
                item.get("predicate")
                for item in command.get("goal_conditions", [])
                if isinstance(item, dict)
            }
            condition_entity_ref = next(
                (
                    item.get("object")
                    for item in command.get("goal_conditions", [])
                    if isinstance(item, dict) and item.get("object")
                ),
                command.get("target_ref"),
            )
            if "VISIBLE" in condition_predicates:
                trigger = {
                    "type": "ON_ENTITY_VISIBLE",
                    "entity_ref": condition_entity_ref,
                }
            elif "AFTER" in condition_predicates:
                trigger = {
                    "type": "AFTER_ENTITY",
                    "entity_ref": condition_entity_ref,
                }
            elif action == "WAIT" and command.get("condition"):
                trigger = {
                    "type": "CONDITION",
                    "description": command["condition"],
                }
            elif previous_id is not None:
                trigger = {"type": "AFTER_STEP", "step_id": previous_id}
            elif action == "TURN" and "distance_m" in command:
                trigger = {"type": "AT_DISTANCE", "distance_m": command["distance_m"]}
            elif action == "TURN":
                trigger = {"type": "AT_JUNCTION"}
            elif target is not None and (
                action
                in {
                    "FOLLOW",
                    "APPROACH",
                    "NAVIGATE_TO",
                    "YIELD",
                    "OVERTAKE",
                    "PASS_BY",
                    "AVOID",
                    "ENTER_AREA",
                    "EXIT_AREA",
                }
                or command.get("purpose") in {"YIELD", "OVERTAKE"}
            ):
                trigger = {"type": "OBJECT_PRESENT"}

            preconditions: list[str] = []
            on_blocked = "SAFE_STOP"
            completion = None
            if action == "SET_SPEED":
                preconditions = ["PATH_CLEAR"]
                on_blocked = "WAIT_FOR_SAFE"
                completion = {"type": "TARGET_SPEED_REACHED"}
            elif action == "ADJUST_SPEED" and command.get("change") == "INCREASE":
                preconditions = ["PATH_CLEAR"]
                on_blocked = "WAIT_FOR_SAFE"
            elif (
                action == "ADJUST_SPEED"
                and command.get("purpose") == "YIELD"
                and target is not None
            ):
                preconditions = ["TARGET_VISIBLE"]
                completion = {"type": "TARGET_CLEARED"}
            elif action in {"CHANGE_LANE", "MERGE"}:
                direction = command.get("direction")
                if direction in {"LEFT", "RIGHT"}:
                    preconditions = [
                        f"{direction}_LANE_EXISTS",
                        f"{direction}_LANE_SAFE",
                        "LANE_CHANGE_LEGAL",
                    ]
                else:
                    preconditions = ["TARGET_LANE_SAFE", "LANE_CHANGE_LEGAL"]
                on_blocked = "WAIT_FOR_SAFE"
                completion = {"type": "LANE_CHANGE_COMPLETED"}
            elif action in {"TURN", "U_TURN"}:
                preconditions = ["JUNCTION_REACHED", "PATH_CLEAR"]
                on_blocked = "WAIT_FOR_SAFE"
                completion = {"type": "JUNCTION_EXITED"}
            elif action in {"YIELD", "OVERTAKE", "PASS_BY", "AVOID"}:
                preconditions = ["TARGET_VISIBLE", "PATH_CLEAR"]
                on_blocked = "SAFE_STOP"
                completion = {"type": "TARGET_CLEARED"}
            elif action == "FOLLOW":
                preconditions = ["TARGET_VISIBLE", "PATH_CLEAR"]
                on_blocked = "WAIT_FOR_SAFE"
                completion = {
                    "type": "DURATION_ELAPSED"
                    if "duration_s" in parameters
                    else "FOLLOWING_ESTABLISHED"
                }
            elif action in {"APPROACH", "NAVIGATE_TO"}:
                preconditions = ["TARGET_REACHABLE", "PATH_CLEAR"]
                on_blocked = "WAIT_FOR_SAFE"
                completion = {"type": "TARGET_REACHED"}
            elif action == "WAIT":
                completion = {
                    "type": "DURATION_ELAPSED"
                    if "duration_s" in parameters
                    else "WAIT_CONDITION_MET"
                }
            elif action == "PARK":
                preconditions = ["PATH_CLEAR", "PARKING_SPACE_AVAILABLE"]
                on_blocked = "REQUEST_CLARIFICATION"
                completion = {"type": "PARKING_COMPLETED"}
            elif action == "REVERSE":
                preconditions = ["PATH_CLEAR"]
                on_blocked = "SAFE_STOP"
                completion = {"type": "ACTION_REACHED"}
            elif action in {"ENTER_AREA", "EXIT_AREA"}:
                preconditions = ["AREA_ACCESSIBLE", "PATH_CLEAR"]
                on_blocked = "WAIT_FOR_SAFE"
                completion = {
                    "type": "AREA_ENTERED" if action == "ENTER_AREA" else "AREA_EXITED"
                }
            elif action == "STOP" or action == "EMERGENCY_BRAKE":
                predicates = {
                    item.get("predicate")
                    for item in command.get("goal_conditions", [])
                    if isinstance(item, dict)
                }
                completion = {
                    "type": (
                        "STOPPED_BEFORE_TARGET"
                        if "BEFORE" in predicates
                        else "VEHICLE_STOPPED"
                    )
                }
            elif action in {"PROCEED", "RESUME"}:
                preconditions = ["PATH_CLEAR"]
                on_blocked = "WAIT_FOR_SAFE"

            steps.append(
                make_step(
                    step_id,
                    action,
                    parameters=parameters,
                    trigger=trigger,
                    depends_on=[previous_id] if previous_id else [],
                    preconditions=preconditions,
                    on_blocked=on_blocked,
                    purpose=command.get("purpose"),
                    target=target,
                    target_ref=command.get("target_ref"),
                    goal_conditions=command.get("goal_conditions"),
                    completion=completion,
                )
            )
        return steps

    @staticmethod
    def _decode_payload(response: str) -> dict[str, Any]:
        candidate = response.strip()
        if candidate.startswith("```"):
            lines = candidate.splitlines()
            candidate = "\n".join(lines[1:-1]).strip()
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start < 0 or end <= start:
                raise
            payload = json.loads(candidate[start : end + 1])
        if not isinstance(payload, dict):
            raise TypeError("Model output must be a JSON object")
        return payload

    @staticmethod
    def _normalize_payload(
        payload: dict[str, Any], normalized_text: str
    ) -> dict[str, Any]:
        action_aliases = {
            "RETURN_TO_LANE": "RESUME",
            "RESTORE_NORMAL": "RESUME",
            "SLOW_DOWN": "ADJUST_SPEED",
            "SPEED_UP": "ADJUST_SPEED",
            "BRAKE": "STOP",
        }
        target_aliases = {
            "TRAFFIC_CONSTRUCTION": "CONSTRUCTION_ZONE",
            "TRAFFIC_CONSTRUCTION_ZONE": "CONSTRUCTION_ZONE",
            "CONSTRUCTION": "CONSTRUCTION_ZONE",
            "CONE": "TRAFFIC_CONE",
            "CAR": "VEHICLE",
            "WALKER": "PEDESTRIAN",
        }
        commands = payload.get("commands")
        if not isinstance(commands, list):
            return payload

        payload.setdefault("missing_slots", [])
        payload.setdefault("warnings", [])
        payload.setdefault("status", "VALID" if commands else "NEEDS_CLARIFICATION")
        payload.setdefault(
            "driving_style",
            "CONSERVATIVE"
            if any(token in normalized_text for token in ("危险", "施工", "雨天", "减速", "稳"))
            else "NORMAL",
        )

        for command in commands:
            action = action_aliases.get(command.get("action"), command.get("action"))
            command["action"] = action
            if command.get("target_type") in target_aliases:
                command["target_type"] = target_aliases[command["target_type"]]
            if action == "SET_SPEED" and "target_speed_mps" not in command:
                if command.get("purpose") in {"RESTORE_NORMAL", "RESUME"}:
                    command.clear()
                    command["action"] = "RESUME"
                elif "change" in command:
                    command["action"] = "ADJUST_SPEED"

        explicit_speed = re.search(
            r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>km/h|m/s)", normalized_text
        )
        speed_commands = [
            command for command in commands if command.get("action") == "SET_SPEED"
        ]
        if explicit_speed and speed_commands:
            value = float(explicit_speed.group("value"))
            unit = explicit_speed.group("unit")
            target_speed = value / 3.6 if unit == "km/h" else value
            for command in speed_commands:
                command["target_speed_mps"] = round(target_speed, 3)
        elif not explicit_speed:
            for command in commands:
                if command.get("action") == "SET_SPEED":
                    command["action"] = "ADJUST_SPEED"
                    command.pop("target_speed_mps", None)
                    command["change"] = (
                        "DECREASE"
                        if any(token in normalized_text for token in ("减速", "安全车速", "慢"))
                        else "INCREASE"
                    )

        if "安全车速" in normalized_text:
            payload["driving_style"] = "CONSERVATIVE"
            if "危险" in normalized_text:
                payload["category"] = "EMERGENCY_RESPONSE"
            for command in commands:
                if command.get("action") in {"SET_SPEED", "ADJUST_SPEED"}:
                    command["action"] = "ADJUST_SPEED"
                    command.pop("target_speed_mps", None)
                    command["change"] = "DECREASE"

        if "施工路段" in normalized_text and any(
            token in normalized_text for token in ("并道", "减速")
        ):
            payload["category"] = "EMERGENCY_RESPONSE"
            payload["driving_style"] = "CONSERVATIVE"

        if "减速" in normalized_text and not any(
            command.get("action") in {"ADJUST_SPEED", "SET_SPEED", "STOP", "EMERGENCY_BRAKE"}
            for command in commands
        ):
            commands.insert(0, {"action": "ADJUST_SPEED", "change": "DECREASE"})

        has_avoid_language = any(
            token in normalized_text for token in ("绕开", "避让", "超越", "超车")
        )
        if not has_avoid_language:
            commands[:] = [
                command for command in commands if command.get("action") != "AVOID"
            ]

        if "慢车" in normalized_text:
            for command in commands:
                if command.get("action") == "AVOID":
                    command["action"] = "OVERTAKE"
                    command["target_type"] = "SLOW_VEHICLE"
                    command.setdefault("target_relation", "AHEAD")

        braking_boundary = classify_chinese_braking(normalized_text)
        if braking_boundary:
            candidate_actions = {"STOP", "EMERGENCY_BRAKE"}
            if braking_boundary.action == "ADJUST_SPEED":
                candidate_actions.add("ADJUST_SPEED")
            matched = False
            for command in commands:
                if command.get("action") not in candidate_actions:
                    continue
                matched = True
                command["action"] = braking_boundary.action
                if braking_boundary.action == "ADJUST_SPEED":
                    command["change"] = "DECREASE"
                else:
                    command.pop("change", None)
            if not matched and braking_boundary.urgency != "NORMAL":
                command = {"action": braking_boundary.action}
                if braking_boundary.action == "ADJUST_SPEED":
                    command["change"] = "DECREASE"
                commands.append(command)

            payload["urgency"] = braking_boundary.urgency
            actions = {command.get("action") for command in commands}
            if braking_boundary.action == "EMERGENCY_BRAKE":
                payload["category"] = "EMERGENCY_RESPONSE"
            elif actions <= {"STOP", "ADJUST_SPEED"}:
                payload["category"] = "BASIC_CONTROL"

        if "加塞" in normalized_text and "避让" in normalized_text:
            for command in commands:
                if command.get("action") == "EMERGENCY_BRAKE":
                    command.clear()
                    command.update(
                        {
                            "action": "AVOID",
                            "target_type": "VEHICLE",
                            "target_relation": "AHEAD",
                        }
                    )

        actions = [command.get("action") for command in commands]
        if ("回归原车道" in normalized_text or "恢复正常行驶" in normalized_text) and "RESUME" not in actions:
            commands.append({"action": "RESUME"})

        actions = [command.get("action") for command in commands]
        has_yielding_speed_change = any(
            command.get("action") == "ADJUST_SPEED"
            and command.get("purpose") == "YIELD"
            for command in commands
        )
        slow_vehicle_is_handled = (
            "慢车" in normalized_text and "OVERTAKE" in actions
        )
        if "慢车" in normalized_text and any(
            token in normalized_text for token in ("绕开", "超越", "超车")
        ) and "OVERTAKE" not in actions and not any(
            command.get("action") == "CHANGE_LANE"
            and command.get("purpose") == "OVERTAKE"
            for command in commands
        ):
            commands.insert(
                0,
                {
                    "action": "OVERTAKE",
                    "target_type": "SLOW_VEHICLE",
                    "target_relation": "AHEAD",
                },
            )
        elif (
            any(token in normalized_text for token in ("绕开", "避让"))
            and not {"AVOID", "YIELD"}.intersection(actions)
            and not has_yielding_speed_change
            and not slow_vehicle_is_handled
        ):
            if "行人" in normalized_text:
                target_type = "PEDESTRIAN"
            elif "锥桶" in normalized_text:
                target_type = "TRAFFIC_CONE"
            elif "施工" in normalized_text:
                target_type = "CONSTRUCTION_ZONE"
            else:
                target_type = "OBSTACLE"
            commands.insert(
                0,
                {
                    "action": "AVOID",
                    "target_type": target_type,
                    "target_relation": "AHEAD",
                },
            )
        return payload

    @staticmethod
    def _heuristic_confidence(status: str, missing_slots: list[str]) -> float:
        if status == "VALID":
            return 0.85
        if status == "NEEDS_CLARIFICATION":
            return max(0.2, 0.55 - 0.1 * len(missing_slots))
        return 0.1
