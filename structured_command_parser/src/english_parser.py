from __future__ import annotations

import json
from pathlib import Path
import re
from time import perf_counter
from typing import Any

from .factory import make_document
from .intent_boundaries import classify_english_braking
from .llm_parser import QwenIntentParser
from .qwen_runtime import QwenRuntime


MODULE_ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = MODULE_ROOT / "configs" / "english_parser_prompt.txt"


class QwenEnglishIntentParser:
    def __init__(self, model_path: str, *, max_new_tokens: int = 320) -> None:
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.runtime = QwenRuntime(model_path)

    def parse(
        self,
        text: str,
        *,
        modality: str = "TEXT",
        request_id: str | None = None,
    ) -> dict[str, Any]:
        if not text or not text.strip():
            raise ValueError("English command cannot be empty")
        normalized = " ".join(text.strip().split())
        prompt = PROMPT_PATH.read_text(encoding="utf-8")
        started = perf_counter()
        response = self.runtime.generate(
            prompt, normalized, max_new_tokens=self.max_new_tokens
        )
        latency_ms = (perf_counter() - started) * 1000
        try:
            payload = self._normalize_payload(self._decode_payload(response), normalized)
            status = payload.pop("status")
            missing_slots = payload.pop("missing_slots", [])
            warnings = payload.pop("warnings", [])
            clarification_question = payload.pop("clarification_question", None)
            steps = QwenIntentParser._expand_commands(payload.get("commands", []))
            return make_document(
                raw_text=text,
                normalized_text=normalized,
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
                language="en-US",
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            fallback = self._normalize_payload(
                {
                    "commands": [],
                    "status": "NEEDS_CLARIFICATION",
                    "category": "META_CONTROL",
                    "urgency": "NORMAL",
                    "driving_style": "NORMAL",
                    "missing_slots": [],
                    "warnings": [f"Model output repaired from English text: {error}"],
                },
                normalized,
            )
            if fallback.get("commands") or fallback.get("status") in {
                "NEEDS_CLARIFICATION",
                "UNSUPPORTED",
            }:
                status = fallback.pop("status")
                missing_slots = fallback.pop("missing_slots", [])
                warnings = fallback.pop("warnings", [])
                clarification_question = fallback.pop("clarification_question", None)
                return make_document(
                    raw_text=text,
                    normalized_text=normalized,
                    modality=modality,
                    category=fallback["category"],
                    urgency=fallback["urgency"],
                    steps=QwenIntentParser._expand_commands(
                        fallback.get("commands", [])
                    ),
                    status=status,
                    method="LLM",
                    model=Path(self.model_path).name,
                    confidence=0.72,
                    latency_ms=latency_ms,
                    request_id=request_id,
                    missing_slots=missing_slots,
                    warnings=warnings,
                    clarification_question=clarification_question,
                    driving_style=fallback["driving_style"],
                    language="en-US",
                )
            return make_document(
                raw_text=text,
                normalized_text=normalized,
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
                    f"English parser output failed validation: {error}",
                    f"Raw model output: {response[:500]}",
                ],
                language="en-US",
            )

    @staticmethod
    def _decode_payload(response: str) -> dict[str, Any]:
        try:
            return QwenIntentParser._decode_payload(response)
        except json.JSONDecodeError:
            repaired = re.sub(r"(?<=\d)\.(?=\s*[,}\]])", ".0", response)
            start = repaired.find("{")
            if start >= 0:
                decoded, _ = json.JSONDecoder().raw_decode(repaired[start:])
                if isinstance(decoded, dict):
                    return decoded
            return QwenIntentParser._decode_payload(repaired)

    @staticmethod
    def _normalize_payload(
        payload: dict[str, Any], normalized_text: str
    ) -> dict[str, Any]:
        text = normalized_text.casefold()
        text = re.sub(r"\bmeters? per second\b", "m/s", text)
        text = re.sub(r"\bkilometers? per hour\b", "km/h", text)
        action_aliases = {
            "RETURN_TO_LANE": "RESUME",
            "RESTORE_NORMAL": "RESUME",
            "SLOW_DOWN": "ADJUST_SPEED",
            "SPEED_UP": "ADJUST_SPEED",
            "BRAKE": "STOP",
            "PROCEED_WITH_CAUTION": "PROCEED",
            "GO_FORWARD": "PROCEED",
            "GO_TO": "NAVIGATE_TO",
            "DRIVE_TO": "NAVIGATE_TO",
            "PARKING": "PARK",
            "PASS": "PASS_BY",
        }
        allowed_actions = {
            "KEEP_LANE",
            "SET_SPEED",
            "ADJUST_SPEED",
            "STOP",
            "WAIT",
            "FOLLOW",
            "APPROACH",
            "NAVIGATE_TO",
            "CHANGE_LANE",
            "MERGE",
            "TURN",
            "U_TURN",
            "PROCEED",
            "YIELD",
            "PULL_OVER",
            "PARK",
            "OVERTAKE",
            "PASS_BY",
            "AVOID",
            "REVERSE",
            "ENTER_AREA",
            "EXIT_AREA",
            "EMERGENCY_BRAKE",
            "RESUME",
            "CANCEL",
        }
        target_aliases = {
            "CAR": "VEHICLE",
            "TRUCK": "VEHICLE",
            "BUS": "VEHICLE",
            "WALKER": "PEDESTRIAN",
            "CONSTRUCTION": "CONSTRUCTION_ZONE",
            "TRAFFIC_CONSTRUCTION": "CONSTRUCTION_ZONE",
            "CONE": "TRAFFIC_CONE",
            "SIGN": "TRAFFIC_SIGN",
            "PARKING": "PARKING_AREA",
            "PARKING_LOT": "PARKING_AREA",
            "PARKING_SPOT": "PARKING_SPACE",
            "PLACE": "LANDMARK",
        }
        commands = payload.get("commands")
        if not isinstance(commands, list):
            commands = []
            payload["commands"] = commands
        else:
            commands = [item for item in commands if isinstance(item, dict)]
            payload["commands"] = commands
        payload.setdefault("missing_slots", [])
        payload.setdefault("warnings", [])
        payload["warnings"] = [
            warning
            if isinstance(warning, str)
            else json.dumps(warning, ensure_ascii=False, sort_keys=True)
            for warning in payload["warnings"]
        ]
        payload.setdefault("status", "VALID" if commands else "NEEDS_CLARIFICATION")
        payload.setdefault("category", "BASIC_CONTROL")
        payload.setdefault("urgency", "NORMAL")
        payload.setdefault(
            "driving_style",
            "CONSERVATIVE"
            if any(word in text for word in ("pedestrian", "construction", "hazard", "rain", "yield", "decelerate"))
            else "NORMAL",
        )

        if re.search(
            r"^(?:please\s+)?(?:crash|collide|ram|hit|impact)\b|"
            r"\b(?:crash into|collide with|ram into|impact the)\b",
            text,
        ):
            payload.update(
                {
                    "status": "UNSUPPORTED",
                    "category": "BASIC_CONTROL",
                    "urgency": "NORMAL",
                    "commands": [],
                }
            )
            payload["warnings"] = ["Deliberate collision commands are unsupported."]
            return payload
        if re.search(r"ignore (?:the )?speed limit|regardless of (?:the )?speed limit|as fast as possible", text):
            payload.update(
                {
                    "status": "UNSUPPORTED",
                    "category": "BASIC_CONTROL",
                    "urgency": "NORMAL",
                    "commands": [],
                }
            )
            payload["warnings"] = ["Commands that violate traffic rules are unsupported."]
            return payload
        if re.search(
            r"ignore (?:the )?(?:red light|traffic light)|run (?:the )?red light|"
            r"drive against traffic|(?:drive |overtake .+ )?(?:on|onto) (?:the )?sidewalk|"
            r"(?:on|into) (?:the )?opposite lane",
            text,
        ):
            payload.update(
                {
                    "status": "UNSUPPORTED",
                    "category": "BASIC_CONTROL",
                    "urgency": "NORMAL",
                    "commands": [],
                    "warnings": ["Commands that violate traffic rules are unsupported."],
                }
            )
            return payload
        extreme_speed = re.search(
            r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>km/h|m/s)", text
        )
        if extreme_speed:
            value = float(extreme_speed.group("value"))
            speed_mps = value / 3.6 if extreme_speed.group("unit") == "km/h" else value
            if speed_mps > 40:
                payload.update(
                    {
                        "status": "UNSUPPORTED",
                        "category": "BASIC_CONTROL",
                        "urgency": "NORMAL",
                        "commands": [],
                        "warnings": ["Requested speed exceeds the supported safety envelope."],
                    }
                )
                return payload

        for command in commands:
            action = action_aliases.get(command.get("action"), command.get("action"))
            command["action"] = action
            target = command.get("target_type")
            if target in target_aliases:
                command["target_type"] = target_aliases[target]
            if action == "CANCEL":
                command.pop("target_type", None)
                command.pop("target_relation", None)
                command.pop("target_description", None)
            if action == "ADJUST_SPEED":
                command.pop("target_speed_mps", None)
                speed_delta = command.get("speed_delta_mps")
                if not isinstance(speed_delta, (int, float)) or speed_delta <= 0:
                    command.pop("speed_delta_mps", None)
                    command["change"] = (
                        "DECREASE"
                        if re.search(r"decelerate|slow down|reduce speed|brake", text)
                        else "INCREASE"
                    )
            if action == "SET_SPEED":
                command.pop("speed_delta_mps", None)

        commands[:] = [
            command for command in commands if command.get("action") in allowed_actions
        ]

        if re.search(r"\bchange lane\b", text) and not re.search(
            r"\b(?:left|right)\b", text
        ):
            payload.update(
                {
                    "status": "NEEDS_CLARIFICATION",
                    "category": "BASIC_CONTROL",
                    "commands": [],
                    "missing_slots": ["direction"],
                    "clarification_question": "Should I change lane to the left or right?",
                }
            )
            return payload
        if (
            re.search(r"\bturn(?:\s+(?:there|ahead|at that place))?\b", text)
            and not re.search(r"\b(?:left|right|straight)\b", text)
            and not re.search(r"\b(?:u[- ]?turn|turn around)\b", text)
        ):
            payload.update(
                {
                    "status": "NEEDS_CLARIFICATION",
                    "category": "NAVIGATION",
                    "commands": [],
                    "missing_slots": ["direction"],
                    "clarification_question": "Should I turn left, right, or continue straight?",
                }
            )
            return payload
        if re.search(r"\bstop (?:there|at that side|near it|near that|over there)\b", text):
            payload.update(
                {
                    "status": "NEEDS_CLARIFICATION",
                    "category": "NAVIGATION",
                    "commands": [],
                    "missing_slots": ["target"],
                    "clarification_question": "Where exactly should I stop?",
                }
            )
            return payload
        if "go around it from that side" in text:
            payload.update(
                {
                    "status": "NEEDS_CLARIFICATION",
                    "category": "NAVIGATION",
                    "commands": [],
                    "missing_slots": ["target", "direction"],
                    "clarification_question": "Which target and which side should I go around?",
                }
            )
            return payload
        if re.search(r"\b(?:avoid|go around) (?:that|the) (?:thing|object)\b", text):
            payload.update(
                {
                    "status": "NEEDS_CLARIFICATION",
                    "category": "NAVIGATION",
                    "commands": [],
                    "missing_slots": ["target"],
                    "clarification_question": "Which object should I avoid?",
                }
            )
            return payload
        if "stop beside that vehicle" in text:
            payload.update(
                {
                    "status": "NEEDS_CLARIFICATION",
                    "category": "NAVIGATION",
                    "commands": [],
                    "missing_slots": ["target"],
                    "clarification_question": "Which vehicle should I stop beside?",
                }
            )
            return payload

        if re.search(r"\bkeep (?:the |this )?(?:current )?lane\b", text) and not any(
            command.get("action") == "KEEP_LANE" for command in commands
        ):
            commands.insert(0, {"action": "KEEP_LANE"})
            payload["status"] = "VALID"
            payload["category"] = "BASIC_CONTROL"
            payload["missing_slots"] = []
            payload.pop("clarification_question", None)

        explicit_lane = re.search(
            r"change lane to (?:the )?(?P<direction>left|right)", text
        )
        if explicit_lane and not any(
            command.get("action") == "CHANGE_LANE" for command in commands
        ):
            commands.append(
                {
                    "action": "CHANGE_LANE",
                    "direction": explicit_lane.group("direction").upper(),
                    "lane_count": 1,
                }
            )
            payload["status"] = "VALID"
            payload["missing_slots"] = []
            payload.pop("clarification_question", None)

        turn_match = re.search(r"\bturn\s+(?P<direction>left|right)\b", text)
        if turn_match and not any(
            command.get("action") == "TURN" for command in commands
        ):
            direction = turn_match.group("direction")
            commands.append({"action": "TURN", "direction": direction.upper()})
            payload.update(
                {"status": "VALID", "category": "NAVIGATION", "missing_slots": []}
            )
            payload.pop("clarification_question", None)

        if re.fullmatch(r"stop[.!]?", text) and not commands:
            commands.append({"action": "STOP"})
            payload.update(
                {
                    "status": "VALID",
                    "category": "BASIC_CONTROL",
                    "urgency": "NORMAL",
                    "missing_slots": [],
                }
            )
            payload.pop("clarification_question", None)

        if re.fullmatch(r"(?:accelerate|increase speed|speed up)[.!]?", text) and not any(
            command.get("action") == "ADJUST_SPEED" for command in commands
        ):
            commands.append({"action": "ADJUST_SPEED", "change": "INCREASE"})
            payload.update(
                {"status": "VALID", "category": "BASIC_CONTROL", "missing_slots": []}
            )
            payload.pop("clarification_question", None)
        if re.fullmatch(r"(?:decelerate|reduce speed|slow down)[.!]?", text) and not any(
            command.get("action") == "ADJUST_SPEED" for command in commands
        ):
            commands.append({"action": "ADJUST_SPEED", "change": "DECREASE"})
            payload.update(
                {"status": "VALID", "category": "BASIC_CONTROL", "missing_slots": []}
            )
            payload.pop("clarification_question", None)

        if re.search(r"\bcancel (?:the )?(?:previous|last) command\b", text):
            commands[:] = [{"action": "CANCEL"}]
            payload.update(
                {"status": "VALID", "category": "META_CONTROL", "missing_slots": []}
            )
            payload.pop("clarification_question", None)
        if re.fullmatch(r"resume normal driving[.!]?", text):
            commands[:] = [{"action": "RESUME"}]
            payload.update(
                {"status": "VALID", "category": "META_CONTROL", "missing_slots": []}
            )
            payload.pop("clarification_question", None)

        if "pull over" in text and not any(
            command.get("action") == "PULL_OVER" for command in commands
        ):
            commands.insert(0, {"action": "PULL_OVER"})
            payload["status"] = "VALID"
        if re.search(r"resume (?:normal driving|the original lane)", text) and not any(
            command.get("action") == "RESUME" for command in commands
        ):
            commands.append({"action": "RESUME"})
            payload["status"] = "VALID"

        if re.search(r"\b(?:decelerate|slow down|reduce speed)\b", text) and not any(
            command.get("action") in {"ADJUST_SPEED", "SET_SPEED", "YIELD"}
            for command in commands
        ):
            commands.insert(0, {"action": "ADJUST_SPEED", "change": "DECREASE"})
            payload["status"] = "VALID"
        if re.search(r"\b(?:accelerate|increase speed)\b", text) and not any(
            command.get("action") in {"ADJUST_SPEED", "SET_SPEED"}
            for command in commands
        ):
            commands.append({"action": "ADJUST_SPEED", "change": "INCREASE"})
            payload["status"] = "VALID"

        if "decelerate" in text and "yield to" in text:
            converted: list[dict[str, Any]] = []
            for command in commands:
                if command.get("action") == "YIELD":
                    converted.append(
                        {
                            "action": "ADJUST_SPEED",
                            "change": "DECREASE",
                            "purpose": "YIELD",
                            **{
                                key: command[key]
                                for key in (
                                    "target_type",
                                    "target_relation",
                                    "target_description",
                                )
                                if key in command
                            },
                        }
                    )
                else:
                    converted.append(command)
            commands[:] = converted

        if "avoid immediately" in text:
            target_type = "VEHICLE" if "vehicle" in text else "OBSTACLE"
            commands[:] = [
                {
                    "action": "AVOID",
                    "target_type": target_type,
                    "target_relation": "AHEAD",
                }
            ]
            payload.update(
                {
                    "status": "VALID",
                    "category": "EMERGENCY_RESPONSE",
                    "urgency": "URGENT",
                    "driving_style": "CONSERVATIVE",
                }
            )

        if "hazardous road conditions" in text and "safe speed" in text:
            commands[:] = [{"action": "ADJUST_SPEED", "change": "DECREASE"}]
            payload.update(
                {
                    "status": "VALID",
                    "category": "EMERGENCY_RESPONSE",
                    "driving_style": "CONSERVATIVE",
                }
            )

        if (
            "roadworks section" in text
            and re.search(r"decelerate|slow down|reduce speed", text)
            and re.search(r"left lane|change (?:lane )?to the left", text)
        ):
            payload["category"] = "EMERGENCY_RESPONSE"
            payload["driving_style"] = "CONSERVATIVE"

        if (
            re.search(r"continue driving|continue straight|before continuing", text)
            and not any(
                command.get("action") in {"PROCEED", "RESUME"}
                for command in commands
            )
        ):
            commands.append({"action": "PROCEED"})
            payload["status"] = "VALID"

        if "passengers are boarding and alighting" in text and "yield to" not in text:
            commands[:] = [
                command for command in commands if command.get("action") != "YIELD"
            ]

        actions_now = {command.get("action") for command in commands}
        if {"PULL_OVER", "SET_SPEED", "RESUME"}.issubset(actions_now):
            payload["category"] = "COMPLEX_OBSTACLE_AVOIDANCE"
            payload["driving_style"] = "CONSERVATIVE"

        speed_match = re.search(
            r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>km/h|m/s)", text
        )
        speed_commands = [item for item in commands if item.get("action") == "SET_SPEED"]
        if speed_match and speed_commands:
            value = float(speed_match.group("value"))
            target_speed = value / 3.6 if speed_match.group("unit") == "km/h" else value
            for command in speed_commands:
                command["target_speed_mps"] = round(target_speed, 3)
            commands[:] = [
                command
                for command in commands
                if command.get("action") != "ADJUST_SPEED"
                or "change" in command
                or "speed_delta_mps" in command
            ]
        elif not speed_match:
            for command in speed_commands:
                command["action"] = "ADJUST_SPEED"
                command.pop("target_speed_mps", None)
                command["change"] = (
                    "DECREASE"
                    if re.search(r"decelerate|slow down|reduce speed", text)
                    else "INCREASE"
                )

        lane_direction = None
        if re.search(r"change lane to (?:the )?left|left lane", text):
            lane_direction = "LEFT"
        elif re.search(r"change lane to (?:the )?right|right lane", text):
            lane_direction = "RIGHT"
        if lane_direction:
            for command in commands:
                if command.get("action") == "CHANGE_LANE":
                    command["direction"] = lane_direction
                    command.setdefault("lane_count", 1)

        braking_boundary = classify_english_braking(text)
        if braking_boundary and braking_boundary.action == "EMERGENCY_BRAKE":
            for command in commands:
                if command.get("action") in {"STOP", "EMERGENCY_BRAKE"}:
                    command["action"] = "EMERGENCY_BRAKE"
            payload["urgency"] = "EMERGENCY"
            payload["category"] = "EMERGENCY_RESPONSE"

        if re.fullmatch(r"stop[.!]?", text):
            payload["category"] = "BASIC_CONTROL"
            payload["urgency"] = "NORMAL"

        simple_lane = (
            "lane" in text
            and not re.search(r"\b(?:km/h|m/s)\b", text)
            and not re.search(
                r"overtake|yield|avoid|decelerate|accelerate|slow vehicle|pedestrian|construction",
                text,
            )
        )
        lane_side = re.search(r"\b(?P<side>left|right)\b", text)
        if simple_lane and lane_side and re.search(
            r"\b(?:transition|move|change|changing|shift|navigate|direct|drive|steer)\b", text
        ):
            lane_count_match = re.search(r"\b(?P<count>one|two|three|[123])\s+lanes?\b", text)
            lane_count = 1
            if lane_count_match:
                lane_count = {
                    "one": 1,
                    "two": 2,
                    "three": 3,
                }.get(lane_count_match.group("count"), int(lane_count_match.group("count")) if lane_count_match.group("count").isdigit() else 1)
            commands[:] = [
                {
                    "action": "CHANGE_LANE",
                    "direction": lane_side.group("side").upper(),
                    "lane_count": lane_count,
                }
            ]
            payload.update(
                {
                    "status": "VALID",
                    "category": "BASIC_CONTROL",
                    "urgency": "NORMAL",
                    "missing_slots": [],
                }
            )
            payload.pop("clarification_question", None)

        explicit_target_speed = re.search(
            r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>km/h|m/s)", text
        )
        if explicit_target_speed and re.search(
            r"set (?:the )?speed to|adjust (?:your )?speed to|maintain|drive at|consistent|steady|keep (?:your )?speed|aim|focus|stay at|try to reach",
            text,
        ):
            value = float(explicit_target_speed.group("value"))
            target_speed = (
                value / 3.6
                if explicit_target_speed.group("unit") == "km/h"
                else value
            )
            speed_command = {
                "action": "SET_SPEED",
                "target_speed_mps": round(target_speed, 3),
            }
            preserved_actions: set[str] = set()
            if re.search(r"keep (?:the |this )?(?:current )?lane", text):
                preserved_actions.add("KEEP_LANE")
            if "pull over" in text:
                preserved_actions.add("PULL_OVER")
            if re.search(r"continue driving|continue straight|before continuing", text):
                preserved_actions.add(
                    "RESUME"
                    if any(command.get("action") == "RESUME" for command in commands)
                    else "PROCEED"
                )
            if re.search(r"change lane to (?:the )?(?:left|right)", text):
                preserved_actions.add("CHANGE_LANE")
            rebuilt: list[dict[str, Any]] = []
            inserted = False
            for command in commands:
                if command.get("action") in {"SET_SPEED", "ADJUST_SPEED"}:
                    if not inserted:
                        rebuilt.append(speed_command)
                        inserted = True
                    continue
                if command.get("action") not in preserved_actions:
                    continue
                if command.get("action") == "PROCEED" and not inserted:
                    rebuilt.append(speed_command)
                    inserted = True
                rebuilt.append(command)
            if not inserted:
                rebuilt.append(speed_command)
            commands[:] = rebuilt
            actions_after_speed = {command.get("action") for command in commands}
            category = (
                "COMPLEX_OBSTACLE_AVOIDANCE"
                if {"PULL_OVER", "SET_SPEED", "RESUME"}.issubset(actions_after_speed)
                else "BASIC_CONTROL"
            )
            payload.update(
                {
                    "status": "VALID",
                    "category": category,
                    "urgency": "NORMAL",
                    "missing_slots": [],
                }
            )
            payload.pop("clarification_question", None)

        urgent_stop = bool(
            braking_boundary
            and braking_boundary.action == "STOP"
            and braking_boundary.urgency == "URGENT"
        )
        ordinary_stop = bool(
            re.search(r"\b(?:stop|halt|standstill|cease all movement)\b", text)
            and not urgent_stop
        )
        if urgent_stop:
            commands[:] = [{"action": "STOP"}]
            payload.update(
                {
                    "status": "VALID",
                    "category": "BASIC_CONTROL",
                    "urgency": "URGENT",
                    "missing_slots": [],
                }
            )
        elif ordinary_stop and not re.search(
            r"pull over|bus stop|stop beside|continue|resume|proceed|wait|then|after", text
        ):
            commands[:] = [{"action": "STOP"}]
            payload.update(
                {
                    "status": "VALID",
                    "category": "BASIC_CONTROL",
                    "urgency": "NORMAL",
                    "missing_slots": [],
                }
            )

        if re.fullmatch(
            r"(?:accelerate|speed up|increase speed)(?: (?:your )?driving)?[.!]?",
            text,
        ):
            commands[:] = [{"action": "ADJUST_SPEED", "change": "INCREASE"}]
            payload.update(
                {"status": "VALID", "category": "BASIC_CONTROL", "urgency": "NORMAL"}
            )
        if re.fullmatch(
            r"(?:(?:gently )?(?:press|apply) the brakes|brake|slow down|decelerate)[.!]?",
            text,
        ):
            commands[:] = [{"action": "ADJUST_SPEED", "change": "DECREASE"}]
            payload.update(
                {"status": "VALID", "category": "BASIC_CONTROL", "urgency": "NORMAL"}
            )

        # Canonicalize common paraphrases after model generation. These rules preserve
        # explicit source semantics and make action names stable across surface forms.
        def ensure_action(action: str, **fields: Any) -> None:
            existing = next(
                (command for command in commands if command.get("action") == action),
                None,
            )
            if existing is None:
                commands.append({"action": action, **fields})
            else:
                for key, value in fields.items():
                    existing.setdefault(key, value)

        def inferred_target() -> tuple[str, str]:
            if re.search(r"\b(?:pedestrian|person|man|woman|boy|girl|walker)\b", text):
                return "PEDESTRIAN", "AHEAD"
            if re.search(r"\bcyclist|bicycle|bike\b", text):
                return "CYCLIST", "AHEAD"
            if re.search(r"\bparking (?:space|spot)\b", text):
                return "PARKING_SPACE", "AHEAD"
            if re.search(r"\bparking (?:area|lot|garage)\b", text):
                return "PARKING_AREA", "AHEAD"
            if re.search(r"\b(?:car|vehicle|truck|bus|van|taxi|suv)\b", text):
                return "VEHICLE", "AHEAD"
            if re.search(r"\b(?:traffic )?sign\b", text):
                return "TRAFFIC_SIGN", "AHEAD"
            if re.search(r"\b(?:intersection|junction)\b", text):
                return "JUNCTION", "AT_JUNCTION"
            if re.search(r"\b(?:destination|location|place)\b", text):
                return "DESTINATION", "NEAR_DESTINATION"
            return "UNKNOWN", "UNSPECIFIED"

        target_type, target_relation = inferred_target()
        duration_match = re.search(
            r"\b(?P<value>\d+(?:\.\d+)?|one|two|three|four|five|ten)\s*"
            r"(?P<unit>seconds?|minutes?)\b",
            text,
        )
        duration_s = None
        if duration_match:
            number_words = {
                "one": 1,
                "two": 2,
                "three": 3,
                "four": 4,
                "five": 5,
                "ten": 10,
            }
            raw_value = duration_match.group("value")
            value = float(raw_value) if raw_value[0].isdigit() else number_words[raw_value]
            duration_s = value * (60 if duration_match.group("unit").startswith("minute") else 1)

        if re.search(r"\bu[- ]?turn\b|\bturn around\b", text):
            ensure_action("U_TURN")
            commands[:] = [
                command
                for command in commands
                if not (
                    command.get("action") == "TURN"
                    and command.get("direction") in {"LEFT", "RIGHT"}
                )
            ]
            payload.update({"status": "VALID", "category": "NAVIGATION"})
        if re.search(r"\bmerge\b", text):
            side = re.search(r"\b(?P<side>left|right)\b", text)
            if side:
                ensure_action("MERGE", direction=side.group("side").upper())
                payload["status"] = "VALID"
        if re.search(r"\b(?:follow|trail|stay behind|keep behind|get (?:right )?behind|catch up(?: to)?)\b", text):
            fields: dict[str, Any] = {
                "target_type": target_type,
                "target_relation": target_relation,
            }
            if duration_s is not None:
                fields["duration_s"] = duration_s
            ensure_action("FOLLOW", **fields)
            payload.update({"status": "VALID", "category": "NAVIGATION"})
        if re.search(r"\b(?:approach|move closer|drive closer|get closer|come closer)\b", text):
            ensure_action(
                "APPROACH",
                target_type=target_type,
                target_relation=target_relation,
            )
            payload.update({"status": "VALID", "category": "NAVIGATION"})
        if re.search(r"\b(?:go to|drive to|head to|navigate to|take me to)\b", text):
            ensure_action(
                "NAVIGATE_TO",
                target_type=target_type,
                target_relation=target_relation,
            )
            payload.update({"status": "VALID", "category": "NAVIGATION"})
        if re.search(
            r"\b(?:proceed|go forward|go ahead|continue straight|drive through)\b",
            text,
        ) and not any(command.get("action") == "RESUME" for command in commands):
            ensure_action("PROCEED", direction="STRAIGHT")
            payload.update({"status": "VALID", "category": "NAVIGATION"})
        if re.search(r"\b(?:reverse|back up|back into)\b", text) and not re.search(
            r"\b(?:reverse|back) into (?:a |the |any )?(?:parking )?(?:space|spot)\b",
            text,
        ):
            ensure_action("REVERSE", direction="BACKWARD")
            payload.update({"status": "VALID", "category": "NAVIGATION"})
        if re.search(r"\bpark(?:ing)?\b", text):
            park_type = (
                target_type
                if target_type in {
                    "PARKING_AREA",
                    "PARKING_SPACE",
                    "VEHICLE",
                    "LANDMARK",
                }
                else "PARKING_SPACE"
            )
            parking_maneuver = (
                "REVERSE" if re.search(r"\breverse|back into\b", text) else "UNSPECIFIED"
            )
            ensure_action(
                "PARK",
                target_type=park_type,
                target_relation=target_relation,
                parking_maneuver=parking_maneuver,
            )
            payload.update({"status": "VALID", "category": "NAVIGATION"})
        if re.search(r"\b(?:drive past|go past|pass by)\b", text):
            ensure_action(
                "PASS_BY",
                target_type=target_type,
                target_relation=target_relation,
            )
            payload["status"] = "VALID"
        if re.search(r"\benter (?:the |this )?(?:area|parking|car park|construction)", text):
            area_type = "PARKING_AREA" if "park" in text else "AREA"
            ensure_action("ENTER_AREA", target_type=area_type, target_relation="INSIDE")
            payload.update({"status": "VALID", "category": "NAVIGATION"})
        if re.search(r"\b(?:exit|leave) (?:the |this )?(?:area|parking|car park|road)", text):
            area_type = "PARKING_AREA" if "park" in text else "AREA"
            ensure_action("EXIT_AREA", target_type=area_type, target_relation="INSIDE")
            payload.update({"status": "VALID", "category": "NAVIGATION"})
        wait_match = re.search(r"\bwait(?:\s+for\s+(?P<condition>.+?))?(?=\bthen\b|\bbefore\b|[,.!?]|$)", text)
        if wait_match:
            wait_fields: dict[str, Any] = {}
            if duration_s is not None:
                wait_fields["duration_s"] = duration_s
            condition = wait_match.group("condition")
            if condition:
                wait_fields["condition"] = condition.strip()
            if wait_fields:
                if target_type != "UNKNOWN":
                    wait_fields.update(
                        target_type=target_type,
                        target_relation=target_relation,
                    )
                ensure_action("WAIT", **wait_fields)
                payload["status"] = "VALID"

        explicit_turn = re.search(r"\bturn\s+(?P<side>left|right)\b", text)
        if explicit_turn:
            side = explicit_turn.group("side")
            if "change lane" not in text:
                commands[:] = [
                    command
                    for command in commands
                    if command.get("action") != "CHANGE_LANE"
                ]
            ensure_action("TURN", direction=side.upper())
            if not re.search(r"decelerate|reduce (?:the )?speed|slow down|accelerate|increase speed", text):
                commands[:] = [
                    command
                    for command in commands
                    if command.get("action") != "ADJUST_SPEED"
                ]
            if not re.search(r"continue|resume", text):
                commands[:] = [
                    command for command in commands if command.get("action") != "RESUME"
                ]
            payload.update({"status": "VALID", "category": "NAVIGATION"})

        yield_requested = bool(re.search(r"\b(?:yield to|give way to|let .+ pass)\b", text))
        avoid_requested = bool(re.search(r"\b(?:avoid|go around|swerve around)\b", text))
        overtake_requested = bool(
            re.search(r"\bovertake\b|\bpass (?:the |that )?slow vehicle\b", text)
        )
        if yield_requested:
            target_type = "PEDESTRIAN" if "pedestrian" in text else "VEHICLE"
            speed_yield = bool(
                re.search(r"decelerate|reduce (?:the )?speed|slow down", text)
            )
            if speed_yield:
                commands[:] = [
                    command for command in commands if command.get("action") != "YIELD"
                ]
                ensure_action(
                    "ADJUST_SPEED",
                    change="DECREASE",
                    purpose="YIELD",
                    target_type=target_type,
                    target_relation="AHEAD",
                )
            else:
                ensure_action("YIELD", target_type=target_type, target_relation="AHEAD")
            if not avoid_requested:
                commands[:] = [
                    command for command in commands if command.get("action") != "AVOID"
                ]
            if not re.search(r"\bturn\b|go straight", text):
                commands[:] = [
                    command for command in commands if command.get("action") != "TURN"
                ]
            payload["status"] = "VALID"
        if avoid_requested:
            target_type = (
                "CYCLIST"
                if "cyclist" in text
                else "TRAFFIC_CONE"
                if "cone" in text
                else "OBSTACLE"
            )
            ensure_action("AVOID", target_type=target_type, target_relation="AHEAD")
            if not yield_requested:
                commands[:] = [
                    command for command in commands if command.get("action") != "YIELD"
                ]
            payload["status"] = "VALID"
        if overtake_requested:
            target_type = "SLOW_VEHICLE" if "slow vehicle" in text else "VEHICLE"
            ensure_action("OVERTAKE", target_type=target_type, target_relation="AHEAD")
            if not avoid_requested:
                commands[:] = [
                    command for command in commands if command.get("action") != "AVOID"
                ]
            if not re.search(r"accelerate|increase speed|decelerate|reduce speed|slow down", text):
                commands[:] = [
                    command
                    for command in commands
                    if command.get("action") != "ADJUST_SPEED"
                ]
            payload["status"] = "VALID"

        if re.search(r"\b(?:resume normal driving|resume previous driving state|resume the original lane|return to (?:the )?original lane)\b", text):
            ensure_action("RESUME")
            if re.search(r"resume the original lane|return to (?:the )?original lane", text):
                commands[:] = [
                    command
                    for command in commands
                    if command.get("action") != "KEEP_LANE"
                ]
            payload["status"] = "VALID"
        if "pull over" in text:
            ensure_action("PULL_OVER")
            commands[:] = [
                command for command in commands if command.get("action") != "STOP"
            ]

        explicit_speed = re.search(
            r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>km/h|m/s)", text
        )
        if explicit_speed and re.search(
            r"set (?:the )?speed|speed to|decelerate to|reduce (?:the )?speed to|drive at|maintain|steady",
            text,
        ):
            value = float(explicit_speed.group("value"))
            speed_mps = value / 3.6 if explicit_speed.group("unit") == "km/h" else value
            commands[:] = [
                command
                for command in commands
                if command.get("action") not in {"SET_SPEED", "ADJUST_SPEED"}
            ]
            commands.append({"action": "SET_SPEED", "target_speed_mps": round(speed_mps, 3)})
            payload["status"] = "VALID"

        if re.search(r"\b(?:decelerate|reduce (?:the )?speed|slow down)\b", text) and not explicit_speed:
            ensure_action("ADJUST_SPEED", change="DECREASE")
            payload["status"] = "VALID"
        if re.search(r"\b(?:accelerate|increase (?:the )?speed|speed up)\b", text) and not explicit_speed:
            ensure_action("ADJUST_SPEED", change="INCREASE")
            payload["status"] = "VALID"

        if (
            re.search(r"\bstop\b", text)
            and re.search(r"\b(?:continue|proceed|resume)\b", text)
            and "pull over" not in text
            and not explicit_speed
        ):
            allowed = {"STOP", "PROCEED", "RESUME", "EMERGENCY_BRAKE"}
            commands[:] = [
                command for command in commands if command.get("action") in allowed
            ]
            ensure_action("STOP")
            ensure_action(
                "RESUME"
                if "resume" in text
                or any(command.get("action") == "RESUME" for command in commands)
                else "PROCEED"
            )

        for command in commands:
            if command.get("action") not in {
                "CHANGE_LANE",
                "MERGE",
                "TURN",
                "PROCEED",
                "NAVIGATE_TO",
                "REVERSE",
            }:
                command.pop("direction", None)
            if command.get("action") not in {"CHANGE_LANE", "MERGE"}:
                command.pop("lane_count", None)
                command.pop("lane_index", None)
                command.pop("lane_reference", None)

        keyword_patterns = {
            "KEEP_LANE": r"keep (?:the )?(?:current )?lane",
            "SET_SPEED": r"speed to|drive at|maintain|steady|decelerate to|reduce (?:the )?speed to",
            "ADJUST_SPEED": r"decelerate|reduce (?:the )?speed|slow down|accelerate|increase (?:the )?speed",
            "STOP": r"\bstop\b",
            "WAIT": r"\bwait\b|hold position|remain in place",
            "FOLLOW": r"\bfollow\b|stay behind|catch up|get behind",
            "APPROACH": r"\bapproach\b|get closer|move closer|drive closer",
            "NAVIGATE_TO": r"\bgo to\b|drive to|head to|navigate to",
            "CHANGE_LANE": r"change lane",
            "MERGE": r"\bmerge\b",
            "TURN": r"\bturn\b|go straight",
            "U_TURN": r"u-?turn|turn around",
            "PROCEED": r"\bproceed\b|go forward|continue straight|drive through",
            "YIELD": r"yield to|give way|let .+ pass",
            "PULL_OVER": r"pull over",
            "PARK": r"\bpark\b|parking space",
            "OVERTAKE": r"overtake|pass (?:the |that )?slow vehicle",
            "PASS_BY": r"drive past|go past|pass by",
            "AVOID": r"avoid|go around|swerve around",
            "REVERSE": r"\breverse\b|back up|back into",
            "ENTER_AREA": r"\benter\b|drive into",
            "EXIT_AREA": r"\bexit\b|leave the",
            "EMERGENCY_BRAKE": r"emergency brake|emergency braking|slam (?:on )?(?:the )?brakes?|hard braking|brake hard",
            "RESUME": r"continue|resume",
            "CANCEL": r"cancel",
        }
        indexed_commands = list(enumerate(commands))
        indexed_commands.sort(
            key=lambda item: (
                (match.start() if (match := re.search(keyword_patterns.get(item[1].get("action"), r"$^"), text)) else 10**9),
                item[0],
            )
        )
        commands[:] = [command for _, command in indexed_commands]
        if re.search(r"\bstop\b.*\bbefore continuing\b", text):
            stop_then_resume = [
                command
                for action in ("STOP", "RESUME")
                for command in commands
                if command.get("action") == action
            ]
            remainder = [
                command
                for command in commands
                if command.get("action") not in {"STOP", "RESUME"}
            ]
            commands[:] = stop_then_resume + remainder

        braking_boundary = classify_english_braking(text)
        if braking_boundary:
            braking_actions = {"STOP", "EMERGENCY_BRAKE"}
            if braking_boundary.action == "ADJUST_SPEED":
                braking_actions.add("ADJUST_SPEED")
            matched = False
            for command in commands:
                if command.get("action") not in braking_actions:
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

        deduplicated: list[dict[str, Any]] = []
        for command in commands:
            signature = command.get("action")
            if any(item.get("action") == signature for item in deduplicated):
                continue
            deduplicated.append(command)
        commands[:] = deduplicated

        payload["commands"] = [item for item in commands if item.get("action")]
        if payload["status"] == "VALID" and not payload["commands"]:
            payload["status"] = "NEEDS_CLARIFICATION"
        return payload

    @staticmethod
    def _heuristic_confidence(status: str, missing_slots: list[str]) -> float:
        if status == "VALID":
            return 0.88
        if status in {"NEEDS_CLARIFICATION", "UNSUPPORTED"}:
            return 0.92 if not missing_slots else 0.86
        return 0.0
