"""Generic text-instruction finite state machine.

The FSM converts a scheduled natural-language driving instruction into a
small, scene-agnostic semantic intent.  It deliberately never branches on
command ids, event ids or scene ids: the same rules apply to every scene and
every scheduled text command.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch


PARSED_INTENTS = (
    "KEEP_LANE",
    "SET_SPEED",
    "DECELERATE",
    "EMERGENCY_BRAKE",
    "YIELD",
    "CHANGE_LANE_LEFT",
    "CHANGE_LANE_RIGHT",
    "STOP",
    "RESUME",
    "TURN_LEFT",
    "TURN_RIGHT",
)

INTENT_TO_ACTION = {
    "KEEP_LANE": "keep_lane",
    "SET_SPEED": "accelerate",
    "DECELERATE": "decelerate",
    "EMERGENCY_BRAKE": "emergency_brake",
    "YIELD": "decelerate",
    "CHANGE_LANE_LEFT": "lane_change_left",
    "CHANGE_LANE_RIGHT": "lane_change_right",
    "STOP": "stop",
    "RESUME": "keep_lane",
    "TURN_LEFT": "turn_left",
    "TURN_RIGHT": "turn_right",
}

_SPEED_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:km/?h|kmh|kph|公里每小时|公里/小时|迈)"
)


@dataclass
class ParsedInstruction:
    parsed_intent: str = "KEEP_LANE"
    requested_lane_direction: str | None = None
    target_speed_kmh: float | None = None
    confidence: float = 1.0
    source_text: str = ""
    semantic_goal: tuple[str, ...] = ()


def _intent_from_goals(goals: Sequence[str]) -> tuple[str | None, str | None, float | None]:
    lowered = [str(goal).lower() for goal in goals]
    if "lane_change_left" in lowered:
        return "CHANGE_LANE_LEFT", "left", None
    if "lane_change_right" in lowered:
        return "CHANGE_LANE_RIGHT", "right", None
    if "turn_left" in lowered:
        return "TURN_LEFT", None, None
    if "turn_right" in lowered:
        return "TURN_RIGHT", None, None
    if "emergency_brake" in lowered or "emergency" in lowered:
        return "EMERGENCY_BRAKE", None, None
    if "stop" in lowered or "stop_if_needed" in lowered:
        return "YIELD", None, None
    if "yield" in lowered:
        return "YIELD", None, None
    if "decelerate" in lowered:
        return "DECELERATE", None, None
    if "resume" in lowered or "resume_speed" in lowered:
        return "RESUME", None, None
    if "accelerate" in lowered or "set_speed" in lowered:
        return "SET_SPEED", None, None
    return None, None, None


class GenericInstructionFSM:
    """Parse scheduled text commands into generic semantic intents."""

    def __init__(
        self,
        default_speed_kmh: float = 40.0,
        parser: Any | None = None,
    ) -> None:
        self.default_speed_kmh = float(default_speed_kmh)
        self.parser = parser
        self._token_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._parse_cache: dict[str, dict[str, Any]] = {}

    def active_command(
        self,
        commands: Sequence[Mapping[str, Any]],
        progress_m: float,
    ) -> dict[str, Any]:
        """Return the newest route-triggered command whose window is active."""

        active: dict[str, Any] = {
            "text": "Continue driving safely in the current lane.",
        }
        for command in sorted(
            commands,
            key=self._command_trigger_m,
        ):
            trigger_m = self._command_trigger_m(command)
            if progress_m + 1e-6 < trigger_m:
                break
            end_progress_m = self._command_end_m(command)
            if end_progress_m is not None and progress_m >= end_progress_m - 1e-6:
                continue
            active = dict(command)
        return active

    @staticmethod
    def _command_trigger_m(command: Mapping[str, Any]) -> float:
        for key in ("trigger_progress_m", "activate_at_m", "announce_at_m"):
            value = command.get(key)
            if value is not None:
                return float(value)
        return 0.0

    @staticmethod
    def _command_end_m(command: Mapping[str, Any]) -> float | None:
        for key in ("end_progress_m", "deactivate_at_m"):
            value = command.get(key)
            if value is not None:
                return float(value)
        return None

    def parse(
        self,
        command: Mapping[str, Any],
        *,
        use_parser_model: bool = True,
    ) -> ParsedInstruction:
        """Map one text command to a generic parsed intent."""

        source_text = str(
            command.get("text")
            or command.get("source_text")
            or command.get("voice_text")
            or command.get("normalized_text")
            or command.get("parser_text_en")
            or ""
        )
        semantic_goal = tuple(
            str(item)
            for item in command.get("semantic_goal", [])
            if isinstance(item, str)
        )
        parsed = self._parse_text_rules(
            source_text,
            semantic_goal=semantic_goal,
        )
        structured = command.get("structured_command")
        if (
            isinstance(structured, dict)
            and parsed.parsed_intent == "KEEP_LANE"
        ):
            parsed = self._merge_structured_command(parsed, structured)
        if (
            parsed.parsed_intent == "KEEP_LANE"
            and use_parser_model
            and self.parser is not None
            and source_text
        ):
            model_parsed = self._parser_result(source_text, command)
            parsed = self._merge_parser_result(parsed, model_parsed)
        parsed.source_text = source_text
        parsed.semantic_goal = semantic_goal
        return parsed

    @staticmethod
    def _merge_structured_command(
        parsed: ParsedInstruction,
        structured: Mapping[str, Any],
    ) -> ParsedInstruction:
        action = str(structured.get("action", "")).upper()
        mapping = {
            "SET_SPEED": "SET_SPEED",
            "KEEP_LANE": "KEEP_LANE",
            "DECELERATE": "DECELERATE",
            "STOP": "STOP",
            "EMERGENCY_BRAKE": "EMERGENCY_BRAKE",
            "CHANGE_LANE": "CHANGE_LANE_LEFT",
            "TURN": "TURN_LEFT",
        }
        intent = mapping.get(action)
        if intent is None:
            return parsed
        direction = str(structured.get("direction", "")).upper()
        if intent == "CHANGE_LANE_LEFT":
            intent = (
                "CHANGE_LANE_RIGHT"
                if direction == "RIGHT"
                else "CHANGE_LANE_LEFT"
            )
        elif intent == "TURN_LEFT":
            intent = "TURN_RIGHT" if direction == "RIGHT" else "TURN_LEFT"
        speed = structured.get("target_speed_kmh")
        try:
            speed = float(speed) if speed is not None else parsed.target_speed_kmh
        except (TypeError, ValueError):
            speed = parsed.target_speed_kmh
        return ParsedInstruction(
            parsed_intent=intent,
            requested_lane_direction=(
                "left" if intent == "CHANGE_LANE_LEFT"
                else "right" if intent == "CHANGE_LANE_RIGHT"
                else None
            ),
            target_speed_kmh=speed,
            confidence=parsed.confidence,
            source_text=parsed.source_text,
            semantic_goal=parsed.semantic_goal,
        )

    def _parse_text_rules(
        self,
        text: str,
        *,
        semantic_goal: Sequence[str] = (),
    ) -> ParsedInstruction:
        lowered = text.lower()
        speed = self._extract_speed(text)

        if "变道" in lowered or "避让" in lowered or "换道" in lowered:
            if "左" in lowered:
                return ParsedInstruction(
                    parsed_intent="CHANGE_LANE_LEFT",
                    requested_lane_direction="left",
                    target_speed_kmh=speed,
                )
            if "右" in lowered:
                return ParsedInstruction(
                    parsed_intent="CHANGE_LANE_RIGHT",
                    requested_lane_direction="right",
                    target_speed_kmh=speed,
                )
            intent, direction, goal_speed = _intent_from_goals(semantic_goal)
            if intent in {"CHANGE_LANE_LEFT", "CHANGE_LANE_RIGHT"}:
                return ParsedInstruction(
                    parsed_intent=intent,
                    requested_lane_direction=direction,
                    target_speed_kmh=goal_speed if goal_speed is not None else speed,
                )
        if "紧急" in lowered and ("刹" in lowered or "停车" in lowered):
            return ParsedInstruction(
                parsed_intent="EMERGENCY_BRAKE", target_speed_kmh=0.0
            )
        if "让行" in lowered or "横穿" in lowered or "避让行人" in lowered:
            return ParsedInstruction(
                parsed_intent="YIELD", target_speed_kmh=speed or 10.0
            )
        if "加塞" in lowered or "急刹" in lowered:
            return ParsedInstruction(
                parsed_intent="DECELERATE", target_speed_kmh=speed or 18.0
            )
        if "减速" in lowered or "降低" in lowered or "慢" in lowered:
            return ParsedInstruction(
                parsed_intent="DECELERATE", target_speed_kmh=speed
            )
        if "提速" in lowered or "加速" in lowered or "恢复车速" in lowered:
            return ParsedInstruction(
                parsed_intent="SET_SPEED", target_speed_kmh=speed
            )
        if "恢复" in lowered or "结束" in lowered:
            return ParsedInstruction(
                parsed_intent="RESUME", target_speed_kmh=speed
            )
        if "停车" in lowered or "停止" in lowered or "停住" in lowered:
            return ParsedInstruction(parsed_intent="STOP", target_speed_kmh=0.0)
        if "右转" in lowered or "向右转" in lowered:
            return ParsedInstruction(
                parsed_intent="TURN_RIGHT", target_speed_kmh=speed or 15.0
            )
        if "左转" in lowered or "向左转" in lowered:
            return ParsedInstruction(
                parsed_intent="TURN_LEFT", target_speed_kmh=speed or 15.0
            )
        if "保持" in lowered or "继续" in lowered or "巡航" in lowered:
            return ParsedInstruction(
                parsed_intent="KEEP_LANE", target_speed_kmh=speed
            )

        intent, direction, goal_speed = _intent_from_goals(semantic_goal)
        if intent is not None:
            return ParsedInstruction(
                parsed_intent=intent,
                requested_lane_direction=direction,
                target_speed_kmh=goal_speed if goal_speed is not None else speed,
            )
        return ParsedInstruction(parsed_intent="KEEP_LANE", target_speed_kmh=speed)

    @staticmethod
    def _extract_speed(text: str) -> float | None:
        match = _SPEED_PATTERN.search(text)
        if match is None:
            return None
        try:
            return max(0.0, min(float(match.group(1)), 100.0))
        except ValueError:
            return None

    def _parser_result(
        self,
        source_text: str,
        command: Mapping[str, Any],
    ) -> dict[str, Any]:
        key = str(command.get("id") or source_text)
        if key in self._parse_cache:
            return self._parse_cache[key]
        try:
            result = self.parser.parse_text(
                source_text,
                request_id=f"fsm-{key}",
                modality="TEXT",
                source_text=source_text,
                source_language="zh-CN",
            )
        except Exception:
            result = {}
        parse_result = result.get("parse_result") or {}
        self._parse_cache[key] = parse_result
        return parse_result

    @staticmethod
    def _merge_parser_result(
        parsed: ParsedInstruction,
        parse_result: Mapping[str, Any],
    ) -> ParsedInstruction:
        if str(parse_result.get("status", "")) != "VALID":
            return parsed
        intent = parsed.parsed_intent
        steps = (parse_result.get("intent") or {}).get("steps") or []
        if steps:
            action = str(steps[0].get("action", "")).upper()
            mapping = {
                "KEEP_LANE": "KEEP_LANE",
                "ADJUST_SPEED": "SET_SPEED",
                "CHANGE_LANE": "CHANGE_LANE_LEFT",
                "STOP": "STOP",
                "EMERGENCY_BRAKE": "EMERGENCY_BRAKE",
                "PARK": "YIELD",
                "TURN": "TURN_LEFT",
            }
            merged = mapping.get(action)
            if merged is not None and intent == "KEEP_LANE":
                parsed = ParsedInstruction(
                    parsed_intent=merged,
                    requested_lane_direction=(
                        "left" if merged == "CHANGE_LANE_LEFT" else None
                    ),
                    target_speed_kmh=parsed.target_speed_kmh,
                    confidence=float(parse_result.get("confidence", 0.0) or 0.0),
                )
        return parsed

    def semantic_text(self, parsed: ParsedInstruction) -> str:
        """Deterministic English text fed to the text encoder."""

        speed = parsed.target_speed_kmh
        if speed is None:
            speed = self.default_speed_kmh
        speed = max(0.0, min(float(speed), 100.0))
        templates = {
            "KEEP_LANE": "Keep the current lane at {speed:.1f} kilometers per hour.",
            "SET_SPEED": "Accelerate smoothly to {speed:.1f} kilometers per hour when safe.",
            "DECELERATE": "Slow down smoothly to {speed:.1f} kilometers per hour.",
            "EMERGENCY_BRAKE": "Brake immediately.",
            "YIELD": "Slow down and yield to the hazard ahead.",
            "CHANGE_LANE_LEFT": "Change to the left lane when it is safe.",
            "CHANGE_LANE_RIGHT": "Change to the right lane when it is safe.",
            "STOP": "Stop the vehicle at a safe position.",
            "RESUME": "Resume normal driving when safe.",
            "TURN_LEFT": "Turn left safely at the next junction.",
            "TURN_RIGHT": "Turn right safely at the next junction.",
        }
        if parsed.parsed_intent in {"EMERGENCY_BRAKE", "STOP"}:
            return templates[parsed.parsed_intent]
        return templates[parsed.parsed_intent].format(speed=speed)

    def encode_tokens(
        self,
        parsed: ParsedInstruction,
        *,
        cache_key: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the semantic text into ModernBERT hidden features."""

        if self.parser is None:
            raise RuntimeError("text encoder is unavailable")
        text = self.semantic_text(parsed)
        key = cache_key or text
        cached = self._token_cache.get(key)
        if cached is not None:
            return cached
        parser = self.parser.parser
        parser.load()
        encoded = parser.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=parser.max_length,
        )
        encoded = {
            name: tensor.to(parser.device) for name, tensor in encoded.items()
        }
        with torch.inference_mode():
            tokens = parser.model.backbone(**encoded).last_hidden_state
        result = (
            tokens.detach().float().cpu(),
            encoded["attention_mask"].detach().bool().cpu(),
        )
        self._token_cache[key] = result
        return result

    def canonical_decision(
        self,
        parsed: ParsedInstruction,
        *,
        frame_id: str,
        request_id: str,
        risk: Mapping[str, Any],
        ego_speed_kmh: float,
    ) -> dict[str, Any]:
        """Build the deterministic text-envelope decision for the safety gate."""

        action = INTENT_TO_ACTION[parsed.parsed_intent]
        target_speed = parsed.target_speed_kmh
        if target_speed is None:
            target_speed = self.default_speed_kmh
        target_speed = max(0.0, min(float(target_speed), 100.0))

        if parsed.parsed_intent == "DECELERATE":
            ceiling = (
                target_speed
                if parsed.target_speed_kmh is not None
                else min(self.default_speed_kmh, max(10.0, ego_speed_kmh))
            )
            target_speed = max(0.0, ceiling)
        elif parsed.parsed_intent == "YIELD":
            target_speed = min(target_speed, 10.0)
        elif parsed.parsed_intent in {"STOP", "EMERGENCY_BRAKE"}:
            target_speed = 0.0
        elif parsed.parsed_intent in {
            "CHANGE_LANE_LEFT",
            "CHANGE_LANE_RIGHT",
            "TURN_LEFT",
            "TURN_RIGHT",
        }:
            target_speed = min(target_speed, 20.0)
        elif parsed.parsed_intent == "RESUME":
            target_speed = min(
                target_speed, self.default_speed_kmh
            )

        recommended = str(risk.get("recommended_action") or "")
        if recommended == "emergency_brake":
            action = "emergency_brake"
            target_speed = 0.0
        elif recommended == "decelerate" and action in {
            "accelerate",
            "keep_lane",
            "lane_change_left",
            "lane_change_right",
            "turn_left",
            "turn_right",
        }:
            action = "decelerate"
            target_speed = min(target_speed, 15.0)
        if action in {"lane_change_left", "lane_change_right"}:
            direction = action.removeprefix("lane_change_")
            if (
                risk.get("lane_change", {})
                .get(direction, {})
                .get("is_safe")
                is not True
            ):
                action = "decelerate"
                target_speed = min(target_speed, 15.0)
        source_action = {
            "keep_lane": "KEEP_LANE",
            "accelerate": "ADJUST_SPEED",
            "decelerate": "ADJUST_SPEED",
            "stop": "STOP",
            "emergency_brake": "EMERGENCY_BRAKE",
            "lane_change_left": "CHANGE_LANE",
            "lane_change_right": "CHANGE_LANE",
            "turn_left": "TURN",
            "turn_right": "TURN",
        }[action]
        return {
            "schema_version": "1.0.0",
            "request_id": request_id,
            "frame_id": frame_id,
            "decision_status": "READY",
            "action": action,
            "target_speed_kmh": round(target_speed, 6),
            "target_lane": (
                action.removeprefix("lane_change_")
                if action.startswith("lane_change_")
                else None
            ),
            "target_location": None,
            "emergency": action == "emergency_brake",
            "reason": "generic_text_envelope",
            "parse_status": "VALID",
            "parse_confidence": round(float(parsed.confidence), 6),
            "source_step_id": "step_1",
            "source_step_action": source_action,
            "source_step_count": 1,
            "matched_entity_id": None,
            "risk_level": str(risk.get("risk_level", "low")),
            "risk_reason_codes": list(risk.get("reason_codes", [])),
            "blocked_reason_codes": [],
        }
