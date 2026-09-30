"""Generic text-instruction finite state machine.

The FSM converts a scheduled natural-language driving instruction into a
small, scene-agnostic semantic intent.  It deliberately never branches on
command ids, event ids or scene ids: the same rules apply to every scene and
every scheduled text command.
"""

from __future__ import annotations

import re
import math
import copy
import hashlib
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import torch

from scene_understanding.src.speed_target import (
    DEFAULT_RELATIVE_SPEED_DELTA_KMH,
    resolve_step_speed_target,
)


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
    r"(\d+(?:\.\d+)?)\s*(?:km/?h|kmh|kph|kilomet(?:er|re)s?\s+per\s+hour|公里每小时|公里/小时|迈)",
    re.IGNORECASE,
)


@dataclass
class ParsedInstruction:
    parsed_intent: str = "KEEP_LANE"
    requested_lane_direction: str | None = None
    target_speed_kmh: float | None = None
    confidence: float = 1.0
    source_text: str = ""
    semantic_goal: tuple[str, ...] = ()
    speed_change: str | None = None
    speed_delta_kmh: float | None = None
    speed_reference_kmh: float | None = None
    parse_status: str = "VALID"
    parse_source: str | None = None


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
        *,
        cache_capacity: int = 128,
    ) -> None:
        self.default_speed_kmh = float(default_speed_kmh)
        self.parser = parser
        if cache_capacity < 1:
            raise ValueError("cache_capacity must be positive")
        self.cache_capacity = int(cache_capacity)
        self._token_cache: OrderedDict = OrderedDict()
        self._parse_cache: OrderedDict = OrderedDict()
        self._speed_target_cache: OrderedDict = OrderedDict()

    def clear_caches(self) -> None:
        """Invalidate language caches after replacing parser weights/config."""
        self._token_cache.clear()
        self._parse_cache.clear()
        self._speed_target_cache.clear()

    def _cache_put(self, cache: OrderedDict, key: Any, value: Any) -> None:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > self.cache_capacity:
            cache.popitem(last=False)

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
        if not isinstance(command.get('driving_intent'), Mapping) and self._requires_plan(source_text):
            result = self._parser_result(source_text, command) if use_parser_model and self.parser is not None else {}
            intent = result.get('intent')
            steps = intent.get('steps') or [] if isinstance(intent, Mapping) else []
            separators = re.findall(r'然后|随后|再次|接着|\bthen\b', source_text, re.IGNORECASE)
            valid = result.get('status') == 'VALID' and self._ordered_steps(steps)
            valid = valid and len(steps) >= max(2, len(separators) + 1)
            if not valid:
                # Never leave a partial cached plan available to the executor.
                self._parse_cache.pop(self._parse_key(command, source_text), None)
                return ParsedInstruction(parsed_intent='STOP', target_speed_kmh=0.,
                    source_text=source_text, parse_status='NEEDS_CLARIFICATION',
                    parse_source='compound_plan_unresolved', confidence=0.)
            parsed = self._merge_parser_result(ParsedInstruction(), result)
            parsed.source_text = source_text
            parsed.semantic_goal = tuple(self._step_goal(step) for step in steps)
            return parsed
        goal_intent, goal_direction, goal_speed = _intent_from_goals(
            semantic_goal
        )
        if goal_intent in {"CHANGE_LANE_LEFT", "CHANGE_LANE_RIGHT"}:
            # Explicit semantic goals take priority over keyword guessing:
            # the text may be phrased colloquially (e.g. "并道至左侧车道")
            # without the generic lane-change keywords.
            return ParsedInstruction(
                parsed_intent=goal_intent,
                requested_lane_direction=goal_direction,
                target_speed_kmh=(
                    goal_speed
                    if goal_speed is not None
                    else self._extract_speed(source_text)
                ),
                source_text=source_text,
                semantic_goal=semantic_goal,
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
            and (command.get("id") is not None or (
                not re.search(r"[\u4e00-\u9fff]",source_text) and not self._neutral_cruise(source_text)))
        ):
            # IDs are transport metadata, not a requirement for user commands.
            # Keep the known neutral cruise fallback deterministic.
            model_parsed = self._parser_result(source_text, command)
            parsed = self._merge_parser_result(parsed, model_parsed)
        parsed.source_text = source_text
        parsed.semantic_goal = semantic_goal
        return parsed

    @staticmethod
    def _parse_key(command, text):
        return (str(command.get('id') or ''), text, str(command.get('parser_text_en') or ''))

    @staticmethod
    def _requires_plan(text: str) -> bool:
        return bool(re.search(
            r'然后|随后|再次|接着|再(?:向|左转|右转|直行|停车)|先.+(?:后|再)|'
            r'\b(?:then|after that|and then)\b|'
            r'(?:slow down|accelerate|turn|overtake|change).+\b(?:and|before|after)\b',
            text, re.IGNORECASE))

    @staticmethod
    def _step_goal(step):
        action = str(step.get('action', '')).upper()
        parameters = step.get('parameters') or {}
        if action in {'TURN', 'CHANGE_LANE'}:
            return action + '_' + str(parameters.get('direction', '')).upper()
        if action == 'PROCEED' and (parameters.get('condition') == 'STRAIGHT_THROUGH_JUNCTION'
                                    or parameters.get('direction') == 'STRAIGHT'):
            return 'PROCEED_STRAIGHT'
        return action

    @staticmethod
    def _ordered_steps(steps) -> bool:
        if not isinstance(steps, list) or len(steps) < 2:
            return False
        seen = set()
        previous = None
        for step in steps:
            if not isinstance(step, Mapping):
                return False
            identity = step.get('step_id')
            dependencies = step.get('depends_on', [])
            trigger = step.get('trigger') or {}
            parameters = step.get('parameters') or {}
            if not isinstance(identity, str) or not identity or identity in seen:
                return False
            if not isinstance(dependencies, list) or not all(isinstance(d, str) and d in seen for d in dependencies):
                return False
            if not isinstance(parameters, Mapping) or not isinstance(trigger, Mapping):
                return False
            if previous is not None and previous not in dependencies:
                return False
            if trigger.get('type') == 'AFTER_STEP' and trigger.get('step_id') not in dependencies:
                return False
            if not isinstance(step.get('action'), str):
                return False
            if step.get('action') in {'TURN', 'CHANGE_LANE'} and parameters.get('direction') not in {'LEFT', 'RIGHT'}:
                return False
            if not step.get('action'):
                return False
            seen.add(identity)
            previous = identity
        return True

    @staticmethod
    def enforce_parse_status(decision, parsed):
        if parsed.parse_status == 'VALID':
            return decision, False
        return dict(action='stop', target_speed_kmh=0.,
                    reason='instruction_requires_clarification',
                    parse_status=parsed.parse_status), True

    @staticmethod
    def _merge_structured_command(
        parsed: ParsedInstruction,
        structured: Mapping[str, Any],
    ) -> ParsedInstruction:
        action = str(structured.get("action", "")).upper()
        mapping = {
            "SET_SPEED": "SET_SPEED",
            "ADJUST_SPEED": "SET_SPEED",
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
        change = str(structured.get("change", "")).strip().upper()
        if action == "ADJUST_SPEED":
            intent = {
                "INCREASE": "SET_SPEED",
                "DECREASE": "DECELERATE",
                "HOLD": "KEEP_LANE",
            }.get(change, intent)
        direction = str(structured.get("direction", "")).upper()
        if intent == "CHANGE_LANE_LEFT":
            if direction not in {"LEFT", "RIGHT"}:
                return parsed
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
            if speed is not None and (not math.isfinite(speed) or speed < 0.0):
                return parsed
        except (TypeError, ValueError):
            speed = parsed.target_speed_kmh
        delta_kmh = structured.get("speed_delta_kmh")
        if delta_kmh is None and structured.get("speed_delta_mps") is not None:
            try:
                delta_kmh = float(structured["speed_delta_mps"]) * 3.6
            except (TypeError, ValueError):
                return parsed
        if delta_kmh is not None:
            try:
                delta_kmh = float(delta_kmh)
            except (TypeError, ValueError):
                return parsed
            if not math.isfinite(delta_kmh) or delta_kmh <= 0.0:
                return parsed
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
            speed_change=change if action == "ADJUST_SPEED" else None,
            speed_delta_kmh=delta_kmh,
        )

    def _parse_text_rules(
        self,
        text: str,
        *,
        semantic_goal: Sequence[str] = (),
    ) -> ParsedInstruction:
        lowered = text.lower()
        speed = self._extract_speed(text)

        if re.fullmatch(r"\s*(?:please\s+)?(?:emergency\s+brake|brake\s+in\s+an\s+emergency)(?:\s+now)?[.!]?\s*", lowered):
            return ParsedInstruction(parsed_intent="EMERGENCY_BRAKE",target_speed_kmh=0.)
        if re.fullmatch(r"\s*(?:please\s+)?(?:stop(?:\s+(?:the\s+)?(?:vehicle|car))?|come\s+to\s+a\s+stop)(?:\s+(?:now|safely))?[.!]?\s*", lowered):
            return ParsedInstruction(parsed_intent="STOP",target_speed_kmh=0.)

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
                parsed_intent="DECELERATE",
                target_speed_kmh=speed,
                speed_change="DECREASE",
            )
        if "提速" in lowered or "加速" in lowered or "恢复车速" in lowered:
            return ParsedInstruction(
                parsed_intent="SET_SPEED",
                target_speed_kmh=speed,
                speed_change="INCREASE",
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
    def _neutral_cruise(text: str) -> bool:
        return bool(re.fullmatch(
            r"\s*(?:keep|continue(?:\s+driving)?)(?:\s+safely)?\s+(?:(?:in\s+)?the\s+)?(?:current\s+)?lane"
            r"(?:\s+at\s+\d+(?:\.\d+)?\s*(?:km/?h|kph|kilomet(?:er|re)s?\s+per\s+hour))?[.!]?\s*",
            text,re.IGNORECASE))

    def driving_intent(self,command):
        """Retain a supplied plan or a multi-step result already parsed this frame."""
        if isinstance(command.get('driving_intent'),Mapping):
            return copy.deepcopy(command['driving_intent'])
        text=str(command.get('text') or command.get('source_text') or command.get('voice_text')
                 or command.get('normalized_text') or command.get('parser_text_en') or '')
        cached=self._parse_cache.get(self._parse_key(command, text)) or {}
        intent=cached.get('intent') or {}
        if cached.get('status') != 'VALID' or len(intent.get('steps') or [])<2:return None
        result=copy.deepcopy(cached);result.pop('intent',None)
        normalized_text = result.pop('_normalized_text', text)
        result.setdefault('source','structured_command_parser')
        result['source_kind']='TEXT_MODEL_PARSE'
        result['model_prediction']=True
        return dict(schema_version='1.2.0',request_id=str(command.get('id') or 'plan-'+hashlib.sha256(text.encode()).hexdigest()[:16]),
            input=dict(modality='TEXT',language='zh-CN' if re.search(r'[\u4e00-\u9fff]',text) else 'en-US',raw_text=text,normalized_text=normalized_text),
            intent=copy.deepcopy(intent),parse_result=result)

    def parsed_step(self,step,source_text):
        parsed=self._merge_parser_result(ParsedInstruction(),dict(status='VALID',confidence=1.,intent=dict(steps=[step])))
        parsed.source_text=source_text
        return parsed

    def resolve_relative_speed(
        self,
        command: Mapping[str, Any],
        parsed: ParsedInstruction,
        *,
        ego_speed_kmh: float,
    ) -> ParsedInstruction:
        """Latch one absolute target for a qualitative speed instruction."""

        if parsed.target_speed_kmh is not None:
            return parsed
        change = parsed.speed_change
        if change is None:
            change = {
                "SET_SPEED": "INCREASE",
                "DECELERATE": "DECREASE",
            }.get(parsed.parsed_intent)
        if change not in {"INCREASE", "DECREASE", "HOLD"}:
            return parsed
        speed = float(ego_speed_kmh)
        if not math.isfinite(speed) or speed < 0.0:
            raise ValueError("ego_speed_kmh must be finite and non-negative")
        delta = (
            float(parsed.speed_delta_kmh)
            if parsed.speed_delta_kmh is not None
            else DEFAULT_RELATIVE_SPEED_DELTA_KMH
        )
        if not math.isfinite(delta) or delta <= 0.0:
            raise ValueError("relative speed delta must be finite and positive")
        identity = str(
            command.get("id")
            or command.get("request_id")
            or parsed.source_text
            or "anonymous_instruction"
        )
        key = (identity, parsed.parsed_intent, change, round(delta, 6))
        cached = self._speed_target_cache.get(key)
        if cached is None:
            _, target = resolve_step_speed_target(
                {
                    "action": "ADJUST_SPEED",
                    "parameters": {
                        "change": change,
                        "speed_delta_kmh": delta,
                    },
                },
                speed,
            )
            cached = (round(speed, 6), round(target, 6))
            self._cache_put(self._speed_target_cache, key, cached)
        else:
            self._speed_target_cache.move_to_end(key)
        return replace(
            parsed,
            target_speed_kmh=cached[1],
            speed_change=change,
            speed_reference_kmh=cached[0],
        )

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
        key = self._parse_key(command, source_text)
        if key in self._parse_cache:
            self._parse_cache.move_to_end(key)
            return self._parse_cache[key]
        try:
            model_text = command.get('parser_text_en') or source_text
            result = self.parser.parse_text(
                model_text,
                request_id=f"fsm-{command.get('id') or source_text}",
                modality="TEXT",
                source_text=source_text,
                source_language="zh-CN" if re.search(r"[\u4e00-\u9fff]",source_text) else "en-US",
            )
        except Exception:
            # Transient inference failure must not poison subsequent retries.
            return {}
        if not isinstance(result, Mapping) or not isinstance(result.get('parse_result'), Mapping):
            return {}
        parse_result = dict(result.get("parse_result") or {})
        parse_result['_normalized_text'] = model_text
        # DrivingIntent keeps intent beside parse_result, not inside it.
        if isinstance(result.get("intent"),Mapping):
            parse_result["intent"] = result["intent"]
        self._cache_put(self._parse_cache, key, parse_result)
        return parse_result

    @staticmethod
    def _merge_parser_result(
        parsed: ParsedInstruction,
        parse_result: Mapping[str, Any],
    ) -> ParsedInstruction:
        status = str(parse_result.get("status", "") or "").strip().upper()
        source = parse_result.get("source")
        source_name = str(source) if source is not None else None
        if status != "VALID":
            return replace(
                parsed,
                parse_status=status or "INVALID",
                parse_source=source_name,
            )
        intent = parsed.parsed_intent
        steps = (parse_result.get("intent") or {}).get("steps") or []
        if steps:
            step=steps[0]
            action = str(step.get("action", "")).upper()
            parameters=step.get("parameters") or {}
            mapping = {
                "KEEP_LANE": "KEEP_LANE",
                "SET_SPEED": "SET_SPEED",
                "ADJUST_SPEED": "SET_SPEED",
                "CHANGE_LANE": "CHANGE_LANE_LEFT",
                "STOP": "STOP",
                "EMERGENCY_BRAKE": "EMERGENCY_BRAKE",
                "PARK": "YIELD",
                "TURN": "TURN_LEFT",
            }
            merged = mapping.get(action)
            change = str(parameters.get("change", "")).strip().upper()
            if action == "ADJUST_SPEED":
                merged = {
                    "INCREASE": "SET_SPEED",
                    "DECREASE": "DECELERATE",
                    "HOLD": "KEEP_LANE",
                }.get(change, merged)
            direction=str(parameters.get("direction","")).upper()
            if action in {"CHANGE_LANE","TURN"}:
                if direction not in {"LEFT","RIGHT"}:
                    return parsed
                merged=("CHANGE_LANE_" if action=="CHANGE_LANE" else "TURN_")+direction
            target_speed=parsed.target_speed_kmh
            delta_kmh=None
            try:
                if parameters.get("target_speed_mps") is not None:
                    target_speed=float(parameters["target_speed_mps"])*3.6
                elif parameters.get("target_speed_kmh") is not None:
                    target_speed=float(parameters["target_speed_kmh"])
                if target_speed is not None and (not math.isfinite(target_speed) or target_speed<0):
                    return parsed
                if parameters.get("speed_delta_mps") is not None:
                    delta_kmh=float(parameters["speed_delta_mps"])*3.6
                elif parameters.get("speed_delta_kmh") is not None:
                    delta_kmh=float(parameters["speed_delta_kmh"])
                if delta_kmh is not None and (
                    not math.isfinite(delta_kmh) or delta_kmh <= 0.0
                ):
                    return parsed
            except (TypeError,ValueError):
                return parsed
            if merged in {"STOP","EMERGENCY_BRAKE"}:target_speed=0.
            if merged is not None and intent == "KEEP_LANE":
                parsed = ParsedInstruction(
                    parsed_intent=merged,
                    requested_lane_direction=(
                        direction.lower() if action=="CHANGE_LANE" else None
                    ),
                    target_speed_kmh=target_speed,
                    confidence=float(parse_result.get("confidence", 0.0) or 0.0),
                    speed_change=change if action == "ADJUST_SPEED" else None,
                    speed_delta_kmh=delta_kmh,
                )
        return replace(
            parsed,
            parse_status="VALID",
            parse_source=source_name,
        )

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
        key = (cache_key, text)
        cached = self._token_cache.get(key)
        if cached is not None:
            self._token_cache.move_to_end(key)
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
        self._cache_put(self._token_cache, key, result)
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
        }[INTENT_TO_ACTION[parsed.parsed_intent]]
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
