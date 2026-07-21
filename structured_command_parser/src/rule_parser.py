from __future__ import annotations

import re
from time import perf_counter
from typing import Any

from .factory import make_document, make_step
from .normalizer import normalize_text


_TARGET_SPEED = re.compile(
    r"(?:(?:提速|加速|减速|降速|(?:车速|速度)(?:控制|调整)?|保持车速|保持速度)"
    r"\s*(?:至|到|为|在)\s*|以\s*|(?:尝试)?行驶至\s*)"
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>km/h|m/s)",
    re.IGNORECASE,
)
_RELATIVE_SPEED = re.compile(
    r"(?P<verb>提速|加速|减速|降速)\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>km/h|m/s)",
    re.IGNORECASE,
)
_TURN = re.compile(
    r"(?:(?:前方\s*)?(?P<distance>\d+(?:\.\d+)?)\s*m\s*)?"
    r"(?:的?路口|交叉口)?\s*(?P<direction>左转|右转|直行)"
)
_COMPLEX_HINTS = re.compile(
    r"行人|慢车|公交站|锥桶|施工|绕开|避让|超越|超车|加塞|确认安全|然后|随后|之后|后再|并回"
)


def _speed_to_mps(value: float, unit: str) -> float:
    return value / 3.6 if unit.lower() == "km/h" else value


class RuleIntentParser:
    def parse(
        self,
        raw_text: str,
        *,
        modality: str = "TEXT",
        request_id: str | None = None,
    ) -> dict[str, Any] | None:
        started = perf_counter()
        text = normalize_text(raw_text)

        if re.search(r"撞击|撞向|碰撞|撞上|撞坏|冲撞|撞车|碾压", text):
            return make_document(
                raw_text=raw_text,
                normalized_text=text,
                modality=modality,
                category="BASIC_CONTROL",
                urgency="NORMAL",
                steps=[],
                status="UNSUPPORTED",
                method="RULE",
                model=None,
                confidence=0.99,
                latency_ms=(perf_counter() - started) * 1000,
                request_id=request_id,
                warnings=["该指令要求主动碰撞道路参与者或物体。"],
            )

        if re.search(r"不管限速|无视限速|违反限速|开到最快|能多快开多快", text):
            return make_document(
                raw_text=raw_text,
                normalized_text=text,
                modality=modality,
                category="BASIC_CONTROL",
                urgency="NORMAL",
                steps=[],
                status="UNSUPPORTED",
                method="RULE",
                model=None,
                confidence=0.99,
                latency_ms=(perf_counter() - started) * 1000,
                request_id=request_id,
                warnings=["该指令要求违反限速或交通规则。"],
            )

        ambiguity = self._ambiguity(text)
        if ambiguity is not None:
            category, missing_slots, question = ambiguity
            return make_document(
                raw_text=raw_text,
                normalized_text=text,
                modality=modality,
                category=category,
                urgency="NORMAL",
                steps=[],
                status="NEEDS_CLARIFICATION",
                method="RULE",
                model=None,
                confidence=0.95,
                latency_ms=(perf_counter() - started) * 1000,
                request_id=request_id,
                missing_slots=missing_slots,
                warnings=["文本缺少安全执行所需的关键信息。"],
                clarification_question=question,
            )

        if _COMPLEX_HINTS.search(text):
            return None

        if re.search(r"取消|撤销(?:刚才|上一条)?指令", text):
            return self._document(
                raw_text,
                text,
                modality,
                request_id,
                "META_CONTROL",
                "NORMAL",
                [make_step("step_1", "CANCEL")],
                0.99,
                started,
            )

        if re.search(
            r"紧急(?:制动|刹车|停车)|立即(?:踩|踩下)?(?:制动|刹车|停车)",
            text,
        ):
            return self._document(
                raw_text,
                text,
                modality,
                request_id,
                "EMERGENCY_RESPONSE",
                "EMERGENCY",
                [
                    make_step(
                        "step_1",
                        "EMERGENCY_BRAKE",
                        completion={"type": "VEHICLE_STOPPED"},
                    )
                ],
                0.99,
                started,
            )

        matches: list[tuple[int, dict[str, Any]]] = []
        occupied: list[tuple[int, int]] = []

        def add_match(start: int, end: int, step: dict[str, Any]) -> None:
            matches.append((start, step))
            occupied.append((start, end))

        def overlaps(start: int, end: int) -> bool:
            return any(start < other_end and end > other_start for other_start, other_end in occupied)

        for match in _TARGET_SPEED.finditer(text):
            value = float(match.group("value"))
            unit = match.group("unit")
            add_match(
                match.start(),
                match.end(),
                make_step(
                    "pending",
                    "SET_SPEED",
                    parameters={
                        "target_speed_mps": round(_speed_to_mps(value, unit), 3),
                        "source_value": value,
                        "source_unit": unit,
                    },
                    preconditions=["PATH_CLEAR"],
                    on_blocked="WAIT_FOR_SAFE",
                    completion={"type": "TARGET_SPEED_REACHED"},
                ),
            )

        for match in _RELATIVE_SPEED.finditer(text):
            if overlaps(match.start(), match.end()):
                continue
            value = float(match.group("value"))
            unit = match.group("unit")
            change = "INCREASE" if match.group("verb") in {"提速", "加速"} else "DECREASE"
            add_match(
                match.start(),
                match.end(),
                make_step(
                    "pending",
                    "ADJUST_SPEED",
                    parameters={
                        "change": change,
                        "speed_delta_mps": round(_speed_to_mps(value, unit), 3),
                        "source_value": value,
                        "source_unit": unit,
                    },
                    preconditions=["PATH_CLEAR"] if change == "INCREASE" else [],
                    on_blocked="WAIT_FOR_SAFE" if change == "INCREASE" else "SAFE_STOP",
                ),
            )

        patterns = [
            (r"保持(?:当前|本)?车道", "KEEP_LANE", {}, [], "SAFE_STOP"),
            (r"向?左(?:侧)?变道", "CHANGE_LANE", {"direction": "LEFT", "lane_count": 1}, ["LEFT_LANE_EXISTS", "LEFT_LANE_SAFE", "LANE_CHANGE_LEGAL"], "WAIT_FOR_SAFE"),
            (r"向?右(?:侧)?变道", "CHANGE_LANE", {"direction": "RIGHT", "lane_count": 1}, ["RIGHT_LANE_EXISTS", "RIGHT_LANE_SAFE", "LANE_CHANGE_LEGAL"], "WAIT_FOR_SAFE"),
            (r"(?:正常)?停车(?!场)|停下|停住", "STOP", {}, [], "SAFE_STOP"),
            (r"恢复(?:正常)?行驶|继续行驶", "RESUME", {}, ["PATH_CLEAR"], "WAIT_FOR_SAFE"),
        ]
        for pattern, action, parameters, preconditions, on_blocked in patterns:
            for match in re.finditer(pattern, text):
                if overlaps(match.start(), match.end()):
                    continue
                completion = {"type": "VEHICLE_STOPPED"} if action == "STOP" else None
                add_match(
                    match.start(),
                    match.end(),
                    make_step(
                        "pending",
                        action,
                        parameters=parameters,
                        preconditions=preconditions,
                        on_blocked=on_blocked,
                        completion=completion,
                    ),
                )

        for match in _TURN.finditer(text):
            if overlaps(match.start(), match.end()):
                continue
            direction = {"左转": "LEFT", "右转": "RIGHT", "直行": "STRAIGHT"}[
                match.group("direction")
            ]
            distance = match.group("distance")
            trigger: dict[str, Any] = {"type": "AT_JUNCTION"}
            parameters: dict[str, Any] = {"direction": direction}
            if distance is not None:
                distance_m = float(distance)
                trigger = {"type": "AT_DISTANCE", "distance_m": distance_m}
                parameters["distance_m"] = distance_m
            add_match(
                match.start(),
                match.end(),
                make_step(
                    "pending",
                    "TURN",
                    parameters=parameters,
                    trigger=trigger,
                    preconditions=["JUNCTION_REACHED", "PATH_CLEAR"],
                    on_blocked="WAIT_FOR_SAFE",
                    target={"type": "JUNCTION", "relation": "AHEAD"},
                    completion={"type": "JUNCTION_EXITED"},
                ),
            )

        if not any(step["action"] in {"SET_SPEED", "ADJUST_SPEED"} for _, step in matches):
            qualitative = re.search(
                r"(?P<verb>加速|提速|提高(?:车辆)?(?:车)?速|加快(?:你的)?驾驶|"
                r"减速|降速|慢一点|稳一点|轻踩刹车|踩刹车|缓行)",
                text,
            )
            if qualitative:
                verb = qualitative.group("verb")
                change = (
                    "INCREASE"
                    if re.search(r"加速|提速|提高|加快", verb)
                    else "DECREASE"
                )
                add_match(
                    qualitative.start(),
                    qualitative.end(),
                    make_step(
                        "pending",
                        "ADJUST_SPEED",
                        parameters={"change": change},
                        preconditions=["PATH_CLEAR"] if change == "INCREASE" else [],
                        on_blocked="WAIT_FOR_SAFE" if change == "INCREASE" else "SAFE_STOP",
                    ),
                )

        if not matches:
            return None

        matches.sort(key=lambda item: item[0])
        steps = [step for _, step in matches]
        for index, step in enumerate(steps, start=1):
            step["step_id"] = f"step_{index}"

        actions = {step["action"] for step in steps}
        if actions <= {"CANCEL", "RESUME"}:
            category = "META_CONTROL"
        elif "TURN" in actions:
            category = "NAVIGATION"
        else:
            category = "BASIC_CONTROL"
        style = "CONSERVATIVE" if re.search(r"稳一点|安全车速", text) else "NORMAL"
        return self._document(
            raw_text,
            text,
            modality,
            request_id,
            category,
            "NORMAL",
            steps,
            0.98 if len(steps) == 1 else 0.96,
            started,
            driving_style=style,
        )

    @staticmethod
    def _ambiguity(text: str) -> tuple[str, list[str], str] | None:
        if re.search(r"换个道|变个道", text) and not re.search(r"左|右", text):
            return (
                "BASIC_CONTROL",
                ["intent.steps[0].parameters.direction"],
                "请说明希望向左还是向右变道。",
            )
        if re.search(r"转弯", text) and not re.search(r"左|右|直行", text):
            return (
                "NAVIGATION",
                ["intent.steps[0].parameters.direction"],
                "请说明希望左转、右转还是直行。",
            )
        if re.search(r"绕过去", text) and not re.search(
            r"行人|车辆|慢车|障碍|锥桶|施工|左|右", text
        ):
            return (
                "NAVIGATION",
                ["intent.steps[0].target", "intent.steps[0].parameters.direction"],
                "请说明需要绕开什么目标，以及希望从左侧还是右侧绕行。",
            )
        if re.search(r"停在那辆车旁边|停到那辆车旁边", text):
            return (
                "NAVIGATION",
                ["intent.steps[0].target"],
                "请进一步说明目标车辆的颜色、位置或其他可识别特征。",
            )
        return None

    @staticmethod
    def _document(
        raw_text: str,
        normalized_text: str,
        modality: str,
        request_id: str | None,
        category: str,
        urgency: str,
        steps: list[dict[str, Any]],
        confidence: float,
        started: float,
        *,
        driving_style: str = "NORMAL",
    ) -> dict[str, Any]:
        return make_document(
            raw_text=raw_text,
            normalized_text=normalized_text,
            modality=modality,
            category=category,
            urgency=urgency,
            steps=steps,
            status="VALID",
            method="RULE",
            model=None,
            confidence=confidence,
            latency_ms=(perf_counter() - started) * 1000,
            request_id=request_id,
            driving_style=driving_style,
        )
