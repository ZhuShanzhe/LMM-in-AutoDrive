from __future__ import annotations

import re
from time import perf_counter
from typing import Any

from .factory import make_document, make_step
from .intent_boundaries import classify_chinese_braking
from .normalizer import normalize_text
from .speed_slots import (
    TARGET_SPEED_PATTERN,
    canonical_speed_unit,
    speed_to_mps,
)


_TARGET_SPEED = TARGET_SPEED_PATTERN
_RELATIVE_SPEED = re.compile(
    r"(?P<verb>提速|加速|减速|降速)\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>km/h|m/s)",
    re.IGNORECASE,
)
_TURN = re.compile(
    r"(?:(?:前方\s*)?(?P<distance>\d+(?:\.\d+)?)\s*m(?:后)?\s*)?"
    r"(?:(?:下个|前方|前面(?:的)?)?(?:十字)?(?:路口|交叉口)\s*)?"
    r"(?:继续)?(?:向)?(?P<direction>左转|右转|左拐|右拐|直行|直走)"
)
_COMPLEX_HINTS = re.compile(
    r"行人|骑行者|慢车|低速车|公交|救护车|来车|锥桶|施工|障碍|红灯|"
    r"绕开|绕过|避让|避开|躲开|礼让|让行|超越|超车|超过|加塞|"
    r"确认安全|然后|随后|之后|以后|后再|并回|并入|回到|靠边|路边"
)


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

        if re.search(r"撞击|撞向|撞开|碰撞|撞上|撞坏|冲撞|撞车|碾压", text):
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

        if re.search(
            r"不管限速|无视限速|违反限速|开到最快|能多快开多快|超速|"
            r"无视红灯|闯红灯|逆行|人行道上(?:超过|超越|超车)",
            text,
        ):
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
                warnings=["该指令要求违反限速、信号灯、车道或其他交通规则。"],
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

        if re.search(
            r"取消|撤销(?:刚才|上一条)?(?:指令|命令)|别执行(?:刚才|上一条)(?:那条)?(?:指令|命令)",
            text,
        ):
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

        braking_boundary = classify_chinese_braking(text)
        has_other_action = bool(
            re.search(
                r"变道|并线|转弯|左转|右转|避开|避让|绕开|让行|靠边|"
                r"超车|超越|继续|恢复|跟随|进入|驶出",
                text,
            )
        )
        if braking_boundary and not has_other_action and braking_boundary.urgency != "NORMAL":
            parameters = (
                {"change": "DECREASE"}
                if braking_boundary.action == "ADJUST_SPEED"
                else None
            )
            completion = (
                {"type": "VEHICLE_STOPPED"}
                if braking_boundary.action in {"STOP", "EMERGENCY_BRAKE"}
                else None
            )
            return self._document(
                raw_text,
                text,
                modality,
                request_id,
                "EMERGENCY_RESPONSE"
                if braking_boundary.action == "EMERGENCY_BRAKE"
                else "BASIC_CONTROL",
                braking_boundary.urgency,
                [
                    make_step(
                        "step_1",
                        braking_boundary.action,
                        parameters=parameters,
                        completion=completion,
                    )
                ],
                0.99,
                started,
            )

        complex_result = self._parse_complex_command(
            raw_text,
            text,
            modality,
            request_id,
            started,
        )
        if complex_result is not None:
            return complex_result

        if _COMPLEX_HINTS.search(text):
            return None

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
                        "target_speed_mps": round(speed_to_mps(value, unit), 3),
                        "source_value": value,
                        "source_unit": canonical_speed_unit(unit),
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
                        "speed_delta_mps": round(speed_to_mps(value, unit), 3),
                        "source_value": value,
                        "source_unit": canonical_speed_unit(unit),
                    },
                    preconditions=["PATH_CLEAR"] if change == "INCREASE" else [],
                    on_blocked="WAIT_FOR_SAFE" if change == "INCREASE" else "SAFE_STOP",
                ),
            )

        patterns = [
            (r"保持(?:当前|本)?车道", "KEEP_LANE", {}, [], "SAFE_STOP"),
            (r"(?:向?左(?:侧)?变道|并到左边车道|往左侧车道并线|驶入左侧相邻车道|向左横移一个车道)", "CHANGE_LANE", {"direction": "LEFT", "lane_count": 1}, ["LEFT_LANE_EXISTS", "LEFT_LANE_SAFE", "LANE_CHANGE_LEGAL"], "WAIT_FOR_SAFE"),
            (r"(?:向?右(?:侧)?变道|并到右边车道|往右侧车道并线|驶入右侧相邻车道|向右横移一个车道)", "CHANGE_LANE", {"direction": "RIGHT", "lane_count": 1}, ["RIGHT_LANE_EXISTS", "RIGHT_LANE_SAFE", "LANE_CHANGE_LEGAL"], "WAIT_FOR_SAFE"),
            (r"(?:正常)?停车(?!场)|停下|停住", "STOP", {}, [], "SAFE_STOP"),
            (r"恢复(?:正常|之前的)?(?:行驶状态|行驶)|继续(?:按正常状态)?(?:行驶|前进)", "RESUME", {}, ["PATH_CLEAR"], "WAIT_FOR_SAFE"),
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
            direction_text = match.group("direction")
            direction = (
                "LEFT"
                if direction_text in {"左转", "左拐"}
                else "RIGHT"
                if direction_text in {"右转", "右拐"}
                else "STRAIGHT"
            )
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
                r"(?P<verb>加速|提速|提高(?:车辆)?(?:车)?速|把速度提上去|稍微加点速|"
                r"加快(?:一点|(?:你的)?驾驶)?|减速|降速|降低车速|把车速降下来|"
                r"慢一点|慢一些|稳一点|轻踩刹车(?:减速)?|踩刹车|缓行)",
                text,
            )
            if qualitative:
                verb = qualitative.group("verb")
                change = (
                    "INCREASE"
                    if re.search(r"加速|提速|提高|加快|提上去|加点速", verb)
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

    def _parse_complex_command(
        self,
        raw_text: str,
        text: str,
        modality: str,
        request_id: str | None,
        started: float,
    ) -> dict[str, Any] | None:
        if not _COMPLEX_HINTS.search(text) and not re.search(
            r"路况危险|安全车速|靠边|并道|并线|突然出现", text
        ):
            return None

        candidates: list[tuple[int, dict[str, Any]]] = []

        def add(position: int, action: str, **values: Any) -> None:
            command = {"action": action, **values}
            signature = (
                action,
                command.get("direction"),
                command.get("target_type"),
            )
            if any(
                (
                    existing["action"],
                    existing.get("direction"),
                    existing.get("target_type"),
                )
                == signature
                for _, existing in candidates
            ):
                return
            candidates.append((position, command))

        target_type = "OBSTACLE"
        target_relation = "AHEAD"
        if "行人" in text:
            target_type = "PEDESTRIAN"
            if "横穿" in text:
                target_relation = "AHEAD_CROSSING"
        elif "锥桶" in text:
            target_type = "TRAFFIC_CONE"
        elif "施工" in text:
            target_type = "CONSTRUCTION_ZONE"
        elif "慢车" in text:
            target_type = "SLOW_VEHICLE"
        elif re.search(r"骑行者|自行车", text):
            target_type = "CYCLIST"
        elif re.search(r"车辆|加塞|前车|低速车|公交|救护车|来车", text):
            target_type = "VEHICLE"

        if "左侧" in text:
            target_relation = "LEFT"
        elif "右侧" in text:
            target_relation = "RIGHT"

        pull_over = re.search(r"靠边(?:停车)?|停到路边", text)
        if pull_over:
            add(pull_over.start(), "PULL_OVER")

        for speed in _TARGET_SPEED.finditer(text):
            value = float(speed.group("value"))
            unit = speed.group("unit")
            add(
                speed.start(),
                "SET_SPEED",
                target_speed_mps=round(speed_to_mps(value, unit), 3),
                source_value=value,
                source_unit=canonical_speed_unit(unit),
            )

        if not any(command["action"] == "SET_SPEED" for _, command in candidates):
            speed_change = re.search(
                r"减速|降速|降低(?:车速|速度)|慢一点|慢一些|缓行|保持安全车速|提速|加速",
                text,
            )
            if speed_change:
                decrease = bool(
                    re.search(
                        r"减速|降速|降低(?:车速|速度)|慢一点|慢一些|缓行|安全车速",
                        speed_change.group(),
                    )
                )
                values: dict[str, Any] = {
                    "change": "DECREASE" if decrease else "INCREASE"
                }
                if decrease and "避让" in text and "行人" in text:
                    values.update(
                        purpose="YIELD",
                        target_type="PEDESTRIAN",
                        target_relation=target_relation,
                    )
                add(speed_change.start(), "ADJUST_SPEED", **values)

        yield_match = re.search(r"礼让|让行|(?<!避)让(?:前方)?|给.+?让行", text)
        if yield_match and not re.search(r"减速.{0,4}避让|避让.{0,4}减速", text):
            add(
                yield_match.start(),
                "YIELD",
                target_type=target_type,
                target_relation=target_relation,
            )

        avoid = re.search(r"绕开|绕过|避让|避开|躲开", text)
        if avoid:
            yielding_with_speed = bool(
                re.search(r"减速.{0,4}避让|避让.{0,4}减速", text)
                and target_type == "PEDESTRIAN"
            )
            if target_type == "SLOW_VEHICLE":
                add(
                    avoid.start(),
                    "OVERTAKE",
                    target_type="SLOW_VEHICLE",
                    target_relation="AHEAD",
                )
            elif not yielding_with_speed:
                add(
                    avoid.start(),
                    "AVOID",
                    target_type=target_type,
                    target_relation=target_relation,
                )

        overtake = re.search(r"超越|超车|超过", text)
        if overtake:
            add(
                overtake.start(),
                "OVERTAKE",
                target_type="SLOW_VEHICLE"
                if re.search(r"慢车|低速车", text)
                else "VEHICLE",
                target_relation="AHEAD",
            )

        lane_change = re.search(
            r"(?:向|至|往)?(?P<direction>左|右)(?:边|侧)?(?:相邻)?(?:车道)?"
            r"(?:变道|并道|并线|横移)|"
            r"(?:变道|并道|并线|并入|驶入|并到)(?:至|到|向)?"
            r"(?P<direction_after>左|右)(?:边|侧)?(?:相邻)?车道",
            text,
        )
        if lane_change:
            direction_text = lane_change.group("direction") or lane_change.group(
                "direction_after"
            )
            direction = "LEFT" if direction_text == "左" else "RIGHT"
            values = {"direction": direction, "lane_count": 1}
            if re.search(r"慢车|低速车|前车", text) and re.search(
                r"超越|超车|超过", text
            ):
                values.update(
                    purpose="OVERTAKE",
                    target_type="SLOW_VEHICLE",
                    target_relation="AHEAD",
                )
            add(lane_change.start(), "CHANGE_LANE", **values)

        keep_lane = re.search(r"保持(?:当前|本)?车道", text)
        if keep_lane:
            add(keep_lane.start(), "KEEP_LANE")

        stop = re.search(r"停车(?!场)|停下|停住", text)
        if stop and not pull_over:
            add(stop.start(), "STOP")

        braking_boundary = classify_chinese_braking(text)
        if braking_boundary:
            candidate_actions = {"STOP", "EMERGENCY_BRAKE"}
            if braking_boundary.action == "ADJUST_SPEED":
                candidate_actions.add("ADJUST_SPEED")
            matched = False
            for _, command in candidates:
                if command.get("action") not in candidate_actions:
                    continue
                matched = True
                command["action"] = braking_boundary.action
                if braking_boundary.action == "ADJUST_SPEED":
                    command["change"] = "DECREASE"
                else:
                    command.pop("change", None)
            if not matched and braking_boundary.urgency != "NORMAL":
                marker = re.search(r"刹车|制动|刹停|停车(?!场)|停下|停住", text)
                if marker:
                    values = (
                        {"change": "DECREASE"}
                        if braking_boundary.action == "ADJUST_SPEED"
                        else {}
                    )
                    add(marker.start(), braking_boundary.action, **values)

        resume = re.search(
            r"回归原车道|回到原车道|恢复(?:正常|之前的)?(?:行驶状态|行驶)|"
            r"继续(?:按正常状态)?(?:行驶|前进)",
            text,
        )
        if resume:
            add(resume.start(), "RESUME")

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])
        commands = [command for _, command in candidates]
        steps = self._expand_fast_commands(commands)
        is_emergency = bool(
            braking_boundary
            and braking_boundary.action == "EMERGENCY_BRAKE"
        ) or bool(re.search(r"紧急|突发|突然|危险|加塞|施工路段", text))
        urgency = (
            braking_boundary.urgency
            if braking_boundary
            else ("EMERGENCY" if is_emergency else "NORMAL")
        )
        return self._document(
            raw_text,
            text,
            modality,
            request_id,
            "EMERGENCY_RESPONSE"
            if is_emergency
            else "COMPLEX_OBSTACLE_AVOIDANCE",
            urgency,
            steps,
            0.97,
            started,
            driving_style="CONSERVATIVE",
        )

    @staticmethod
    def _expand_fast_commands(commands: list[dict[str, Any]]) -> list[dict[str, Any]]:
        steps: list[dict[str, Any]] = []
        for index, command in enumerate(commands, start=1):
            action = command["action"]
            step_id = f"step_{index}"
            previous_id = f"step_{index - 1}" if index > 1 else None
            parameters = {
                key: command[key]
                for key in (
                    "direction",
                    "change",
                    "target_speed_mps",
                    "source_value",
                    "source_unit",
                    "lane_count",
                )
                if key in command
            }
            target = None
            if "target_type" in command:
                target = {
                    "type": command["target_type"],
                    "relation": command.get("target_relation", "UNSPECIFIED"),
                }

            trigger: dict[str, Any] = {"type": "IMMEDIATE"}
            if previous_id:
                trigger = {"type": "AFTER_STEP", "step_id": previous_id}
            elif target is not None and action in {"AVOID", "OVERTAKE", "YIELD"}:
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
            elif action == "ADJUST_SPEED" and command.get("purpose") == "YIELD":
                preconditions = ["TARGET_VISIBLE"]
                completion = {"type": "TARGET_CLEARED"}
            elif action == "CHANGE_LANE":
                side = command["direction"]
                preconditions = [
                    f"{side}_LANE_EXISTS",
                    f"{side}_LANE_SAFE",
                    "LANE_CHANGE_LEGAL",
                ]
                on_blocked = "WAIT_FOR_SAFE"
                completion = {"type": "LANE_CHANGE_COMPLETED"}
            elif action in {"AVOID", "OVERTAKE", "YIELD"}:
                preconditions = ["TARGET_VISIBLE", "PATH_CLEAR"]
                completion = {"type": "TARGET_CLEARED"}
            elif action in {"STOP", "EMERGENCY_BRAKE"}:
                completion = {"type": "VEHICLE_STOPPED"}
            elif action == "RESUME":
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
                    completion=completion,
                )
            )
        return steps

    @staticmethod
    def _ambiguity(text: str) -> tuple[str, list[str], str] | None:
        if re.search(
            r"换个道|变个道|往旁边并线|换到另一条车道|并过去", text
        ) and not re.search(r"左|右", text):
            return (
                "BASIC_CONTROL",
                ["intent.steps[0].parameters.direction"],
                "请说明希望向左还是向右变道。",
            )
        if re.search(r"转弯|拐弯|路口转|转一下", text) and not re.search(
            r"左|右|直行|直走", text
        ):
            return (
                "NAVIGATION",
                ["intent.steps[0].parameters.direction"],
                "请说明希望左转、右转还是直行。",
            )
        if re.search(r"绕过去|绕过那个东西", text) and not re.search(
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
        if re.search(r"停到那边", text):
            return (
                "NAVIGATION",
                ["intent.steps[0].target"],
                "请说明需要停车的具体位置。",
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
