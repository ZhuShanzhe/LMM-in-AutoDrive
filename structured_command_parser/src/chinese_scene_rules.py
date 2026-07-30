from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .speed_slots import TARGET_SPEED_PATTERN, canonical_speed_unit, speed_to_mps


_SCENE_HINTS = re.compile(
    r"行人|乘客|车辆|前车|慢车|公交|自行车|骑行者|路口|斑马线|"
    r"锥桶|锥形桶|施工|车道|车距|终点|雨雾|能见度|湿滑|"
    r"变道|并道|并线|超越|超车|超过|绕行|避让|礼让|等待|靠边"
)
_CAP_SPEED = re.compile(
    r"(?:车速|速度)?\s*(?:不得|不能|不要|不应|不高于|不超过|最高(?:为|到)?)\s*"
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>km/h|m/s|km)",
    re.IGNORECASE,
)
_TURN = re.compile(
    r"(?:(?:前方|前面)?(?:十字)?(?:路口|交叉口)\s*)?"
    r"(?:继续)?(?:向)?(?P<direction>左转|右转|左拐|右拐|直行|直走)"
)


@dataclass(frozen=True)
class ChineseSceneParse:
    commands: list[dict[str, Any]]
    entities: list[dict[str, Any]]
    category: str
    urgency: str
    driving_style: str
    max_speed_mps: float | None = None


@dataclass(frozen=True)
class _EntityMention:
    start: int
    end: int
    entity: dict[str, Any]


def _relation(text: str, start: int, end: int, source: str) -> str:
    context = text[max(0, start - 8) : min(len(text), end + 8)]
    if re.search(r"横穿|横向|过街|斑马线", context):
        return "AHEAD_CROSSING"
    if re.search(r"左侧|左边", context):
        return "LEFT"
    if re.search(r"右侧|右边", context):
        return "RIGHT"
    if re.search(r"前方|前车|前面", context) or source in {"路口", "十字路口"}:
        return "AHEAD"
    return "UNSPECIFIED"


def _attributes(source: str, entity_type: str, context: str) -> tuple[dict[str, Any], list[str]]:
    attributes: dict[str, Any] = {}
    descriptors: list[str] = []
    if "白色" in source:
        attributes["color"] = "WHITE"
    elif "红色" in source:
        attributes["color"] = "RED"

    if "公交车" in source:
        attributes["vehicle_subtype"] = "BUS"
        descriptors.append("bus")
        if re.search(r"没有起步|停靠|停止", context):
            attributes["motion_state"] = "STOPPED"
            descriptors.append("stopped_bus")
        elif re.search(r"跟随前方公交车|行驶中的公交", context):
            attributes["motion_state"] = "MOVING"
            descriptors.append("moving_bus")
    if "慢" in source or re.search(r"故障正在减速", context):
        attributes["motion_state"] = "SLOW"
        descriptors.append("slow_vehicle")
    if "施工车辆" in source or re.search(r"施工车辆", context):
        attributes["role"] = "WORK_VEHICLE"
        descriptors.append("work_vehicle")
    if "加塞车辆" in source or re.search(r"加塞车辆|车辆加塞", context):
        attributes["role"] = "CUT_IN_VEHICLE"
        descriptors.append("cut_in_vehicle")
    if source in {"前车", "前方车辆"}:
        attributes["role"] = "LEAD_VEHICLE"
        descriptors.append("front_vehicle")
    if "乘客" in source:
        attributes["role"] = "BUS_PASSENGER"
        descriptors.append("bus_passenger")
    if entity_type == "CONSTRUCTION_ZONE":
        descriptors.append("construction_zone")
    if entity_type == "TRAFFIC_CONE":
        descriptors.append("traffic_cone")
    if entity_type == "DESTINATION":
        attributes["route_role"] = "ENDPOINT"
        descriptors.extend(("route_goal", "route_endpoint"))
    if entity_type == "LANDMARK" and "公交站" in source:
        attributes["landmark_type"] = "BUS_STOP"
        descriptors.append("bus_stop")
    return attributes, list(dict.fromkeys(descriptors))


def _extract_entities(text: str) -> tuple[list[dict[str, Any]], list[_EntityMention]]:
    patterns: tuple[tuple[str, re.Pattern[str]], ...] = (
        ("PEDESTRIAN", re.compile(r"临时横穿行人|横穿马路的行人|斑马线有行人|行人|乘客")),
        ("SLOW_VEHICLE", re.compile(r"红色慢车|慢车|低速车|故障车辆")),
        ("VEHICLE", re.compile(r"白色车辆|红色车辆|施工车辆|加塞车辆|车辆(?=加塞)|前方公交车|公交车|前方车辆|前车|横向来车")),
        ("CYCLIST", re.compile(r"慢速自行车|自行车|骑行者")),
        ("TRAFFIC_CONE", re.compile(r"锥形桶|锥桶")),
        ("CONSTRUCTION_ZONE", re.compile(r"施工路段|施工区域|施工区")),
        ("LANDMARK", re.compile(r"公交站")),
        ("CROSSWALK", re.compile(r"斑马线")),
        ("JUNCTION", re.compile(r"十字路口|交叉口|路口")),
        ("LANE", re.compile(r"左侧车道|右侧车道|原车道|当前车道|本车道|安全车道")),
        ("DESTINATION", re.compile(r"终点")),
        ("ROAD_HAZARD", re.compile(r"雨雾|能见度差|路面湿滑|湿滑路面")),
    )
    raw_mentions: list[tuple[int, int, str, str]] = []
    for entity_type, pattern in patterns:
        for match in pattern.finditer(text):
            raw_mentions.append((match.start(), match.end(), entity_type, match.group(0)))
    raw_mentions.sort(key=lambda item: (item[0], -(item[1] - item[0])))

    selected: list[tuple[int, int, str, str]] = []
    for mention in raw_mentions:
        start, end, _, _ = mention
        if any(start < other_end and end > other_start for other_start, other_end, _, _ in selected):
            continue
        selected.append(mention)
    selected.sort(key=lambda item: item[0])

    entities: list[dict[str, Any]] = []
    mentions: list[_EntityMention] = []
    for index, (start, end, entity_type, source) in enumerate(selected, start=1):
        relation = _relation(text, start, end, source)
        if entity_type == "LANE":
            if "左" in source:
                relation = "LEFT"
            elif "右" in source:
                relation = "RIGHT"
        attributes, descriptors = _attributes(source, entity_type, text)
        entity = {
            "entity_id": f"scene_entity_{index}",
            "type": entity_type,
            "relation": relation,
            "description": source,
            "canonical_attributes": attributes,
            "open_descriptors": descriptors,
            "source_span": source,
        }
        entities.append(entity)
        mentions.append(_EntityMention(start, end, entity))
    if "施工车辆" in text and not any(
        item["type"] == "CONSTRUCTION_ZONE" for item in entities
    ):
        start = text.index("施工")
        entity = {
            "entity_id": f"scene_entity_{len(entities) + 1}",
            "type": "CONSTRUCTION_ZONE",
            "relation": "AHEAD",
            "description": "施工区域",
            "canonical_attributes": {},
            "open_descriptors": ["construction_zone"],
            "source_span": "施工",
        }
        entities.append(entity)
        mentions.append(_EntityMention(start, start + 2, entity))
        mentions.sort(key=lambda item: item.start)
    if re.search(r"安全(?:车距|距离)|保持车距|拉开车距", text) and not any(
        item["type"] in {"VEHICLE", "SLOW_VEHICLE"} for item in entities
    ):
        match = re.search(r"安全(?:车距|距离)|保持车距|拉开车距", text)
        assert match is not None
        entity = {
            "entity_id": f"scene_entity_{len(entities) + 1}",
            "type": "VEHICLE",
            "relation": "AHEAD",
            "description": "隐含前车",
            "canonical_attributes": {"role": "LEAD_VEHICLE"},
            "open_descriptors": ["front_vehicle", "inferred_lead_vehicle"],
            "source_span": match.group(0),
        }
        entities.append(entity)
        mentions.append(_EntityMention(match.start(), match.end(), entity))
        mentions.sort(key=lambda item: item.start)
    return entities, mentions


def _nearest_entity(
    mentions: list[_EntityMention],
    position: float,
    allowed_types: set[str],
    *,
    direction: str | None = None,
) -> dict[str, Any] | None:
    candidates = [item for item in mentions if item.entity["type"] in allowed_types]
    if direction and "LANE" in allowed_types:
        directional = [
            item for item in candidates if item.entity["relation"] == direction
        ]
        if directional:
            candidates = directional
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            abs(item.start - position),
            0 if item.start <= position else 1,
        ),
    ).entity


def parse_chinese_scene_command(text: str) -> ChineseSceneParse | None:
    if not _SCENE_HINTS.search(text):
        return None

    entities, mentions = _extract_entities(text)
    candidates: list[tuple[float, int, int, dict[str, Any]]] = []
    seen: set[tuple[int, int, str, str | None]] = set()

    def add(
        start: float,
        end: int,
        action: str,
        *,
        direction: str | None = None,
        **values: Any,
    ) -> None:
        key = (int(start * 1000), end, action, direction)
        if key in seen:
            return
        seen.add(key)
        command: dict[str, Any] = {"action": action, **values}
        if direction is not None:
            command["direction"] = direction
        candidates.append((start, len(candidates), end, command))

    speed_matches = list(TARGET_SPEED_PATTERN.finditer(text))
    cap_matches = list(_CAP_SPEED.finditer(text))
    max_speed_mps: float | None = None
    for match in speed_matches + cap_matches:
        value = float(match.group("value"))
        unit = match.group("unit")
        converted = round(speed_to_mps(value, unit), 3)
        if match in cap_matches:
            max_speed_mps = converted
        position = float(match.start())
        turn_before = re.search(r"转弯前[^，,。]*$", text[: match.start()])
        if turn_before:
            first_turn = _TURN.search(text)
            if first_turn:
                position = first_turn.start() - 2.0
        add(
            position,
            match.end(),
            "SET_SPEED",
            target_speed_mps=converted,
            source_value=value,
            source_unit=canonical_speed_unit(unit),
        )

    explicit_speed_spans = [(match.start(), match.end()) for match in speed_matches + cap_matches]
    for match in re.finditer(
        r"降低车速|降低速度|减速|降速|保持低速|低速|保持安全车速|保持安全速度|安全车速|缓行",
        text,
    ):
        if re.search(r"正在$", text[max(0, match.start() - 3) : match.start()]):
            continue
        if any(match.start() < end and match.end() > start for start, end in explicit_speed_spans):
            continue
        add(match.start(), match.end(), "ADJUST_SPEED", change="DECREASE")

    for match in re.finditer(r"提高车速|提升车速|提速|加速", text):
        if any(match.start() < end and match.end() > start for start, end in explicit_speed_spans):
            continue
        add(match.start(), match.end(), "ADJUST_SPEED", change="INCREASE")

    keep_patterns = (
        re.compile(r"保持(?:在)?(?P<direction>左侧|右侧|当前|本)?车道"),
        re.compile(r"维持(?:在)?(?P<direction>左侧|右侧|当前|本)?车道"),
        re.compile(r"保持[^，,。]{0,10}(?P<direction>当前|本)车道"),
        re.compile(r"确认(?P<direction>当前|本)车道安全"),
    )
    for pattern in keep_patterns:
        for match in pattern.finditer(text):
            surface = match.group("direction") or ""
            direction = "LEFT" if "左" in surface else "RIGHT" if "右" in surface else None
            add(match.start(), match.end(), "KEEP_LANE", direction=direction)

    lane_matches: list[tuple[int, int, str | None]] = []
    lane_patterns = (
        re.compile(r"(?:向|往)(?P<direction>左|右)(?:边|侧)?(?:车道)?(?:变道|并道|并线)"),
        re.compile(r"(?:变道|并道|并线|并入|驶入|进入|并到)(?:至|到|向)?(?P<direction>左|右)(?:边|侧)?(?:相邻)?车道"),
        re.compile(r"(?:返回|回到|回归)(?P<direction>左|右|原|安全)(?:边|侧)?车道"),
        re.compile(r"(?P<context>左侧|右侧)安全(?:时|后)?(?:再)?(?P<direction>变道)"),
    )
    for pattern in lane_patterns:
        for match in pattern.finditer(text):
            raw_direction = match.group("direction")
            direction = (
                "LEFT"
                if raw_direction == "左"
                else "RIGHT"
                if raw_direction == "右"
                else None
            )
            context = match.groupdict().get("context")
            if raw_direction == "变道" and context:
                direction = "LEFT" if "左" in context else "RIGHT"
            if (
                raw_direction == "原"
                and match.start() < 4
                and re.match(r"返回原车道后|回到原车道后", text)
            ):
                continue
            lane_matches.append((match.start(), match.end(), direction))

    explicit_directions = [direction for _, _, direction in lane_matches if direction]
    for start, end, direction in lane_matches:
        if direction is None:
            previous = next(
                (
                    item
                    for item_start, _, item in reversed(lane_matches)
                    if item_start < start and item is not None
                ),
                explicit_directions[0] if explicit_directions else None,
            )
            if previous is None:
                prefix = text[max(0, start - 30) : start]
                previous = (
                    "LEFT"
                    if "左侧安全" in prefix
                    else "RIGHT"
                    if "右侧安全" in prefix
                    else None
                )
            direction = "RIGHT" if previous == "LEFT" else "LEFT" if previous == "RIGHT" else None
        if direction is not None:
            add(start, end, "CHANGE_LANE", direction=direction, lane_count=1)
        elif re.match(r"(?:回归|回到)原车道", text[start:end]):
            add(start, end, "RESUME")

    turn_commands: list[tuple[float, int, str]] = []
    for match in _TURN.finditer(text):
        prefix = text[max(0, match.start() - 5) : match.start()]
        if re.search(r"完成|已经|确认已", prefix):
            continue
        surface = match.group("direction")
        direction = (
            "LEFT"
            if surface in {"左转", "左拐"}
            else "RIGHT"
            if surface in {"右转", "右拐"}
            else "STRAIGHT"
        )
        if direction == "STRAIGHT" and text[match.end() :].startswith("通过"):
            continue
        values: dict[str, Any] = {}
        distance_match = re.search(
            r"(?P<distance>\d+(?:\.\d+)?)\s*m(?:后)?\s*(?:前方|前面)?$",
            text[max(0, match.start() - 32) : match.start()],
        )
        if distance_match is not None:
            distance_m = float(distance_match.group("distance"))
            values["distance_m"] = distance_m
            values["trigger"] = {
                "type": "AT_DISTANCE",
                "distance_m": distance_m,
            }
        turn_commands.append((float(match.start()), match.end(), direction))
        add(match.start(), match.end(), "TURN", direction=direction, **values)

    if turn_commands and re.search(r"确认行人安全|确认行人通过|行人安全后", text):
        first_turn = turn_commands[0]
        add(first_turn[0] - 1.0, first_turn[1], "YIELD", inferred_from="pedestrian_clear")

    follow_matches = list(
        re.finditer(
            r"跟随|保持(?:与前车)?安全(?:车距|距离)|维持安全(?:车距|距离)|"
            r"保持车距|拉开车距",
            text,
        )
    )
    if follow_matches:
        explicit = next(
            (match for match in follow_matches if match.group(0) == "跟随"),
            follow_matches[0],
        )
        add(explicit.start(), explicit.end(), "FOLLOW")

    for match in re.finditer(r"礼让|让行|避让", text):
        context = text[max(0, match.start() - 16) : min(len(text), match.end() + 16)]
        if re.search(r"行人|乘客|横向来车|来车", context):
            add(match.start(), match.end(), "YIELD")
        else:
            add(match.start(), match.end(), "AVOID")

    for match in re.finditer(r"绕行|绕开|绕过|避开|躲开", text):
        context = text[max(0, match.start() - 18) : min(len(text), match.end() + 18)]
        direction = "LEFT" if "左" in context else "RIGHT" if "右" in context else None
        action = "OVERTAKE" if re.search(r"慢车|低速车", context) else "AVOID"
        add(match.start(), match.end(), action, direction=direction)

    for match in re.finditer(r"超越|超车|超过", text):
        prefix = text[max(0, match.start() - 8) : match.start()]
        suffix = text[match.end() : min(len(text), match.end() + 8)]
        if re.search(r"已经|确认已|已$", prefix):
            continue
        if re.match(r"\d", suffix):
            continue
        if match.group(0) == "超过" and suffix.startswith("后"):
            continue
        add(match.start(), match.end(), "OVERTAKE")

    for match in re.finditer(r"等待|等(?=.+?(?:通过|离开))", text):
        if text[max(0, match.start() - 4) : match.start()].endswith("停车"):
            continue
        add(
            match.start(),
            match.end(),
            "WAIT",
            condition="target clears the path",
        )

    for match in re.finditer(r"靠边(?:停车)?|停到路边", text):
        add(match.start(), match.end(), "PULL_OVER")

    for match in re.finditer(r"停车(?!场)|停下|停住", text):
        if text[max(0, match.start() - 3) : match.start()].endswith("否则"):
            continue
        add(match.start(), match.end(), "STOP")

    for match in re.finditer(r"继续前进|继续行驶|直行通过", text):
        context = text[max(0, match.start() - 20) : match.start()]
        if re.search(r"乘客|公交站|靠边", text[: match.start()]):
            add(match.start(), match.end(), "RESUME")
        else:
            direction = "STRAIGHT" if "直行" in match.group(0) else None
            add(match.start(), match.end(), "PROCEED", direction=direction)

    for match in re.finditer(r"恢复(?:正常|之前的)?(?:行驶状态|行驶)", text):
        add(match.start(), match.end(), "RESUME")

    for match in re.finditer(r"(?:行驶|驶|前进|导航)?至终点|驶向终点", text):
        add(match.start(), match.end(), "NAVIGATE_TO")

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]))
    commands = [item[3] for item in candidates]

    def assign_target(command: dict[str, Any], position: float) -> None:
        action = command["action"]
        direction = command.get("direction")
        allowed: set[str] = set()
        if action == "CHANGE_LANE":
            allowed = {"LANE"}
        elif action in {"TURN", "PROCEED"}:
            allowed = {"JUNCTION"}
        elif action == "PULL_OVER":
            allowed = {"LANDMARK", "CURB"}
        elif action == "NAVIGATE_TO":
            allowed = {"DESTINATION"}
        elif action in {"WAIT", "YIELD"}:
            allowed = {"PEDESTRIAN", "VEHICLE"}
        elif action == "FOLLOW":
            allowed = {"VEHICLE", "SLOW_VEHICLE", "CYCLIST"}
        elif action == "OVERTAKE":
            allowed = {"SLOW_VEHICLE", "VEHICLE", "CYCLIST"}
        elif action == "AVOID":
            allowed = {
                "VEHICLE",
                "SLOW_VEHICLE",
                "CYCLIST",
                "PEDESTRIAN",
                "TRAFFIC_CONE",
                "CONSTRUCTION_ZONE",
            }
        if not allowed:
            return
        entity = _nearest_entity(mentions, position, allowed, direction=direction)
        if entity is None:
            return
        command["target_ref"] = entity["entity_id"]
        command["target"] = {
            key: entity[key]
            for key in (
                "type",
                "relation",
                "description",
                "canonical_attributes",
                "open_descriptors",
            )
        }
        if action == "FOLLOW" and re.search(
            r"安全(?:车距|距离)|保持车距|拉开车距",
            text,
        ):
            command["goal_conditions"] = [
                {
                    "predicate": "SAFE_DISTANCE",
                    "subject": "ego",
                    "object": entity["entity_id"],
                    "source_span": "安全车距",
                }
            ]

    for (position, _, _, _), command in zip(candidates, commands):
        assign_target(command, position)

    junction = next((item.entity for item in mentions if item.entity["type"] == "JUNCTION"), None)
    if junction and re.search(r"通过(?:前方)?路口后", text):
        for command in commands:
            if command["action"] in {"CHANGE_LANE", "SET_SPEED", "FOLLOW"}:
                command.setdefault(
                    "trigger",
                    {"type": "AFTER_ENTITY", "entity_ref": junction["entity_id"]},
                )
                break

    action_names = {command["action"] for command in commands}
    entity_types = {entity["type"] for entity in entities}
    basic_actions = {
        "KEEP_LANE",
        "CHANGE_LANE",
        "SET_SPEED",
        "ADJUST_SPEED",
        "STOP",
        "FOLLOW",
    }
    hazard_entities = {
        "PEDESTRIAN",
        "SLOW_VEHICLE",
        "CYCLIST",
        "TRAFFIC_CONE",
        "CONSTRUCTION_ZONE",
    }
    if action_names & {"TURN", "PROCEED", "NAVIGATE_TO"}:
        category = "NAVIGATION"
    elif action_names <= basic_actions and not entity_types & hazard_entities:
        category = "BASIC_CONTROL"
    else:
        category = "COMPLEX_OBSTACLE_AVOIDANCE"
    emergency = bool(re.search(r"紧急|突发|危险|突然|施工路段", text))
    urgency = "EMERGENCY" if emergency else "NORMAL"
    style = (
        "CONSERVATIVE"
        if re.search(r"安全|减速|低速|礼让|避让|等待|施工|雨|雾|湿滑", text)
        else "NORMAL"
    )
    return ChineseSceneParse(
        commands=commands,
        entities=entities,
        category="EMERGENCY_RESPONSE" if emergency else category,
        urgency=urgency,
        driving_style=style,
        max_speed_mps=max_speed_mps,
    )
