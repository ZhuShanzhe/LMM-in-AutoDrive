"""Build the reviewed Scene 2 DrivingIntent inputs used by the closed loop.

These documents are an explicit test oracle. They do not replace or modify the
parser result recorded by the parser benchmark. The integrated runner may use
them when an actual parser result does not cover the reviewed command contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
COMMAND_SUITE = ROOT / "experiment" / "CARLA" / "configs" / "scene_2_command_suite.json"
OUTPUT = (
    ROOT
    / "experiment"
    / "CARLA"
    / "configs"
    / "scene_2_expected_driving_intents.json"
)

TRANSLATIONS = {
    "s2_t05_cmd_01": "Keep the current lane, slow down to 45 km/h, pass the junction ahead, and continue straight.",
    "s2_t05_cmd_02": "After passing the junction, keep the current lane, slow down to 40 km/h, follow the white vehicle ahead, and maintain a safe following distance.",
    "s2_t05_cmd_03": "When you see the pedestrian crossing ahead, slow down and yield; after the pedestrian clears, change to the left lane and overtake the red slow vehicle.",
    "s2_t05_cmd_04": "After confirming that the red slow vehicle has been passed, safely return to the right lane, resume 45 km/h, and maintain a safe following distance.",
    "s2_t05_cmd_05": "Passengers are boarding and alighting at the bus stop ahead; pull over, slow down to 30 km/h, and continue after the passengers leave the lane.",
    "s2_t05_cmd_06": "Wait for the last passenger to leave, confirm that the bus remains stopped, change to the left lane, overtake the bus, and return to the right lane.",
    "s2_t05_cmd_07": "There is a slow cyclist on the right; slow down, change to the left lane to avoid it, pass the cyclist, and return to the right lane.",
    "s2_t05_cmd_08": "Slow down when approaching the intersection, yield to cross traffic, and proceed straight after the intersection is clear.",
    "s2_t05_cmd_09": "Turn left at the intersection ahead, slow down to 30 km/h before the turn, and enter the right lane after completing the turn.",
    "s2_t05_cmd_10": "After the left turn, keep the current lane, follow the bus ahead at a safe distance, and drive at 35 km/h.",
    "s2_t05_cmd_11": "Turn right at the junction ahead, slow down to 30 km/h before turning, yield to pedestrians, complete the right turn, and keep the lane.",
    "s2_t05_cmd_12": "The red vehicle ahead has broken down and is slowing; reduce speed, change to the left lane when safe, overtake it, and return to the original lane.",
    "s2_t05_cmd_13": "After returning to the original lane, resume 45 km/h, maintain a safe following distance, pass the junction ahead, and continue straight.",
    "s2_t05_cmd_14": "There is a pedestrian at the crosswalk ahead; slow down to 25 km/h, stop and yield, then turn right and continue after the pedestrian clears.",
    "s2_t05_cmd_15": "After confirming the road is safe, keep the current lane, resume 45 km/h, maintain a safe following distance, and drive to the destination.",
}

COMMAND_TEXTS = {
    "s2_t05_cmd_01": "保持当前车道，减速至45公里每小时，通过前方路口后继续直行。",
    "s2_t05_cmd_02": "通过路口后保持当前车道，减速至40公里每小时，跟随前方白色车辆，并保持安全车距。",
    "s2_t05_cmd_03": "看到前方横穿马路的行人，减速避让，行人通过后向左变道超越红色慢车。",
    "s2_t05_cmd_04": "确认已经超过红色慢车，安全返回右侧车道，恢复至45公里每小时并保持安全车距。",
    "s2_t05_cmd_05": "前方公交站有行人上下车，靠边减速至30公里每小时，确认乘客离开车道后继续行驶。",
    "s2_t05_cmd_06": "等待最后一名乘客离开，确认公交车没有起步后向左变道，超过公交车再返回右侧车道。",
    "s2_t05_cmd_07": "右侧有慢速自行车，先减速并向左变道避让，超过自行车后回到右侧车道。",
    "s2_t05_cmd_08": "接近前方十字路口时减速，礼让横向来车，确认路口清空后直行通过。",
    "s2_t05_cmd_09": "前方十字路口左转，转弯前减速至30公里每小时，完成左转后进入右侧车道。",
    "s2_t05_cmd_10": "完成左转后保持当前车道，跟随前方公交车并保持安全车距，以35公里每小时行驶。",
    "s2_t05_cmd_11": "前方路口右转，转弯前减速至30公里每小时，礼让行人，完成右转后保持当前车道。",
    "s2_t05_cmd_12": "前方红色车辆发生故障并减速，先减速，确认安全后向左变道超越，再返回原车道。",
    "s2_t05_cmd_13": "返回原车道后恢复至45公里每小时，保持安全车距，通过前方路口后继续直行。",
    "s2_t05_cmd_14": "前方人行横道有行人，减速至25公里每小时并停车礼让，行人通过后右转并继续行驶。",
    "s2_t05_cmd_15": "确认道路安全后保持当前车道，恢复至45公里每小时，保持安全车距并行驶至终点。",
}


def entity(
    entity_id: str,
    entity_type: str,
    relation: str,
    description: str,
    *,
    attributes: dict[str, Any] | None = None,
    descriptors: list[str] | None = None,
    source_span: str | None = None,
) -> dict[str, Any]:
    return {
        "entity_id": entity_id,
        "type": entity_type,
        "relation": relation,
        "description": description,
        "canonical_attributes": attributes or {},
        "open_descriptors": descriptors or [],
        "source_span": source_span or description,
    }


def step(
    action: str,
    *,
    target_ref: str | None = None,
    parameters: dict[str, Any] | None = None,
    completion: str | None = None,
    trigger: dict[str, Any] | None = None,
    preconditions: list[str] | None = None,
    on_blocked: str = "WAIT_FOR_SAFE",
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "action": action,
        "parameters": parameters or {},
        "preconditions": preconditions or [],
        "on_blocked": on_blocked,
    }
    if target_ref is not None:
        result["target_ref"] = target_ref
    if completion is not None:
        result["completion"] = {"type": completion}
    if trigger is not None:
        result["trigger"] = trigger
    return result


def reviewed_contracts() -> dict[str, dict[str, Any]]:
    lane_left = lambda: entity("lane_left", "LANE", "LEFT", "左侧车道")
    lane_right = lambda: entity("lane_right", "LANE", "RIGHT", "右侧车道")
    junction = lambda: entity(
        "junction_ahead", "JUNCTION", "AT_JUNCTION", "前方路口"
    )
    lead_vehicle = lambda: entity(
        "lead_vehicle", "VEHICLE", "AHEAD", "前方车辆"
    )
    pedestrian = lambda: entity(
        "crossing_pedestrian",
        "PEDESTRIAN",
        "AHEAD_CROSSING",
        "前方横穿道路的行人",
    )
    red_slow_vehicle = lambda: entity(
        "red_slow_vehicle",
        "SLOW_VEHICLE",
        "AHEAD",
        "前方红色慢车",
        attributes={"color": "RED"},
        descriptors=["slow"],
        source_span="红色慢车",
    )
    bus = lambda: entity(
        "stopped_bus",
        "VEHICLE",
        "AHEAD",
        "前方公交车",
        attributes={"vehicle_subtype": "BUS"},
        source_span="公交车",
    )
    passenger = lambda: entity(
        "bus_passenger",
        "PEDESTRIAN",
        "AHEAD_CROSSING",
        "公交站上下车并可能进入车道的乘客",
        source_span="行人上下车",
    )
    cyclist = lambda: entity(
        "slow_cyclist",
        "CYCLIST",
        "FRONT_RIGHT",
        "右前方慢速自行车",
        source_span="右侧有慢速自行车",
    )

    return {
        "s2_t05_cmd_01": {
            "entities": [junction()],
            "steps": [
                step("KEEP_LANE", completion="ACTION_REACHED"),
                step(
                    "SET_SPEED",
                    parameters={
                        "target_speed_mps": 12.5,
                        "source_value": 45,
                        "source_unit": "km/h",
                    },
                    completion="TARGET_SPEED_REACHED",
                ),
                step(
                    "TURN",
                    target_ref="junction_ahead",
                    parameters={"direction": "STRAIGHT"},
                    completion="JUNCTION_EXITED",
                    preconditions=["JUNCTION_REACHED", "PATH_CLEAR"],
                ),
            ],
        },
        "s2_t05_cmd_02": {
            "entities": [
                entity(
                    "white_vehicle",
                    "VEHICLE",
                    "AHEAD",
                    "前方白色车辆",
                    attributes={"color": "WHITE"},
                    source_span="前方白色车辆",
                ),
            ],
            "steps": [
                step("KEEP_LANE", completion="ACTION_REACHED"),
                step(
                    "SET_SPEED",
                    parameters={
                        "target_speed_mps": 40.0 / 3.6,
                        "source_value": 40,
                        "source_unit": "km/h",
                    },
                    completion="TARGET_SPEED_REACHED",
                ),
                step(
                    "FOLLOW",
                    target_ref="white_vehicle",
                    parameters={"following_distance_m": 18.0},
                    completion="FOLLOWING_ESTABLISHED",
                    preconditions=["TARGET_VISIBLE", "PATH_CLEAR"],
                ),
            ],
        },
        "s2_t05_cmd_03": {
            "entities": [pedestrian(), lane_left(), red_slow_vehicle()],
            "steps": [
                step(
                    "ADJUST_SPEED",
                    target_ref="crossing_pedestrian",
                    parameters={"change": "DECREASE"},
                    completion="ACTION_REACHED",
                    preconditions=["TARGET_VISIBLE"],
                ),
                step(
                    "YIELD",
                    target_ref="crossing_pedestrian",
                    completion="TARGET_CLEARED",
                    preconditions=["TARGET_VISIBLE"],
                ),
                step(
                    "CHANGE_LANE",
                    target_ref="lane_left",
                    parameters={"direction": "LEFT"},
                    completion="LANE_CHANGE_COMPLETED",
                    preconditions=[
                        "LEFT_LANE_EXISTS",
                        "LEFT_LANE_SAFE",
                        "LANE_CHANGE_LEGAL",
                    ],
                ),
                step(
                    "OVERTAKE",
                    target_ref="red_slow_vehicle",
                    completion="TARGET_CLEARED",
                    preconditions=["TARGET_VISIBLE", "PATH_CLEAR"],
                ),
            ],
        },
        "s2_t05_cmd_04": {
            "entities": [lane_right(), red_slow_vehicle(), lead_vehicle()],
            "steps": [
                step(
                    "CHANGE_LANE",
                    target_ref="lane_right",
                    parameters={"direction": "RIGHT"},
                    completion="LANE_CHANGE_COMPLETED",
                    preconditions=[
                        "RIGHT_LANE_EXISTS",
                        "RIGHT_LANE_SAFE",
                        "LANE_CHANGE_LEGAL",
                    ],
                ),
                step(
                    "SET_SPEED",
                    parameters={
                        "target_speed_mps": 12.5,
                        "source_value": 45,
                        "source_unit": "km/h",
                    },
                    completion="TARGET_SPEED_REACHED",
                ),
                step(
                    "FOLLOW",
                    target_ref="lead_vehicle",
                    parameters={"following_distance_m": 18.0},
                    completion="FOLLOWING_ESTABLISHED",
                    preconditions=["TARGET_VISIBLE", "PATH_CLEAR"],
                ),
            ],
        },
        "s2_t05_cmd_05": {
            "entities": [bus(), passenger()],
            "steps": [
                step(
                    "PULL_OVER",
                    target_ref="stopped_bus",
                    parameters={"direction": "RIGHT"},
                    completion="ACTION_REACHED",
                    preconditions=["TARGET_VISIBLE", "PATH_CLEAR"],
                ),
                step(
                    "SET_SPEED",
                    parameters={
                        "target_speed_mps": 8.333333,
                        "source_value": 30,
                        "source_unit": "km/h",
                    },
                    completion="TARGET_SPEED_REACHED",
                ),
                step(
                    "WAIT",
                    target_ref="bus_passenger",
                    completion="TARGET_CLEARED",
                    trigger={
                        "type": "CONDITION",
                        "description": "等待乘客离开车辆行驶路径",
                    },
                    preconditions=["TARGET_VISIBLE"],
                ),
                step("RESUME", completion="ACTION_REACHED"),
            ],
        },
        "s2_t05_cmd_06": {
            "entities": [passenger(), bus(), lane_left(), lane_right()],
            "steps": [
                step(
                    "WAIT",
                    target_ref="bus_passenger",
                    completion="TARGET_CLEARED",
                    trigger={
                        "type": "CONDITION",
                        "description": "等待最后一名乘客离开",
                    },
                    preconditions=["TARGET_VISIBLE"],
                ),
                step(
                    "CHANGE_LANE",
                    target_ref="lane_left",
                    parameters={"direction": "LEFT"},
                    completion="LANE_CHANGE_COMPLETED",
                    preconditions=[
                        "LEFT_LANE_EXISTS",
                        "LEFT_LANE_SAFE",
                        "LANE_CHANGE_LEGAL",
                    ],
                ),
                step(
                    "OVERTAKE",
                    target_ref="stopped_bus",
                    completion="TARGET_CLEARED",
                    preconditions=["TARGET_VISIBLE", "PATH_CLEAR"],
                ),
                step(
                    "CHANGE_LANE",
                    target_ref="lane_right",
                    parameters={"direction": "RIGHT"},
                    completion="LANE_CHANGE_COMPLETED",
                    preconditions=[
                        "RIGHT_LANE_EXISTS",
                        "RIGHT_LANE_SAFE",
                        "LANE_CHANGE_LEGAL",
                    ],
                ),
            ],
        },
        "s2_t05_cmd_07": {
            "entities": [cyclist(), lane_left(), lane_right()],
            "steps": [
                step(
                    "ADJUST_SPEED",
                    target_ref="slow_cyclist",
                    parameters={"change": "DECREASE"},
                    completion="ACTION_REACHED",
                    preconditions=["TARGET_VISIBLE"],
                ),
                step(
                    "CHANGE_LANE",
                    target_ref="lane_left",
                    parameters={"direction": "LEFT"},
                    completion="LANE_CHANGE_COMPLETED",
                    preconditions=[
                        "LEFT_LANE_EXISTS",
                        "LEFT_LANE_SAFE",
                        "LANE_CHANGE_LEGAL",
                    ],
                ),
                step(
                    "AVOID",
                    target_ref="slow_cyclist",
                    completion="TARGET_CLEARED",
                    preconditions=["TARGET_VISIBLE", "PATH_CLEAR"],
                ),
                step(
                    "OVERTAKE",
                    target_ref="slow_cyclist",
                    completion="TARGET_CLEARED",
                    preconditions=["TARGET_VISIBLE", "PATH_CLEAR"],
                ),
                step(
                    "CHANGE_LANE",
                    target_ref="lane_right",
                    parameters={"direction": "RIGHT"},
                    completion="LANE_CHANGE_COMPLETED",
                    preconditions=[
                        "RIGHT_LANE_EXISTS",
                        "RIGHT_LANE_SAFE",
                        "LANE_CHANGE_LEGAL",
                    ],
                ),
            ],
        },
        "s2_t05_cmd_08": {
            "entities": [
                junction(),
                entity(
                    "cross_traffic",
                    "VEHICLE",
                    "LEFT",
                    "路口横向来车",
                    source_span="横向来车",
                ),
            ],
            "steps": [
                step(
                    "ADJUST_SPEED",
                    target_ref="cross_traffic",
                    parameters={"change": "DECREASE"},
                    completion="ACTION_REACHED",
                    preconditions=["TARGET_VISIBLE"],
                ),
                step(
                    "YIELD",
                    target_ref="cross_traffic",
                    completion="TARGET_CLEARED",
                    preconditions=["TARGET_VISIBLE"],
                ),
                step(
                    "PROCEED",
                    target_ref="junction_ahead",
                    parameters={"direction": "STRAIGHT"},
                    completion="JUNCTION_EXITED",
                    preconditions=["JUNCTION_REACHED", "PATH_CLEAR"],
                ),
            ],
        },
        "s2_t05_cmd_09": {
            "entities": [junction(), lane_right()],
            "steps": [
                step(
                    "SET_SPEED",
                    parameters={
                        "target_speed_mps": 8.333333,
                        "source_value": 30,
                        "source_unit": "km/h",
                    },
                    completion="TARGET_SPEED_REACHED",
                ),
                step(
                    "TURN",
                    target_ref="junction_ahead",
                    parameters={"direction": "LEFT"},
                    completion="JUNCTION_EXITED",
                    preconditions=["JUNCTION_REACHED", "PATH_CLEAR"],
                ),
                step(
                    "CHANGE_LANE",
                    target_ref="lane_right",
                    parameters={"direction": "RIGHT"},
                    completion="LANE_CHANGE_COMPLETED",
                    preconditions=[
                        "RIGHT_LANE_EXISTS",
                        "RIGHT_LANE_SAFE",
                        "LANE_CHANGE_LEGAL",
                    ],
                ),
            ],
        },
        "s2_t05_cmd_10": {
            "entities": [bus()],
            "steps": [
                step("KEEP_LANE", completion="ACTION_REACHED"),
                step(
                    "SET_SPEED",
                    parameters={
                        "target_speed_mps": 9.722222,
                        "source_value": 35,
                        "source_unit": "km/h",
                    },
                    completion="TARGET_SPEED_REACHED",
                ),
                step(
                    "FOLLOW",
                    target_ref="stopped_bus",
                    parameters={"following_distance_m": 18.0},
                    completion="FOLLOWING_ESTABLISHED",
                    preconditions=["TARGET_VISIBLE", "PATH_CLEAR"],
                ),
            ],
        },
        "s2_t05_cmd_11": {
            "entities": [junction(), pedestrian()],
            "steps": [
                step(
                    "SET_SPEED",
                    parameters={
                        "target_speed_mps": 8.333333,
                        "source_value": 30,
                        "source_unit": "km/h",
                    },
                    completion="TARGET_SPEED_REACHED",
                ),
                step(
                    "YIELD",
                    target_ref="crossing_pedestrian",
                    completion="TARGET_CLEARED",
                    preconditions=["TARGET_VISIBLE"],
                ),
                step(
                    "TURN",
                    target_ref="junction_ahead",
                    parameters={"direction": "RIGHT"},
                    completion="JUNCTION_EXITED",
                    preconditions=["JUNCTION_REACHED", "PATH_CLEAR"],
                ),
                step("KEEP_LANE", completion="ACTION_REACHED"),
            ],
        },
        "s2_t05_cmd_12": {
            "entities": [red_slow_vehicle(), lane_left(), lane_right()],
            "steps": [
                step(
                    "ADJUST_SPEED",
                    target_ref="red_slow_vehicle",
                    parameters={"change": "DECREASE"},
                    completion="ACTION_REACHED",
                    preconditions=["TARGET_VISIBLE"],
                ),
                step(
                    "CHANGE_LANE",
                    target_ref="lane_left",
                    parameters={"direction": "LEFT"},
                    completion="LANE_CHANGE_COMPLETED",
                    preconditions=[
                        "LEFT_LANE_EXISTS",
                        "LEFT_LANE_SAFE",
                        "LANE_CHANGE_LEGAL",
                    ],
                ),
                step(
                    "OVERTAKE",
                    target_ref="red_slow_vehicle",
                    completion="TARGET_CLEARED",
                    preconditions=["TARGET_VISIBLE", "PATH_CLEAR"],
                ),
                step(
                    "CHANGE_LANE",
                    target_ref="lane_right",
                    parameters={"direction": "RIGHT"},
                    completion="LANE_CHANGE_COMPLETED",
                    preconditions=[
                        "RIGHT_LANE_EXISTS",
                        "RIGHT_LANE_SAFE",
                        "LANE_CHANGE_LEGAL",
                    ],
                ),
            ],
        },
        "s2_t05_cmd_13": {
            "entities": [lead_vehicle(), junction()],
            "steps": [
                step(
                    "SET_SPEED",
                    parameters={
                        "target_speed_mps": 12.5,
                        "source_value": 45,
                        "source_unit": "km/h",
                    },
                    completion="TARGET_SPEED_REACHED",
                ),
                step(
                    "FOLLOW",
                    target_ref="lead_vehicle",
                    parameters={"following_distance_m": 18.0},
                    completion="FOLLOWING_ESTABLISHED",
                    preconditions=["TARGET_VISIBLE", "PATH_CLEAR"],
                ),
                step(
                    "TURN",
                    target_ref="junction_ahead",
                    parameters={"direction": "STRAIGHT"},
                    completion="JUNCTION_EXITED",
                    preconditions=["JUNCTION_REACHED", "PATH_CLEAR"],
                ),
            ],
        },
        "s2_t05_cmd_14": {
            "entities": [pedestrian(), junction()],
            "steps": [
                step(
                    "SET_SPEED",
                    parameters={
                        "target_speed_mps": 6.944444,
                        "source_value": 25,
                        "source_unit": "km/h",
                    },
                    completion="TARGET_SPEED_REACHED",
                ),
                step(
                    "STOP",
                    target_ref="crossing_pedestrian",
                    completion="STOPPED_BEFORE_TARGET",
                    preconditions=["TARGET_VISIBLE"],
                    on_blocked="SAFE_STOP",
                ),
                step(
                    "YIELD",
                    target_ref="crossing_pedestrian",
                    completion="TARGET_CLEARED",
                    preconditions=["TARGET_VISIBLE"],
                ),
                step(
                    "TURN",
                    target_ref="junction_ahead",
                    parameters={"direction": "RIGHT"},
                    completion="JUNCTION_EXITED",
                    preconditions=["JUNCTION_REACHED", "PATH_CLEAR"],
                ),
                step(
                    "PROCEED",
                    parameters={"direction": "STRAIGHT"},
                    completion="ACTION_REACHED",
                ),
            ],
        },
        "s2_t05_cmd_15": {
            "entities": [lead_vehicle()],
            "steps": [
                step("KEEP_LANE", completion="ACTION_REACHED"),
                step(
                    "SET_SPEED",
                    parameters={
                        "target_speed_mps": 12.5,
                        "source_value": 45,
                        "source_unit": "km/h",
                    },
                    completion="TARGET_SPEED_REACHED",
                ),
                step(
                    "FOLLOW",
                    target_ref="lead_vehicle",
                    parameters={"following_distance_m": 18.0},
                    completion="FOLLOWING_ESTABLISHED",
                    preconditions=["TARGET_VISIBLE", "PATH_CLEAR"],
                ),
                step(
                    "PROCEED",
                    parameters={"direction": "STRAIGHT"},
                    completion="TARGET_REACHED",
                ),
            ],
        },
    }


def finalize_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    finalized: list[dict[str, Any]] = []
    for index, source in enumerate(steps, start=1):
        item = dict(source)
        step_id = f"step_{index}"
        previous = f"step_{index - 1}" if index > 1 else None
        item["step_id"] = step_id
        if "trigger" not in item:
            item["trigger"] = (
                {"type": "IMMEDIATE"}
                if previous is None
                else {"type": "AFTER_STEP", "step_id": previous}
            )
        item["depends_on"] = [] if previous is None else [previous]
        finalized.append(item)
    return finalized


def build_document(command: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "1.2.0",
        "request_id": command["id"],
        "input": {
            "modality": "VOICE",
            "language": "zh-CN",
            "raw_text": command["text"],
            "translated_text": TRANSLATIONS[command["id"]],
            "source_language": "zh-CN",
            "normalized_text": command["text"],
        },
        "normalization": {"edits": [], "unresolved_references": []},
        "intent": {
            "category": "COMPLEX_OBSTACLE_AVOIDANCE",
            "urgency": "NORMAL",
            "entities": contract["entities"],
            "suppressed_intents": [],
            "steps": finalize_steps(contract["steps"]),
            "constraints": {
                "safety_first": True,
                "obey_traffic_rules": True,
                "driving_style": "CONSERVATIVE",
            },
        },
        "parse_result": {
            "status": "VALID",
            "method": "RULE",
            "model": "scene2-reviewed-contract-v1",
            "confidence": 1.0,
            "missing_slots": [],
            "warnings": [
                "Reviewed execution contract; not the raw parser model output."
            ],
            "latency_ms": 0.0,
        },
    }


def main() -> None:
    suite = json.loads(COMMAND_SUITE.read_text(encoding="utf-8"))
    for command in suite["commands"]:
        command["text"] = COMMAND_TEXTS[command["id"]]
    suite["input"]["description"] = (
        "场景二离线解析和 CARLA 闭环共用的15条中文复合指令"
    )
    COMMAND_SUITE.write_text(
        json.dumps(suite, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    contracts = reviewed_contracts()
    command_ids = [item["id"] for item in suite["commands"]]
    if set(command_ids) != set(contracts):
        missing = sorted(set(command_ids) - set(contracts))
        extra = sorted(set(contracts) - set(command_ids))
        raise RuntimeError(f"contract mismatch: missing={missing}, extra={extra}")
    result = {
        "schema_version": "scene_2_expected_driving_intents/v1",
        "scene_id": suite["scene_id"],
        "provenance": {
            "source": "reviewed_scene_2_command_contract",
            "raw_parser_outputs_modified": False,
            "purpose": "closed_loop_oracle_and_parser_gap_fallback",
        },
        "driving_intents": [
            build_document(command, contracts[command["id"]])
            for command in suite["commands"]
        ],
    }
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(result['driving_intents'])} intents to {OUTPUT}")


if __name__ == "__main__":
    main()
