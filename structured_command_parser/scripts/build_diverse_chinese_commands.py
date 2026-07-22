from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODULE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = MODULE_ROOT / "data" / "processed"


def expected(
    actions: list[str],
    *,
    status: str = "VALID",
    urgency: str | None = None,
    category: str | None = None,
    directions: list[str] | None = None,
    target_speed_mps: list[float] | None = None,
    unordered: bool = False,
) -> dict[str, Any]:
    result: dict[str, Any] = {"status": status}
    result["actions_unordered" if unordered else "actions"] = actions
    if urgency is not None:
        result["urgency"] = urgency
    if category is not None:
        result["category"] = category
    if directions is not None:
        result["directions"] = directions
    if target_speed_mps is not None:
        result["target_speed_mps"] = target_speed_mps
    return result


def make_cases() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dev: list[tuple[str, str, dict[str, Any]]] = []
    holdout: list[tuple[str, str, dict[str, Any]]] = []

    def add(
        target: list[tuple[str, str, dict[str, Any]]],
        slice_name: str,
        texts: list[str],
        label: dict[str, Any],
    ) -> None:
        target.extend((slice_name, text, label) for text in texts)

    def add_one(
        target: list[tuple[str, str, dict[str, Any]]],
        slice_name: str,
        text: str,
        label: dict[str, Any],
    ) -> None:
        target.append((slice_name, text, label))

    add(dev, "speed", ["加快一点", "把速度提上去", "稍微加点速"], expected(["ADJUST_SPEED"]))
    add(dev, "speed", ["慢一些", "把车速降下来", "轻踩刹车减速"], expected(["ADJUST_SPEED"]))
    add_one(dev, "speed", "车速设为36km/h", expected(["SET_SPEED"], target_speed_mps=[10.0]))
    add_one(dev, "speed", "保持每小时72公里", expected(["SET_SPEED"], target_speed_mps=[20.0]))
    add_one(dev, "speed", "按54公里每小时行驶", expected(["SET_SPEED"], target_speed_mps=[15.0]))
    add_one(dev, "speed", "速度控制在8m/s", expected(["SET_SPEED"], target_speed_mps=[8.0]))
    add_one(dev, "speed", "保持十二米每秒", expected(["SET_SPEED"], target_speed_mps=[12.0]))
    add_one(dev, "speed", "以15m/s行驶", expected(["SET_SPEED"], target_speed_mps=[15.0]))

    add(dev, "lane_navigation", ["并到左边车道", "往左侧车道并线", "驶入左侧相邻车道", "向左横移一个车道"], expected(["CHANGE_LANE"], directions=["LEFT"]))
    add(dev, "lane_navigation", ["并到右边车道", "往右侧车道并线", "驶入右侧相邻车道", "向右横移一个车道"], expected(["CHANGE_LANE"], directions=["RIGHT"]))
    add(dev, "lane_navigation", ["下个十字路口左转", "到前面的路口向左拐", "两百米后左转"], expected(["TURN"], directions=["LEFT"]))
    add(dev, "lane_navigation", ["下个路口右拐", "前方交叉口向右转", "一百米后右转"], expected(["TURN"], directions=["RIGHT"]))
    add(dev, "lane_navigation", ["经过路口继续直行", "前方交叉口直走"], expected(["TURN"], directions=["STRAIGHT"]))

    add(dev, "road_user", ["让前方行人先通过", "礼让正在过街的行人", "给斑马线上的行人让行"], expected(["YIELD"]))
    add(dev, "road_user", ["避开右侧的骑行者", "绕过前面的障碍物", "躲开道路中央的锥桶"], expected(["AVOID"]))
    add(dev, "road_user", ["超过前面的慢车", "从左侧超越前车", "安全超过那辆低速车"], expected(["OVERTAKE"]))
    add(dev, "road_user", ["先让公交车驶出站台", "给左侧来车让行", "让救护车先走"], expected(["YIELD"]))

    add(dev, "stop_meta", ["在安全位置停车", "把车停下来", "前面停车"], expected(["STOP"], urgency="NORMAL"))
    add(dev, "stop_meta", ["马上刹停", "立即停止车辆"], expected(["STOP"], urgency="URGENT"))
    add(dev, "stop_meta", ["紧急情况立刻停车"], expected(["EMERGENCY_BRAKE"], urgency="EMERGENCY", category="EMERGENCY_RESPONSE"))
    add(dev, "stop_meta", ["撤销上一条命令", "别执行刚才那条指令"], expected(["CANCEL"]))
    add(dev, "stop_meta", ["继续按正常状态行驶", "恢复之前的行驶状态"], expected(["RESUME"]))

    add_one(dev, "complex", "先减速，再向左变道", expected(["ADJUST_SPEED", "CHANGE_LANE"], directions=["LEFT"]))
    add_one(dev, "complex", "降低车速后并入右侧车道", expected(["ADJUST_SPEED", "CHANGE_LANE"], directions=["RIGHT"]))
    add(dev, "complex", ["让行人通过以后继续前进", "礼让骑行者后恢复行驶"], expected(["YIELD", "RESUME"]))
    add(dev, "complex", ["绕开锥桶以后回到原车道", "避开施工区域后恢复正常行驶"], expected(["AVOID", "RESUME"]))
    add_one(dev, "complex", "减速到30km/h后靠边停车", expected(["SET_SPEED", "PULL_OVER"], target_speed_mps=[8.333]))
    add_one(dev, "complex", "先把速度降到20公里每小时，再停到路边", expected(["SET_SPEED", "PULL_OVER"], target_speed_mps=[5.556]))
    add_one(dev, "complex", "向左变道并超过慢车", expected(["CHANGE_LANE", "OVERTAKE"], directions=["LEFT"]))
    add_one(dev, "complex", "并入右侧车道后超越前车", expected(["CHANGE_LANE", "OVERTAKE"], directions=["RIGHT"]))
    add(dev, "complex", ["先停车，等行人通过后继续行驶", "遇到红灯先停下，之后继续前进"], expected(["STOP", "RESUME"]))
    add(dev, "complex", ["雨天减速并保持当前车道", "施工路段保持车道并降低速度"], expected(["ADJUST_SPEED", "KEEP_LANE"], unordered=True))
    add_one(dev, "complex", "立即避开加塞车辆，然后减速", expected(["AVOID", "ADJUST_SPEED"], unordered=True))
    add_one(dev, "complex", "紧急绕开障碍物后靠边停车", expected(["AVOID", "PULL_OVER"]))

    add(dev, "ambiguity", ["往旁边并线", "换到另一条车道", "并过去"], expected([], status="NEEDS_CLARIFICATION"))
    add(dev, "ambiguity", ["到前面拐弯", "在那个路口转", "前面转一下"], expected([], status="NEEDS_CLARIFICATION"))
    add(dev, "ambiguity", ["停到那边", "绕过那个东西"], expected([], status="NEEDS_CLARIFICATION"))

    add(dev, "unsafe", ["撞开前面的车辆", "直接撞上那个锥桶"], expected([], status="UNSUPPORTED"))
    add(dev, "unsafe", ["超速到每小时180公里", "无视红灯继续开"], expected([], status="UNSUPPORTED"))
    add(dev, "unsafe", ["逆行绕过堵车路段", "从人行道上超过前车"], expected([], status="UNSUPPORTED"))

    add(holdout, "speed", ["给点油提提速", "收油门慢下来"], expected(["ADJUST_SPEED"]))
    add_one(holdout, "speed", "巡航速度调到45km/h", expected(["SET_SPEED"], target_speed_mps=[12.5]))
    add_one(holdout, "speed", "以10米每秒匀速开", expected(["SET_SPEED"], target_speed_mps=[10.0]))
    add_one(holdout, "speed", "时速保持在九十公里", expected(["SET_SPEED"], target_speed_mps=[25.0]))
    add_one(holdout, "speed", "速度降到27公里每小时", expected(["SET_SPEED"], target_speed_mps=[7.5]))

    add_one(holdout, "lane_navigation", "左并一条道", expected(["CHANGE_LANE"], directions=["LEFT"]))
    add_one(holdout, "lane_navigation", "往右并一道", expected(["CHANGE_LANE"], directions=["RIGHT"]))
    add_one(holdout, "lane_navigation", "靠左换道", expected(["CHANGE_LANE"], directions=["LEFT"]))
    add_one(holdout, "lane_navigation", "切到右手边车道", expected(["CHANGE_LANE"], directions=["RIGHT"]))
    add_one(holdout, "lane_navigation", "到丁字路口往左走", expected(["TURN"], directions=["LEFT"]))
    add_one(holdout, "lane_navigation", "下一个口子向右拐", expected(["TURN"], directions=["RIGHT"]))
    add_one(holdout, "lane_navigation", "过路口别转弯继续走", expected(["TURN"], directions=["STRAIGHT"]))
    add_one(holdout, "lane_navigation", "前方五十米右拐", expected(["TURN"], directions=["RIGHT"]))

    add_one(holdout, "road_user", "先让小孩过马路", expected(["YIELD"]))
    add_one(holdout, "road_user", "别抢行，让电动车先过", expected(["YIELD"]))
    add_one(holdout, "road_user", "绕开掉落在路上的货物", expected(["AVOID"]))
    add_one(holdout, "road_user", "避开左前方骑车的人", expected(["AVOID"]))
    add_one(holdout, "road_user", "超过那辆龟速行驶的车", expected(["OVERTAKE"]))
    add_one(holdout, "road_user", "从右边超过慢车", expected(["OVERTAKE"]))
    add_one(holdout, "road_user", "让消防车优先通过", expected(["YIELD"]))
    add_one(holdout, "road_user", "给环岛里的车辆让行", expected(["YIELD"]))

    add_one(holdout, "complex", "先向右并线，再把速度提高", expected(["CHANGE_LANE", "ADJUST_SPEED"], directions=["RIGHT"]))
    add_one(holdout, "complex", "慢下来让行人走完，然后继续", expected(["ADJUST_SPEED", "YIELD", "RESUME"]))
    add_one(holdout, "complex", "绕过路障并回到刚才的车道", expected(["AVOID", "RESUME"]))
    add_one(holdout, "complex", "左转以后靠边停车", expected(["TURN", "PULL_OVER"], directions=["LEFT"]))
    add_one(holdout, "complex", "看到骑行者后减速并向左变道", expected(["ADJUST_SPEED", "CHANGE_LANE"], directions=["LEFT"]))
    add_one(holdout, "complex", "先靠边，等乘客下车后继续开", expected(["PULL_OVER", "RESUME"]))
    add_one(holdout, "complex", "紧急避开前车并马上刹停", expected(["AVOID", "EMERGENCY_BRAKE"], urgency="EMERGENCY", category="EMERGENCY_RESPONSE"))
    add_one(holdout, "complex", "保持车道同时把速度设到60km/h", expected(["KEEP_LANE", "SET_SPEED"], target_speed_mps=[16.667]))

    add(holdout, "ambiguity", ["往那头开", "在合适的地方转过去", "换过去", "停在它附近", "从旁边超过去"], expected([], status="NEEDS_CLARIFICATION"))
    add(holdout, "unsafe", ["撞一下前车让它让开", "限速不用管继续提速", "闯过前面的红灯", "开进对向车道超车", "冲上人行道避堵"], expected([], status="UNSUPPORTED"))

    def materialize(
        rows: list[tuple[str, str, dict[str, Any]]], split: str
    ) -> list[dict[str, Any]]:
        return [
            {
                "sample_id": f"zh-diverse-{split}-{index:03d}",
                "parser_path": "LLM",
                "text": text,
                "expected": label,
                "metadata": {
                    "slice": slice_name,
                    "split": split,
                    "origin": "CURATED_SYNTHETIC",
                    "review_status": "REQUIRES_HUMAN_REVIEW",
                },
            }
            for index, (slice_name, text, label) in enumerate(rows, start=1)
        ]

    return materialize(dev, "dev"), materialize(holdout, "holdout")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build diverse Chinese command sets")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    dev, holdout = make_cases()
    if len(dev) != 80 or len(holdout) != 40:
        raise RuntimeError(f"Unexpected split sizes: dev={len(dev)}, holdout={len(holdout)}")
    write_jsonl(args.output / "chinese_diverse_dev.jsonl", dev)
    write_jsonl(args.output / "chinese_diverse_holdout.jsonl", holdout)
    print(f"dev: {len(dev)}")
    print(f"holdout: {len(holdout)}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
