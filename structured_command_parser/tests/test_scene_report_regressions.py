from __future__ import annotations

import unittest

from structured_command_parser.src.rule_parser import RuleIntentParser
from structured_command_parser.src.schema_tools import validate_document


SCENE_2_CASES = (
    (
        "保持当前车道，减速至45公里每小时，通过前方路口后继续直行。",
        ["KEEP_LANE", "SET_SPEED", "TURN"],
    ),
    (
        "通过路口后向右变道，跟随前方白色车辆，并保持安全车距。",
        ["CHANGE_LANE", "FOLLOW"],
    ),
    (
        "看到前方横穿马路的行人，减速避让，行人通过后向左变道超越红色慢车。",
        ["ADJUST_SPEED", "YIELD", "CHANGE_LANE", "OVERTAKE"],
    ),
    (
        "确认已经超过红色慢车，安全返回右侧车道，恢复至45公里每小时并保持车距。",
        ["CHANGE_LANE", "SET_SPEED", "FOLLOW"],
    ),
    (
        "前方公交站有行人上下车，靠边减速至30公里每小时，确认乘客离开车道后继续行驶。",
        ["PULL_OVER", "SET_SPEED", "RESUME"],
    ),
    (
        "等待最后一名乘客离开，确认公交车没有起步后向左变道，超过公交车再返回右侧车道。",
        ["WAIT", "CHANGE_LANE", "OVERTAKE", "CHANGE_LANE"],
    ),
    (
        "右侧有慢速自行车，先减速并向左变道避让，超过自行车后回到右侧车道。",
        ["ADJUST_SPEED", "CHANGE_LANE", "AVOID", "OVERTAKE", "CHANGE_LANE"],
    ),
    (
        "接近前方十字路口时减速，礼让横向来车，确认路口清空后直行通过。",
        ["ADJUST_SPEED", "YIELD", "PROCEED"],
    ),
    (
        "前方十字路口左转，转弯前减速至30公里每小时，完成左转后进入右侧车道。",
        ["SET_SPEED", "TURN", "CHANGE_LANE"],
    ),
    (
        "完成左转后保持当前车道，跟随前方公交车，维持安全车距并以35公里每小时行驶。",
        ["KEEP_LANE", "FOLLOW", "SET_SPEED"],
    ),
    (
        "前方路口右转，转弯前减速至30公里每小时，确认行人安全后完成右转并保持车道。",
        ["SET_SPEED", "YIELD", "TURN", "KEEP_LANE"],
    ),
    (
        "前方红色车辆发生故障正在减速，先降低车速，确认左侧安全后变道超越，再返回原车道。",
        ["ADJUST_SPEED", "CHANGE_LANE", "OVERTAKE", "CHANGE_LANE"],
    ),
    (
        "返回原车道后恢复至45公里每小时，保持安全车距，通过前方路口后继续直行。",
        ["SET_SPEED", "FOLLOW", "TURN"],
    ),
    (
        "前方斑马线有行人，减速至25公里每小时并停车礼让，行人离开后右转继续行驶。",
        ["SET_SPEED", "STOP", "YIELD", "TURN", "PROCEED"],
    ),
    (
        "确认道路安全后保持当前车道，恢复至45公里每小时，维持安全车距并行驶至终点。",
        ["KEEP_LANE", "SET_SPEED", "FOLLOW", "NAVIGATE_TO"],
    ),
)


SCENE_3_CASES = (
    ("前方路况危险，保持安全车速。", {"ADJUST_SPEED"}),
    ("雨天路面湿滑，降低车速并保持与前车安全距离。", {"ADJUST_SPEED", "FOLLOW"}),
    ("能见度差，保持右侧车道，车速不高于40公里每小时。", {"KEEP_LANE", "SET_SPEED"}),
    ("前车距离过近，减速拉开车距，准备随时刹车。", {"ADJUST_SPEED", "FOLLOW"}),
    ("突发车辆加塞，紧急避让。", {"AVOID"}),
    ("加塞车辆通过后，确认当前车道安全，低速继续行驶。", {"KEEP_LANE", "ADJUST_SPEED", "PROCEED"}),
    ("施工路段，减速并道至左侧车道。", {"ADJUST_SPEED", "CHANGE_LANE"}),
    ("前方锥桶逐渐收窄车道，减速确认左侧安全后并入左侧车道。", {"ADJUST_SPEED", "CHANGE_LANE"}),
    ("看到锥形桶和临时横穿行人，保持低速，必要时停车。", {"STOP"}),
    ("等临时横穿行人通过后，保持低速，确认锥桶边界清晰后继续前进。", {"WAIT", "ADJUST_SPEED", "PROCEED"}),
    ("前方施工车辆占道，左侧安全时绕行，否则停车等待。", {"AVOID"}),
    ("前方施工车辆占道，确认左侧安全后减速绕行，超过后回到安全车道。", {"ADJUST_SPEED", "AVOID", "CHANGE_LANE"}),
    ("驶离施工路段后，保持安全速度行驶至终点。", {"ADJUST_SPEED", "NAVIGATE_TO"}),
    ("雨雾还没有完全消散，保持当前车道，车速不超过45公里每小时。", {"KEEP_LANE", "SET_SPEED"}),
    ("终点前保持安全车速和当前车道，确认前方通畅后继续行驶。", {"ADJUST_SPEED", "KEEP_LANE", "PROCEED"}),
)


class SceneReportRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = RuleIntentParser()

    def parse(self, text: str) -> dict:
        result = self.parser.parse(text, modality="VOICE", request_id="scene-report")
        self.assertIsNotNone(result)
        validate_document(result)
        self.assertEqual(result["parse_result"]["status"], "VALID")
        self.assertLess(result["parse_result"]["latency_ms"], 50.0)
        return result

    def test_scene_2_action_sequences(self) -> None:
        for text, expected in SCENE_2_CASES:
            with self.subTest(text=text):
                result = self.parse(text)
                actions = [step["action"] for step in result["intent"]["steps"]]
                self.assertEqual(actions, expected)

    def test_scene_3_required_actions(self) -> None:
        for text, required in SCENE_3_CASES:
            with self.subTest(text=text):
                result = self.parse(text)
                actions = {step["action"] for step in result["intent"]["steps"]}
                self.assertTrue(required <= actions, (required, actions))
                self.assertNotIn("OVERTAKE", actions if "不超过" in text else set())

    def test_explicit_speed_caps_are_preserved(self) -> None:
        cases = (
            ("能见度差，保持右侧车道，车速不高于40公里每小时。", 40.0),
            ("雨雾还没有完全消散，保持当前车道，车速不超过45公里每小时。", 45.0),
        )
        for text, expected_kmh in cases:
            with self.subTest(text=text):
                result = self.parse(text)
                speed_steps = [
                    step
                    for step in result["intent"]["steps"]
                    if step["action"] == "SET_SPEED"
                ]
                self.assertEqual(len(speed_steps), 1)
                self.assertAlmostEqual(
                    speed_steps[0]["parameters"]["target_speed_mps"],
                    expected_kmh / 3.6,
                    places=3,
                )
                self.assertAlmostEqual(
                    result["intent"]["constraints"]["max_speed_mps"],
                    expected_kmh / 3.6,
                    places=3,
                )

    def test_entities_and_safe_distance_are_groundable(self) -> None:
        result = self.parse(
            "通过路口后向右变道，跟随前方白色车辆，并保持安全车距。"
        )
        entities = result["intent"]["entities"]
        self.assertTrue(any(item["type"] == "JUNCTION" for item in entities))
        white_vehicle = next(
            item
            for item in entities
            if item["type"] == "VEHICLE"
            and item["canonical_attributes"].get("color") == "WHITE"
        )
        follow = next(
            step for step in result["intent"]["steps"] if step["action"] == "FOLLOW"
        )
        self.assertEqual(follow["target_ref"], white_vehicle["entity_id"])
        self.assertEqual(follow["goal_conditions"][0]["predicate"], "SAFE_DISTANCE")

    def test_work_zone_entities_and_lateral_direction_are_preserved(self) -> None:
        result = self.parse(
            "前方施工车辆占道，确认左侧安全后减速绕行，超过后回到安全车道。"
        )
        descriptors = {
            descriptor
            for entity in result["intent"]["entities"]
            for descriptor in entity["open_descriptors"]
        }
        self.assertIn("work_vehicle", descriptors)
        self.assertIn("construction_zone", descriptors)
        directions = [
            step["parameters"].get("direction")
            for step in result["intent"]["steps"]
            if step["action"] in {"AVOID", "CHANGE_LANE"}
        ]
        self.assertIn("LEFT", directions)
        self.assertIn("RIGHT", directions)

    def test_completed_actions_are_not_reissued(self) -> None:
        result = self.parse(
            "确认已经超过红色慢车，安全返回右侧车道，恢复至45公里每小时并保持车距。"
        )
        actions = [step["action"] for step in result["intent"]["steps"]]
        self.assertNotIn("OVERTAKE", actions)


if __name__ == "__main__":
    unittest.main()
