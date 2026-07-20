from __future__ import annotations

import unittest

from structured_command_parser.scripts.build_diverse_chinese_commands import make_cases


ALLOWED_ACTIONS = {
    "KEEP_LANE",
    "SET_SPEED",
    "ADJUST_SPEED",
    "STOP",
    "CHANGE_LANE",
    "TURN",
    "YIELD",
    "PULL_OVER",
    "OVERTAKE",
    "AVOID",
    "EMERGENCY_BRAKE",
    "RESUME",
    "CANCEL",
}


class DiverseDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dev, self.holdout = make_cases()

    def test_split_sizes_and_texts_are_disjoint(self) -> None:
        self.assertEqual(len(self.dev), 80)
        self.assertEqual(len(self.holdout), 40)
        dev_texts = {row["text"] for row in self.dev}
        holdout_texts = {row["text"] for row in self.holdout}
        self.assertEqual(len(dev_texts), 80)
        self.assertEqual(len(holdout_texts), 40)
        self.assertTrue(dev_texts.isdisjoint(holdout_texts))

    def test_labels_use_supported_contract_values(self) -> None:
        for row in self.dev + self.holdout:
            label = row["expected"]
            self.assertIn(label["status"], {"VALID", "NEEDS_CLARIFICATION", "UNSUPPORTED"})
            actions = label.get("actions", label.get("actions_unordered", []))
            self.assertTrue(set(actions).issubset(ALLOWED_ACTIONS), row["sample_id"])
            if label["status"] != "VALID":
                self.assertEqual(actions, [], row["sample_id"])
            if "target_speed_mps" in label:
                self.assertEqual(len(label["target_speed_mps"]), 1, row["sample_id"])
                self.assertGreater(label["target_speed_mps"][0], 0)

    def test_provenance_is_explicit(self) -> None:
        for row in self.dev + self.holdout:
            metadata = row["metadata"]
            self.assertEqual(metadata["origin"], "CURATED_SYNTHETIC")
            self.assertEqual(metadata["review_status"], "REQUIRES_HUMAN_REVIEW")


if __name__ == "__main__":
    unittest.main()
