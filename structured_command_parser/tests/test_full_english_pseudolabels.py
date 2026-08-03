from __future__ import annotations

import unittest

from structured_command_parser.scripts.build_full_english_pseudolabels import (
    assign_splits,
    pseudo_label,
    sparse_augmentation_rows,
)


class FullEnglishPseudoLabelTests(unittest.TestCase):
    def test_braking_boundary_is_preserved(self) -> None:
        cases = (
            ("Emergency brake now.", "EMERGENCY_BRAKE", "EMERGENCY"),
            ("Stop immediately.", "STOP", "URGENT"),
            ("Stop the vehicle.", "STOP", "NORMAL"),
            ("Brake gently.", "ADJUST_SPEED", "NORMAL"),
        )
        for index, (text, action, urgency) in enumerate(cases):
            with self.subTest(text=text):
                row = pseudo_label(
                    {
                        "sample_id": f"test-{index}",
                        "source": "Talk2Car",
                        "source_split": "test",
                        "text_en": text,
                        "metadata": {},
                    }
                )
                self.assertEqual(row["expected"]["actions"], [action])
                self.assertEqual(row["expected"]["urgency"], urgency)

    def test_identical_normalized_text_stays_in_one_split(self) -> None:
        rows = [
            pseudo_label(
                {
                    "sample_id": f"same-{index}",
                    "source": "Talk2Car",
                    "source_split": "train",
                    "text_en": text,
                    "metadata": {},
                }
            )
            for index, text in enumerate(("Stop now", "  stop   now ", "STOP NOW"))
        ]
        rows.extend(
            pseudo_label(
                {
                    "sample_id": f"other-{index}",
                    "source": "Talk2Car",
                    "source_split": "train",
                    "text_en": f"Turn left after {index + 1} meters",
                    "metadata": {},
                }
            )
            for index in range(20)
        )
        assign_splits(rows)
        self.assertEqual(len({row["split"] for row in rows if row["sample_id"].startswith("same-")}), 1)

    def test_sparse_augmentation_covers_missing_actions(self) -> None:
        rows = sparse_augmentation_rows()
        actions = {action for row in rows for action in row["expected"]["actions"]}
        self.assertIn("EMERGENCY_BRAKE", actions)
        self.assertIn("CANCEL", actions)
        self.assertTrue(all(row["split"] == "train" for row in rows))


if __name__ == "__main__":
    unittest.main()
