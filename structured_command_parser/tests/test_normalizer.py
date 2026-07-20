from __future__ import annotations

import unittest

from structured_command_parser.src.normalizer import (
    chinese_integer_to_int,
    normalize_text,
)


class NormalizerTest(unittest.TestCase):
    def test_chinese_numbers(self) -> None:
        self.assertEqual(chinese_integer_to_int("六十"), 60)
        self.assertEqual(chinese_integer_to_int("三百"), 300)
        self.assertEqual(chinese_integer_to_int("一百零五"), 105)

    def test_units(self) -> None:
        self.assertEqual(normalize_text("提速至六十公里每小时"), "提速至60 km/h")
        self.assertEqual(normalize_text("前方三百米路口右转"), "前方300 m路口右转")

    def test_additional_speed_units(self) -> None:
        self.assertEqual(normalize_text("保持速度在17米/秒"), "保持速度在17 m/s")
        self.assertEqual(
            normalize_text("保持速度在每秒13.5米"), "保持速度在13.5 m/s"
        )
        self.assertEqual(
            normalize_text("以每小时44.6公里的速度行驶"),
            "以44.6 km/h的速度行驶",
        )
        self.assertEqual(
            normalize_text("保持时速44.6公里"), "保持速度为 44.6 km/h"
        )


if __name__ == "__main__":
    unittest.main()
