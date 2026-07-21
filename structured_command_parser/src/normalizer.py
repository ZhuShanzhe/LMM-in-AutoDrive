from __future__ import annotations

import re
import unicodedata


_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_SMALL_UNITS = {"十": 10, "百": 100, "千": 1000}
_NUMBER_BEFORE_UNIT = re.compile(
    r"[零〇一二两三四五六七八九十百千]+(?=\s*(?:公里每小时|千米每小时|公里/小时|千米/小时|km/h|m/s|公里|千米|米|秒))",
    re.IGNORECASE,
)


def chinese_integer_to_int(value: str) -> int:
    if not value:
        raise ValueError("Chinese number cannot be empty")
    if all(character in _DIGITS for character in value):
        return int("".join(str(_DIGITS[character]) for character in value))

    total = 0
    section = 0
    number = 0
    for character in value:
        if character in _DIGITS:
            number = _DIGITS[character]
            continue
        unit = _SMALL_UNITS.get(character)
        if unit is None:
            raise ValueError(f"Unsupported Chinese number character: {character}")
        if number == 0:
            number = 1
        section += number * unit
        number = 0
    return total + section + number


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).strip()
    normalized = _NUMBER_BEFORE_UNIT.sub(
        lambda match: str(chinese_integer_to_int(match.group(0))), normalized
    )
    normalized = re.sub(
        r"每秒\s*(?P<value>\d+(?:\.\d+)?)\s*米",
        r"\g<value> m/s",
        normalized,
    )
    normalized = re.sub(
        r"每小时\s*(?P<value>\d+(?:\.\d+)?)\s*(?:千米|公里)",
        r"\g<value> km/h",
        normalized,
    )
    normalized = re.sub(
        r"千米每小时|公里每小时|千米/小时|公里/小时",
        "km/h",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"米每秒|米/秒", "m/s", normalized, flags=re.IGNORECASE)
    normalized = re.sub(
        r"时速\s*(?P<value>\d+(?:\.\d+)?)\s*(?:千米|公里)",
        r"速度为 \g<value> km/h",
        normalized,
    )
    normalized = re.sub(r"(?<=\d)\s*(?:千米|公里)(?!\s*/)", " km", normalized)
    normalized = re.sub(r"(?<=\d)\s*米", " m", normalized)
    normalized = re.sub(r"(?<=\d)\s*秒", " s", normalized)
    normalized = re.sub(r"(?<=\d)\s*(km/h|m/s|km|m|s)\b", r" \1", normalized)
    normalized = re.sub(r"[，、；;]+", "，", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()
