from __future__ import annotations

import copy
import re
from typing import Any, Sequence

from .normalizer import normalize_text


TARGET_SPEED_PATTERN = re.compile(
    r"(?:"
    r"(?:提速|加速|减速|降速)\s*(?:至|到|为|在)\s*|"
    r"(?:车速|速度|巡航速度)\s*"
    r"(?:控制|调整|调|设置|设定|设|降低|降|保持|维持)?\s*"
    r"(?:至|到|为|在)?\s*|"
    r"(?:保持车速|保持速度)\s*(?:至|到|为|在)?\s*|"
    r"(?:保持|维持|按|以)\s*|"
    r"(?:尝试)?行驶至\s*|"
    r"时速\s*(?:保持|维持)?\s*(?:至|到|为|在)?\s*"
    r")"
    r"(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>"
    r"km/h|m/s|"
    r"km(?=\s*(?:的)?(?:速度|车速|行驶|巡航|$))"
    r")",
    re.IGNORECASE,
)


def canonical_speed_unit(unit: str) -> str:
    return "km/h" if unit.casefold() in {"km", "km/h"} else "m/s"


def speed_to_mps(value: float, unit: str) -> float:
    return value / 3.6 if canonical_speed_unit(unit) == "km/h" else value


def extract_source_target_speeds(text: str) -> list[dict[str, Any]]:
    normalized = normalize_text(text)
    slots: list[dict[str, Any]] = []
    for match in TARGET_SPEED_PATTERN.finditer(normalized):
        value = float(match.group("value"))
        source_unit = canonical_speed_unit(match.group("unit"))
        slots.append(
            {
                "target_speed_mps": round(speed_to_mps(value, source_unit), 3),
                "source_value": value,
                "source_unit": source_unit,
                "source_span": match.group(0),
            }
        )
    return slots


def restore_source_target_speeds(
    commands: Sequence[dict[str, Any]],
    source_text: str,
) -> tuple[list[dict[str, Any]], bool]:
    """Make explicit source-language speed values authoritative."""

    slots = extract_source_target_speeds(source_text)
    result = [copy.deepcopy(command) for command in commands]
    if not slots:
        return result, False

    set_speed_indices = [
        index
        for index, command in enumerate(result)
        if command.get("action") == "SET_SPEED"
    ]
    adjust_speed_indices = [
        index
        for index, command in enumerate(result)
        if command.get("action") == "ADJUST_SPEED"
    ]
    changed = False
    for slot_index, slot in enumerate(slots):
        if slot_index < len(set_speed_indices):
            command = result[set_speed_indices[slot_index]]
        elif adjust_speed_indices:
            command = result[adjust_speed_indices.pop(0)]
            command["action"] = "SET_SPEED"
            command.pop("change", None)
            command.pop("speed_delta_mps", None)
            changed = True
        else:
            command = {"action": "SET_SPEED"}
            result.append(command)
            changed = True
        for key in ("target_speed_mps", "source_value", "source_unit"):
            if command.get(key) != slot[key]:
                changed = True
            command[key] = slot[key]
    return result, changed
