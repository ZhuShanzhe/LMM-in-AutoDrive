from __future__ import annotations

import re
from typing import Any


_ACTION_PATTERNS = (
    (
        "EMERGENCY_BRAKE",
        re.compile(
            r"\b(?:emergency brake|slam (?:on )?the brakes?|brake hard|"
            r"hard braking|full braking)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "U_TURN",
        re.compile(r"\b(?:make (?:a )?u[- ]?turn|u[- ]?turn|turn around)\b", re.IGNORECASE),
    ),
    (
        "CHANGE_LANE",
        re.compile(
            r"\b(?:change|switch|shift|move|transition)(?: over)? "
            r"(?:to|into) (?:the )?(?:left|right)(?:-hand)? lane\b|"
            r"\b(?:change|switch|shift) lanes? (?:to )?(?:the )?"
            r"(?:left|right)\b|"
            r"\b(?:take|use) (?:the )?(?:left|right)(?:-hand)? lane\b",
            re.IGNORECASE,
        ),
    ),
    (
        "MERGE",
        re.compile(
            r"\bmerge (?:to|into|onto) (?:the )?(?:left|right)(?:-hand)? "
            r"(?:lane|traffic)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "OVERTAKE",
        re.compile(r"\b(?:overtake|get past|pass) (?!(?:by|through)\b)", re.IGNORECASE),
    ),
    (
        "RESUME",
        re.compile(
            r"\b(?:resume|return|get(?:ting)? back|move back)\s+"
            r"(?:(?:into|to)\s+)?(?:the\s+)?"
            r"(?:original|previous)\s+(?:lane|route|course)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "PULL_OVER",
        re.compile(
            r"\b(?:pull over|pull up)(?: (?:to|on) (?:the )?"
            r"(?:left|right)(?:-hand)?(?: side)?)?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "ADJUST_SPEED",
        re.compile(
            r"\b(?:accelerate|speed up|slow down|decelerate|reduce speed|brake)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "SET_SPEED",
        re.compile(
            r"\b(?:set|maintain|hold|drive at)(?: the)? speed(?: at| to| of)? "
            r"\d+(?:\.\d+)?\s*(?:km/h|m/s)\b|"
            r"\b(?:set|maintain|hold|drive at) \d+(?:\.\d+)?\s*(?:km/h|m/s)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "STOP",
        re.compile(
            r"(?<!bus )(?<!traffic )\bstop\b|"
            r"\b(?:halt|come to a standstill|cease all movement)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "WAIT",
        re.compile(r"\b(?:wait|hold position|stay put)\b", re.IGNORECASE),
    ),
    (
        "FOLLOW",
        re.compile(
            r"\b(?:follow|trail|stay behind|catch up with|keep up with)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "APPROACH",
        re.compile(r"\b(?:approach|move closer|drive closer|get closer)\b", re.IGNORECASE),
    ),
    (
        "NAVIGATE_TO",
        re.compile(r"\b(?:go to|drive to|head to|navigate to|take me to)\b", re.IGNORECASE),
    ),
    (
        "TURN",
        re.compile(
            r"\b(?:turn (?:to )?(?:left|right)|make (?:a )?(?:left|right) turn)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "PROCEED",
        re.compile(
            r"\b(?:continue|proceed|go|drive|carry on) "
            r"(?:straight|forward|ahead|through)\b|"
            r"\bkeep going straight\b",
            re.IGNORECASE,
        ),
    ),
    (
        "YIELD",
        re.compile(r"\b(?:yield|give way|let .+? pass|allow .+? to pass)\b", re.IGNORECASE),
    ),
    ("PARK", re.compile(r"\bpark(?:ing)?\b", re.IGNORECASE)),
    (
        "PASS_BY",
        re.compile(r"\b(?:drive past|go past|pass by)\b", re.IGNORECASE),
    ),
    (
        "AVOID",
        re.compile(
            r"\b(?:avoid|go around|drive around|steer clear|maneuver around)\b",
            re.IGNORECASE,
        ),
    ),
    ("REVERSE", re.compile(r"\b(?:reverse|back up)\b", re.IGNORECASE)),
    ("ENTER_AREA", re.compile(r"\benter\b", re.IGNORECASE)),
    ("EXIT_AREA", re.compile(r"\b(?:exit|leave)\b", re.IGNORECASE)),
    (
        "KEEP_LANE",
        re.compile(
            r"\b(?:keep|maintain|stay|remain) (?:in )?(?:the )?"
            r"(?:current )?lane\b",
            re.IGNORECASE,
        ),
    ),
    (
        "CANCEL",
        re.compile(
            r"\b(?:cancel|abort|revoke|never mind|changed my mind|"
            r"ignore what i just asked)\b",
            re.IGNORECASE,
        ),
    ),
)


def _direction(surface: str) -> str | None:
    if re.search(r"\bleft(?:-hand)?\b", surface, re.IGNORECASE):
        return "LEFT"
    if re.search(r"\bright(?:-hand)?\b", surface, re.IGNORECASE):
        return "RIGHT"
    if re.search(r"\b(?:straight|forward|ahead|through)\b", surface, re.IGNORECASE):
        return "STRAIGHT"
    return None


def _command(action: str, surface: str, text: str) -> dict[str, Any]:
    command: dict[str, Any] = {"action": action}
    if action in {"CHANGE_LANE", "MERGE", "TURN", "PROCEED", "PULL_OVER"}:
        direction = _direction(surface)
        if direction is None and action == "PULL_OVER":
            suffix = text[text.casefold().find(surface.casefold()) :][:48]
            direction = _direction(suffix)
        if direction is not None:
            command["direction"] = direction
    if action == "ADJUST_SPEED":
        command["change"] = (
            "INCREASE"
            if re.search(r"\b(?:accelerate|speed up)\b", surface, re.IGNORECASE)
            else "DECREASE"
        )
    if action == "SET_SPEED":
        speed = re.search(
            r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>km/h|m/s)",
            surface,
            re.IGNORECASE,
        )
        if speed:
            value = float(speed.group("value"))
            unit = speed.group("unit").casefold()
            command["target_speed_mps"] = round(value / 3.6 if unit == "km/h" else value, 3)
            command["source_value"] = value
            command["source_unit"] = unit
    return command


def decompose_atomic_actions(text: str) -> list[dict[str, Any]]:
    """Recover explicit atomic actions in surface order.

    The result is deliberately high precision. If no lexical action is found,
    the caller can fall back to the learned multi-label heads.
    """

    matches: list[tuple[int, int, int, str, str]] = []
    for priority, (action, pattern) in enumerate(_ACTION_PATTERNS):
        for match in pattern.finditer(text):
            matches.append(
                (match.start(), match.end(), priority, action, match.group(0))
            )
    matches.sort(key=lambda item: (item[0], item[2], -(item[1] - item[0])))

    commands: list[dict[str, Any]] = []
    accepted_spans: list[tuple[int, int]] = []
    for start, end, _, action, surface in matches:
        if any(start < other_end and end > other_start for other_start, other_end in accepted_spans):
            continue
        commands.append(_command(action, surface, text))
        accepted_spans.append((start, end))
    return commands
