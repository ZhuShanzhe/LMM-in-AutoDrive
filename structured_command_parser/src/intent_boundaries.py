from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class BrakingBoundary:
    action: str
    urgency: str
    evidence: str


_EN_EMERGENCY = re.compile(
    r"\b(?:emergency (?:brake|braking|stop)|slam(?:med|ming)? (?:on )?(?:the )?brakes?|"
    r"hard (?:brake|braking)|brake hard|full[- ]force brak(?:e|ing)|"
    r"maximum brak(?:e|ing))\b",
    re.IGNORECASE,
)
_EN_DANGER = re.compile(
    r"\b(?:emergency|imminent (?:collision|crash)|about to (?:collide|crash)|"
    r"collision risk|danger|hazard|sudden(?:ly)? .{0,30}(?:pedestrian|child|obstacle)|"
    r"(?:pedestrian|child|obstacle) .{0,30}sudden(?:ly)?)\b",
    re.IGNORECASE,
)
_EN_URGENT = re.compile(
    r"\b(?:immediate|immediately|instantly|now|without delay|at once|right away|urgently)\b",
    re.IGNORECASE,
)
_EN_STOP = re.compile(
    r"\b(?:stop|halt|standstill|cease all movement)\b", re.IGNORECASE
)
_EN_BRAKE = re.compile(
    r"\b(?:brake|brakes|braking|hit (?:the )?brakes?|slow down|decelerate|reduce (?:the )?speed)\b",
    re.IGNORECASE,
)

_ZH_EMERGENCY = re.compile(
    r"紧急(?:制动|刹车|停车)|紧急情况|急刹(?:车)?|猛(?:踩|踏)(?:制动|刹车)|"
    r"踩死刹车|全力(?:制动|刹车)|最大(?:力度|制动力)(?:制动|刹车)?"
)
_ZH_DANGER = re.compile(
    r"紧急|突然(?:出现|冲出|横穿)|突发|危险|即将(?:碰撞|撞上)|碰撞风险|"
    r"马上要撞|快要撞|行人.{0,12}冲出|儿童.{0,12}冲出"
)
_ZH_URGENT = re.compile(r"立即|立刻|马上|现在就|赶紧|尽快")
_ZH_STOP = re.compile(r"停车(?!场)|停下|停住|停止车辆|刹停")
_ZH_BRAKE = re.compile(r"刹车|制动|踩(?:下)?刹车|减速|降速|降低(?:车速|速度)|慢一点|慢一些")


def classify_english_braking(text: str) -> BrakingBoundary | None:
    explicit_emergency = bool(_EN_EMERGENCY.search(text))
    danger = bool(_EN_DANGER.search(text))
    urgent = bool(_EN_URGENT.search(text))
    stop = bool(_EN_STOP.search(text))
    brake = bool(_EN_BRAKE.search(text))

    if explicit_emergency or (danger and urgent and (stop or brake)):
        return BrakingBoundary("EMERGENCY_BRAKE", "EMERGENCY", "EXPLICIT_EMERGENCY_OR_DANGER")
    if urgent and stop:
        return BrakingBoundary("STOP", "URGENT", "URGENT_STOP_WITHOUT_DANGER")
    if urgent and brake:
        return BrakingBoundary("ADJUST_SPEED", "URGENT", "URGENT_BRAKE_WITHOUT_STOP")
    if stop:
        return BrakingBoundary("STOP", "NORMAL", "ORDINARY_STOP")
    if brake:
        return BrakingBoundary("ADJUST_SPEED", "NORMAL", "ORDINARY_BRAKE")
    return None


def classify_chinese_braking(text: str) -> BrakingBoundary | None:
    explicit_emergency = bool(_ZH_EMERGENCY.search(text))
    danger = bool(_ZH_DANGER.search(text))
    urgent = bool(_ZH_URGENT.search(text))
    stop = bool(_ZH_STOP.search(text))
    brake = bool(_ZH_BRAKE.search(text))

    if explicit_emergency or (danger and urgent and (stop or brake)):
        return BrakingBoundary("EMERGENCY_BRAKE", "EMERGENCY", "EXPLICIT_EMERGENCY_OR_DANGER")
    if urgent and stop:
        return BrakingBoundary("STOP", "URGENT", "URGENT_STOP_WITHOUT_DANGER")
    if urgent and brake:
        return BrakingBoundary("ADJUST_SPEED", "URGENT", "URGENT_BRAKE_WITHOUT_STOP")
    if stop:
        return BrakingBoundary("STOP", "NORMAL", "ORDINARY_STOP")
    if brake:
        return BrakingBoundary("ADJUST_SPEED", "NORMAL", "ORDINARY_BRAKE")
    return None
