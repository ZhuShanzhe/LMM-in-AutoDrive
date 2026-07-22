from __future__ import annotations

ACTION_LABELS = (
    "KEEP_LANE",
    "SET_SPEED",
    "ADJUST_SPEED",
    "STOP",
    "WAIT",
    "FOLLOW",
    "APPROACH",
    "NAVIGATE_TO",
    "CHANGE_LANE",
    "MERGE",
    "TURN",
    "U_TURN",
    "PROCEED",
    "YIELD",
    "PULL_OVER",
    "PARK",
    "OVERTAKE",
    "PASS_BY",
    "AVOID",
    "REVERSE",
    "ENTER_AREA",
    "EXIT_AREA",
    "EMERGENCY_BRAKE",
    "RESUME",
    "CANCEL",
)

STATUS_LABELS = ("VALID", "NEEDS_CLARIFICATION", "UNSUPPORTED")
CATEGORY_LABELS = (
    "BASIC_CONTROL",
    "NAVIGATION",
    "COMPLEX_OBSTACLE_AVOIDANCE",
    "EMERGENCY_RESPONSE",
    "META_CONTROL",
)
URGENCY_LABELS = ("NORMAL", "URGENT", "EMERGENCY")
DIRECTION_LABELS = ("LEFT", "RIGHT", "STRAIGHT", "BACKWARD")
CHANGE_LABELS = ("NONE", "INCREASE", "DECREASE")


def label_schema() -> dict[str, list[str]]:
    return {
        "actions": list(ACTION_LABELS),
        "statuses": list(STATUS_LABELS),
        "categories": list(CATEGORY_LABELS),
        "urgencies": list(URGENCY_LABELS),
        "directions": list(DIRECTION_LABELS),
        "changes": list(CHANGE_LABELS),
    }
