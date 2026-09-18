import re
from typing import Any, Dict, List

from .text_metrics import strip_punctuation

DIRECTION_PATTERNS: Dict[str, List[str]] = {
    "left": [r"左转", r"向左", r"左侧", r"左拐", r"\bleft\b"],
    "right": [r"右转", r"向右", r"右侧", r"右拐", r"\bright\b"],
    "straight": [r"直行", r"向前", r"\bstraight\b", r"\bforward\b"],
    "uturn": [r"掉头", r"调头", r"u[- ]?turn"],
    "lane_change": [r"变道", r"并线", r"换道", r"lane change"],
}

ACTION_PATTERNS: Dict[str, List[str]] = {
    "stop": [r"停车", r"停下", r"刹停", r"\bstop\b"],
    "park": [r"靠边停车", r"靠边", r"泊车", r"\bpark\b"],
    "brake": [r"刹车", r"制动", r"\bbrake\b"],
    "decelerate": [r"减速", r"降速", r"慢行", r"\bdecelerate\b"],
    "accelerate": [r"加速", r"提速", r"\baccelerate\b"],
    "follow": [r"跟随", r"跟着", r"\bfollow\b"],
    "yield": [r"让行", r"礼让", r"\byield\b"],
    "overtake": [r"超车", r"\bovertake\b"],
}

NEGATION_PATTERNS: List[str] = [
    r"不", r"别", r"无需", r"不用", r"禁止", r"取消",
    r"\bno\b", r"\bnot\b", r"\bdon't\b",
]

_NUMBER_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*"
    r"(公里每小时|千米每小时|km/h|kmh|迈|公里|千米|米|秒|号|车道)?",
    re.IGNORECASE,
)

_UNIT_CANON = {
    "公里每小时": "km/h", "千米每小时": "km/h", "km/h": "km/h", "kmh": "km/h",
    "迈": "km/h", "公里": "km", "千米": "km", "米": "m", "秒": "s",
    "号": "index", "车道": "lane",
}

def _match(text: str, patterns: Dict[str, List[str]]) -> List[str]:
    return [label for label, pats in patterns.items() if any(re.search(p, text, re.IGNORECASE) for p in pats)]

def extract_quantities(text: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for match in _NUMBER_RE.finditer(strip_punctuation(text)):
        value, unit = match.group(1), (match.group(2) or "").strip()
        out.append({"value": float(value), "unit": _UNIT_CANON.get(unit, unit)})
    return out

def extract_slots(text: str) -> Dict[str, Any]:
    text = strip_punctuation(text)
    if not text:
        return {"direction": [], "action": [], "negation": False, "quantities": []}
    return {
        "direction": _match(text, DIRECTION_PATTERNS),
        "action": _match(text, ACTION_PATTERNS),
        "negation": any(re.search(p, text, re.IGNORECASE) for p in NEGATION_PATTERNS),
        "quantities": extract_quantities(text),
    }

def slot_preservation(ref_slots: Dict[str, Any], hyp_slots: Dict[str, Any]) -> Dict[str, float]:
    dir_ok = float(set(ref_slots["direction"]) == set(hyp_slots["direction"]))
    act_ok = float(bool(set(ref_slots["action"]) & set(hyp_slots["action"])))
    neg_ok = float(ref_slots["negation"] == hyp_slots["negation"])
    ref_q = {(q["value"], q["unit"]) for q in ref_slots["quantities"]}
    hyp_q = {(q["value"], q["unit"]) for q in hyp_slots["quantities"]}
    qty_ok = float(len(ref_q & hyp_q) / len(ref_q)) if ref_q else 1.0
    return {
        "direction_match": dir_ok,
        "action_match": act_ok,
        "negation_match": neg_ok,
        "quantity_recall": round(qty_ok, 4),
        "critical_score": round((dir_ok + neg_ok + qty_ok) / 3.0, 4),
    }

__all__ = [
    "DIRECTION_PATTERNS",
    "ACTION_PATTERNS",
    "NEGATION_PATTERNS",
    "extract_quantities",
    "extract_slots",
    "slot_preservation",
]
