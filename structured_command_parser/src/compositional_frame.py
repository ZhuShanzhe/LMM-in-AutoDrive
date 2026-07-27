from __future__ import annotations

import re
from typing import Any, Iterable, Sequence


SEMANTIC_TAG_LABELS = (
    "O",
    "B_ENTITY",
    "I_ENTITY",
    "B_RELATION",
    "I_RELATION",
)

COLORS = {
    "black": "BLACK",
    "blue": "BLUE",
    "brown": "BROWN",
    "gray": "GRAY",
    "grey": "GRAY",
    "green": "GREEN",
    "orange": "ORANGE",
    "red": "RED",
    "silver": "SILVER",
    "white": "WHITE",
    "yellow": "YELLOW",
}

VEHICLE_SUBTYPES = {
    "bus": "BUS",
    "car": "CAR",
    "hatchback": "HATCHBACK",
    "lorry": "TRUCK",
    "minivan": "VAN",
    "pickup": "PICKUP",
    "sedan": "SEDAN",
    "suv": "SUV",
    "taxi": "TAXI",
    "truck": "TRUCK",
    "van": "VAN",
    "vehicle": "UNSPECIFIED",
}

ENTITY_HEADS = {
    **{name: "VEHICLE" for name in VEHICLE_SUBTYPES},
    "bicycle": "CYCLIST",
    "bike": "CYCLIST",
    "cyclist": "CYCLIST",
    "pedestrian": "PEDESTRIAN",
    "person": "PEDESTRIAN",
    "walker": "PEDESTRIAN",
    "cone": "TRAFFIC_CONE",
    "cones": "TRAFFIC_CONE",
    "barrier": "OBSTACLE",
    "obstacle": "OBSTACLE",
    "traffic light": "TRAFFIC_LIGHT",
    "light": "TRAFFIC_LIGHT",
    "traffic sign": "TRAFFIC_SIGN",
    "sign": "TRAFFIC_SIGN",
    "crosswalk": "CROSSWALK",
    "junction": "JUNCTION",
    "intersection": "JUNCTION",
    "parking space": "PARKING_SPACE",
    "parking lot": "PARKING_AREA",
    "bus stop": "LANDMARK",
    "curb": "CURB",
    "one": "UNKNOWN",
}

ORDINALS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
}

RELATION_PHRASES = (
    ("in front of", "IN_FRONT_OF"),
    ("next to", "NEXT_TO"),
    ("to the left of", "LEFT_OF"),
    ("to the right of", "RIGHT_OF"),
    ("after", "AFTER"),
    ("before", "BEFORE"),
    ("behind", "BEHIND"),
    ("between", "BETWEEN"),
    ("near", "NEAR"),
    ("past", "PAST"),
    ("inside", "INSIDE"),
    ("until", "UNTIL"),
    ("when you see", "VISIBLE"),
    ("once you see", "VISIBLE"),
    ("after you spot", "VISIBLE"),
    ("comes into view", "VISIBLE"),
    ("upon seeing", "VISIBLE"),
    ("once you have passed", "AFTER"),
    ("when visible", "VISIBLE"),
)

LOCATOR_RELATIONS = (
    (r"\b(?:ahead|in front|up ahead)\b", "AHEAD"),
    (r"\b(?:behind|at the rear)\b", "BEHIND"),
    (r"\b(?:front left|ahead on the left)\b", "FRONT_LEFT"),
    (r"\b(?:front right|ahead on the right)\b", "FRONT_RIGHT"),
    (r"\b(?:on|to) the left\b|\bleft-side\b", "LEFT"),
    (r"\b(?:on|to) the right\b|\bright-side\b", "RIGHT"),
)

_MODIFIER = (
    r"(?:the|a|an|this|that|first|second|third|fourth|fifth|"
    r"black|blue|brown|gray|grey|green|orange|red|silver|white|yellow|"
    r"large|small|slow|moving|parked|stopped|nearby|ahead|front|rear|left|right)"
)
_HEAD_PATTERN = "|".join(
    re.escape(head) for head in sorted(ENTITY_HEADS, key=len, reverse=True)
)
ENTITY_PATTERN = re.compile(
    rf"\b(?:{_MODIFIER}\s+){{0,6}}(?:{_HEAD_PATTERN})\b",
    re.IGNORECASE,
)


def decode_token_spans(
    text: str,
    offsets: Sequence[Sequence[int]],
    tag_probabilities: Any,
    *,
    min_confidence: float = 0.55,
) -> list[dict[str, Any]]:
    """Decode BIO token predictions into auditable character spans."""

    labels = tag_probabilities.argmax(dim=-1).tolist()
    confidence = tag_probabilities.max(dim=-1).values.tolist()
    spans: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    for label_index, score, offset in zip(labels, confidence, offsets):
        start, end = int(offset[0]), int(offset[1])
        label = SEMANTIC_TAG_LABELS[int(label_index)]
        if start == end or score < min_confidence or label == "O":
            if active is not None:
                spans.append(active)
                active = None
            continue
        prefix, role = label.split("_", 1)
        if prefix == "B" or active is None or active["role"] != role:
            if active is not None:
                spans.append(active)
            active = {
                "role": role,
                "start": start,
                "end": end,
                "confidence_values": [float(score)],
            }
        else:
            active["end"] = end
            active["confidence_values"].append(float(score))
    if active is not None:
        spans.append(active)

    decoded = []
    for span in spans:
        value = text[span["start"] : span["end"]].strip(" ,.;:")
        if not value:
            continue
        decoded.append(
            {
                "role": span["role"],
                "text": value,
                "start": span["start"],
                "end": span["end"],
                "confidence": round(
                    sum(span["confidence_values"])
                    / len(span["confidence_values"]),
                    4,
                ),
            }
        )
    return decoded


def _fallback_entity_spans(text: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "ENTITY",
            "text": match.group(0),
            "start": match.start(),
            "end": match.end(),
            "confidence": 1.0,
        }
        for match in ENTITY_PATTERN.finditer(text)
    ]


def _fallback_relation_spans(text: str) -> list[dict[str, Any]]:
    spans = []
    for phrase, _ in RELATION_PHRASES:
        for match in re.finditer(rf"\b{re.escape(phrase)}\b", text, re.IGNORECASE):
            if phrase == "past" and re.search(
                r"\b(?:get|drive|go|pass)\s+$",
                text[max(0, match.start() - 12) : match.start()],
                re.IGNORECASE,
            ):
                continue
            spans.append(
                {
                    "role": "RELATION",
                    "text": match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 1.0,
                }
            )
    return spans


def _deduplicate_spans(spans: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, int, int], dict[str, Any]] = {}
    for span in spans:
        key = (str(span["role"]), int(span["start"]), int(span["end"]))
        if key not in best or float(span.get("confidence", 0)) > float(
            best[key].get("confidence", 0)
        ):
            best[key] = dict(span)
    return sorted(best.values(), key=lambda item: (item["start"], item["end"]))


def _merge_non_overlapping_spans(
    spans: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    ranked = sorted(
        _deduplicate_spans(spans),
        key=lambda item: (
            -float(item.get("confidence", 0)),
            -(int(item["end"]) - int(item["start"])),
            int(item["start"]),
        ),
    )
    selected: list[dict[str, Any]] = []
    for span in ranked:
        start, end = int(span["start"]), int(span["end"])
        if any(
            start < int(other["end"]) and end > int(other["start"])
            for other in selected
        ):
            continue
        selected.append(span)
    return sorted(selected, key=lambda item: (item["start"], item["end"]))


def _valid_relation_span(text: str, span: dict[str, Any]) -> bool:
    past = re.search(r"\bpast\b", str(span["text"]), re.IGNORECASE)
    if past is None:
        return True
    past_start = int(span["start"]) + past.start()
    prefix = text[max(0, past_start - 12) : past_start]
    return not re.search(
        r"\b(?:get|drive|go|pass)\s*$",
        prefix,
        re.IGNORECASE,
    )


def _valid_model_entity_span(span: dict[str, Any]) -> bool:
    tokens = set(re.findall(r"[a-z0-9-]+", str(span["text"]).casefold()))
    non_referential = {
        "a",
        "an",
        "the",
        "you",
        "it",
        "left",
        "left-hand",
        "hand",
        "right",
        "right-hand",
        "straight",
        "forward",
        "ahead",
        "lane",
        "gap",
        "opportunity",
        "chance",
        "safe",
        "safely",
    }
    return bool(tokens - non_referential)


def _canonical_entity(source_span: str, entity_id: str, context: str) -> dict[str, Any]:
    normalized = source_span.casefold()
    head = next(
        (
            candidate
            for candidate in sorted(ENTITY_HEADS, key=len, reverse=True)
            if re.search(rf"\b{re.escape(candidate)}\b", normalized)
        ),
        None,
    )
    entity_type = ENTITY_HEADS.get(head or "", "UNKNOWN")
    if head == "one" and re.search(
        r"\b(?:follow|trail|behind|ahead)\b", context, re.IGNORECASE
    ):
        entity_type = "VEHICLE"
    attributes: dict[str, Any] = {}
    for token, canonical in COLORS.items():
        if re.search(rf"\b{re.escape(token)}\b", normalized):
            attributes["color"] = canonical
            break
    if entity_type == "VEHICLE":
        for token, canonical in VEHICLE_SUBTYPES.items():
            if re.search(rf"\b{re.escape(token)}\b", normalized):
                attributes["vehicle_subtype"] = canonical
                break
    for token, ordinal in ORDINALS.items():
        if re.search(rf"\b{token}\b", normalized):
            attributes["ordinal"] = ordinal
            break

    locator = "UNSPECIFIED"
    locator_context = context.casefold()
    for pattern, canonical in LOCATOR_RELATIONS:
        if re.search(pattern, locator_context):
            locator = canonical
            break
    if locator == "UNSPECIFIED" and any(
        re.search(rf"\b{re.escape(phrase)}\b", locator_context)
        for phrase in ("before", "in front of", "past")
    ):
        locator = "AHEAD"

    canonical_tokens = {
        "the",
        "a",
        "an",
        "this",
        "that",
        "ahead",
        "front",
        "rear",
        "left",
        "right",
        "slow",
        "moving",
        "parked",
        "stopped",
        "nearby",
        *COLORS,
        *VEHICLE_SUBTYPES,
        *ENTITY_HEADS,
        *ORDINALS,
    }
    leftovers = [
        token
        for token in re.findall(r"[a-z0-9-]+", normalized)
        if token not in canonical_tokens
    ]
    return {
        "entity_id": entity_id,
        "type": entity_type,
        "relation": locator,
        "description": source_span,
        "canonical_attributes": attributes,
        "open_descriptors": [" ".join(leftovers)] if leftovers else [],
        "source_span": source_span,
    }


def _predicate_for_relation(text: str) -> str | None:
    normalized = text.casefold()
    return next(
        (
            predicate
            for phrase, predicate in sorted(
                RELATION_PHRASES,
                key=lambda item: len(item[0]),
                reverse=True,
            )
            if re.search(rf"\b{re.escape(phrase)}\b", normalized)
        ),
        None,
    )


def extract_semantic_frame(
    text: str,
    *,
    predicted_spans: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Extract entities and compositional predicates without generating JSON."""

    model_spans = list(predicted_spans or [])
    entity_spans = [span for span in model_spans if span.get("role") == "ENTITY"]
    relation_spans = [
        span for span in model_spans if span.get("role") == "RELATION"
    ]
    entity_spans = [
        span
        for span in entity_spans
        if _valid_model_entity_span(span)
    ]
    entity_spans = _merge_non_overlapping_spans(
        entity_spans + _fallback_entity_spans(text)
    )
    relation_spans = _merge_non_overlapping_spans(
        span
        for span in relation_spans + _fallback_relation_spans(text)
        if _valid_relation_span(text, span)
    )

    entities = []
    for index, span in enumerate(entity_spans, start=1):
        start = max(0, int(span["start"]) - 32)
        end = min(len(text), int(span["end"]) + 24)
        entities.append(
            _canonical_entity(
                str(span["text"]),
                f"target_{index}",
                text[start:end],
            )
        )

    conditions: list[dict[str, Any]] = []
    for relation in relation_spans:
        predicate = _predicate_for_relation(str(relation["text"]))
        if predicate is None:
            continue
        following = [
            (entity, span)
            for entity, span in zip(entities, entity_spans)
            if int(span["start"]) >= int(relation["end"])
        ]
        if not following:
            continue
        primary, _ = min(following, key=lambda item: int(item[1]["start"]))
        condition: dict[str, Any] = {
            "predicate": predicate,
            "subject": "EGO",
            "object": primary["entity_id"],
            "source_span": str(relation["text"]),
        }
        if predicate == "BETWEEN" and len(following) >= 2:
            secondary, _ = sorted(
                following, key=lambda item: int(item[1]["start"])
            )[1]
            condition["secondary_object"] = secondary["entity_id"]
        conditions.append(condition)

    if entities and re.search(
        r"\b(?:keep|maintain) (?:a )?safe distance\b|"
        r"\bnot too close\b|"
        r"\bwithout (?:getting|being) too close\b|"
        r"\b(?:maintain(?:ing)?|keep(?:ing)?) (?:a )?safe gap\b",
        text,
        re.IGNORECASE,
    ):
        conditions.append(
            {
                "predicate": "SAFE_DISTANCE",
                "subject": "EGO",
                "object": entities[0]["entity_id"],
                "source_span": "keep a safe distance",
            }
        )

    unique_conditions = {
        (
            item["predicate"],
            item["subject"],
            item.get("object"),
            item.get("secondary_object"),
        ): item
        for item in conditions
    }
    return {
        "entities": entities,
        "goal_conditions": list(unique_conditions.values()),
        "semantic_spans": _deduplicate_spans(entity_spans + relation_spans),
    }


def enrich_commands_with_frame(
    commands: list[dict[str, Any]],
    frame: dict[str, Any],
) -> list[dict[str, Any]]:
    """Attach shared entity references and relevant goals to classifier actions."""

    entities = list(frame.get("entities") or [])
    conditions = list(frame.get("goal_conditions") or [])
    if not commands or not entities:
        return commands

    targetable = {
        "ADJUST_SPEED",
        "STOP",
        "WAIT",
        "FOLLOW",
        "APPROACH",
        "NAVIGATE_TO",
        "YIELD",
        "PULL_OVER",
        "PARK",
        "OVERTAKE",
        "PASS_BY",
        "AVOID",
        "EMERGENCY_BRAKE",
        "TURN",
        "PROCEED",
    }
    primary = entities[0]
    for command in commands:
        action = command.get("action")
        if action not in targetable:
            continue
        command["target_ref"] = primary["entity_id"]
        command["target_type"] = primary["type"]
        command["target_relation"] = primary["relation"]
        command["target_description"] = primary["description"]
        command["target_attributes"] = dict(primary["canonical_attributes"])
        command["target_open_descriptors"] = list(primary["open_descriptors"])

    for condition in conditions:
        predicate = condition["predicate"]
        preferred_actions = {
            "BEFORE": {"STOP", "PARK"},
            "BEHIND": {"STOP", "PARK", "FOLLOW"},
            "IN_FRONT_OF": {"STOP", "PARK", "NAVIGATE_TO"},
            "NEXT_TO": {"PULL_OVER", "PARK", "STOP"},
            "NEAR": {"APPROACH", "NAVIGATE_TO", "STOP"},
            "PAST": {"PASS_BY", "PROCEED", "STOP"},
            "AFTER": set(),
            "UNTIL": {"FOLLOW", "PROCEED"},
            "VISIBLE": {"STOP", "PULL_OVER", "TURN", "PROCEED"},
            "SAFE_DISTANCE": {"FOLLOW"},
            "BETWEEN": {"STOP", "PARK"},
            "INSIDE": {"PARK", "NAVIGATE_TO"},
        }.get(predicate, set())
        destination = next(
            (
                command
                for command in commands
                if command.get("action") in preferred_actions
            ),
            commands[-1],
        )
        destination.setdefault("goal_conditions", []).append(dict(condition))
    return commands
