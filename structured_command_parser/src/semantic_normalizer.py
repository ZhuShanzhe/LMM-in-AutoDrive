from __future__ import annotations

import re
from typing import Any


ACTION_EXPRESSIONS = (
    (
        "CHANGE_LANE",
        re.compile(
            r"\b(?:change|switch|shift|move|get|merge)\b.{0,28}\b(?:lane|over)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "TURN",
        re.compile(
            r"\b(?:turn|make (?:a )?(?:left|right))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "PROCEED",
        re.compile(
            r"\b(?:continue|proceed|go|drive)\b.{0,18}\b(?:straight|forward|ahead)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "OVERTAKE",
        re.compile(r"\b(?:overtake|get past|pass)\b", re.IGNORECASE),
    ),
    (
        "RESUME",
        re.compile(
            r"\b(?:resume|return|get back|move back)\b.{0,24}\b(?:lane|route|course)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "FOLLOW",
        re.compile(r"\b(?:follow|trail|keep up with|stay behind)\b", re.IGNORECASE),
    ),
    (
        "STOP",
        re.compile(r"\b(?:stop|halt|pull up)\b", re.IGNORECASE),
    ),
)

CANONICAL_REWRITES = (
    (
        re.compile(
            r"\bwait for (?:a )?safe gap,?\s*then "
            r"(?:move|switch|shift) (?:to|into) (?:the )?"
            r"(left|right)(?:-hand)? lane\b",
            re.IGNORECASE,
        ),
        r"change to the \1 lane when safe",
        "SYNONYM_CANONICALIZATION",
    ),
    (
        re.compile(
            r"\bfind (?:a|an) (?:safe )?(?:chance|opportunity) to "
            r"(?:move|merge|get) into the left(?:-hand)? lane\b",
            re.IGNORECASE,
        ),
        "change to the left lane when safe",
        "SYNONYM_CANONICALIZATION",
    ),
    (
        re.compile(
            r"\bfind (?:a|an) (?:safe )?(?:chance|opportunity) to "
            r"(?:move|merge|get) into the right(?:-hand)? lane\b",
            re.IGNORECASE,
        ),
        "change to the right lane when safe",
        "SYNONYM_CANONICALIZATION",
    ),
    (
        re.compile(
            r"\b(?:get|move|go) back (?:into|to) (?:the )?"
            r"(?:original|previous) lane\b",
            re.IGNORECASE,
        ),
        "resume the original lane",
        "ELLIPSIS_RESOLUTION",
    ),
    (
        re.compile(r"\bkeep up with\b", re.IGNORECASE),
        "follow",
        "SYNONYM_CANONICALIZATION",
    ),
    (
        re.compile(
            r"\btake the (left|right)(?:-hand)? lane\b",
            re.IGNORECASE,
        ),
        r"change to the \1 lane",
        "SYNONYM_CANONICALIZATION",
    ),
    (
        re.compile(
            r"\buse (?:the )?(left|right)(?:-hand)? lane\b",
            re.IGNORECASE,
        ),
        r"change to the \1 lane",
        "SYNONYM_CANONICALIZATION",
    ),
    (
        re.compile(r"\bcarry on straight\b", re.IGNORECASE),
        "continue straight",
        "SYNONYM_CANONICALIZATION",
    ),
    (
        re.compile(r"\bdo not get too close\b|\bdon't get too close\b", re.IGNORECASE),
        "keep a safe distance",
        "NEGATIVE_CONSTRAINT_NORMALIZATION",
    ),
)

ASR_CANDIDATES = (
    {
        "pattern": re.compile(r"前方路口又转"),
        "source": "又转",
        "replacement": "右转",
        "confidence": 0.62,
        "reason": "ASR_HOMOPHONE_CANDIDATE",
    },
    {
        "pattern": re.compile(r"\bwrite (?:turn|lane)\b", re.IGNORECASE),
        "source": "write",
        "replacement": "right",
        "confidence": 0.92,
        "reason": "ASR_HOMOPHONE_CORRECTION",
    },
)

NEGATION_PREFIX = re.compile(
    r"\b(?:do not|don't|never|avoid|no need to|skip)\s+",
    re.IGNORECASE,
)


def _direction(text: str) -> str | None:
    normalized = text.casefold()
    if re.search(r"\b(?:left|left-hand)\b", normalized):
        return "LEFT"
    if re.search(r"\b(?:right|right-hand)\b", normalized):
        return "RIGHT"
    if re.search(r"\b(?:straight|forward|ahead)\b", normalized):
        return "STRAIGHT"
    return None


def _suppressed_intents(text: str) -> list[dict[str, Any]]:
    suppressed: list[dict[str, Any]] = []
    for match in NEGATION_PREFIX.finditer(text):
        start = match.start()
        tail = text[match.end() :]
        boundary = re.search(
            r"[,;.]|\b(?:but|instead|then)\b|"
            r"\band\s+(?=(?:continue|proceed|keep|go|drive|turn|change|"
            r"merge|stop|follow)\b)",
            tail,
            re.IGNORECASE,
        )
        end = match.end() + (boundary.start() if boundary else len(tail))
        span = text[start:end].strip(" ,;.")
        action = next(
            (
                name
                for name, pattern in ACTION_EXPRESSIONS
                if pattern.search(span)
            ),
            None,
        )
        if action is None:
            continue
        item: dict[str, Any] = {
            "action": action,
            "reason": "EXPLICIT_NEGATION",
            "source_span": span,
        }
        direction = _direction(span)
        if direction is not None:
            item["parameters"] = {"direction": direction}
        suppressed.append(item)
    return suppressed


def normalize_semantics(
    text: str,
    *,
    source_text: str | None = None,
) -> dict[str, Any]:
    """Normalize surface variation while retaining every auditable edit."""

    normalized = " ".join(text.strip().split())
    edits: list[dict[str, Any]] = []
    warnings: list[str] = []
    requires_confirmation = False

    for pattern, replacement, edit_type in CANONICAL_REWRITES:
        while True:
            match = pattern.search(normalized)
            if match is None:
                break
            expanded_replacement = match.expand(replacement)
            edits.append(
                {
                    "type": edit_type,
                    "source_span": match.group(0),
                    "replacement": expanded_replacement,
                    "confidence": 1.0,
                    "requires_confirmation": False,
                }
            )
            normalized = (
                normalized[: match.start()]
                + expanded_replacement
                + normalized[match.end() :]
            )

    asr_source = source_text if source_text is not None else text
    for candidate in ASR_CANDIDATES:
        match = candidate["pattern"].search(asr_source)
        if match is None:
            continue
        confirm = float(candidate["confidence"]) < 0.85
        edits.append(
            {
                "type": candidate["reason"],
                "source_span": candidate["source"],
                "replacement": candidate["replacement"],
                "confidence": candidate["confidence"],
                "requires_confirmation": confirm,
            }
        )
        if confirm:
            requires_confirmation = True
            warnings.append(
                f"Ambiguous ASR candidate {candidate['source']!r} may mean "
                f"{candidate['replacement']!r}; direction was not silently corrected."
            )
        else:
            normalized = re.sub(
                re.escape(candidate["source"]),
                candidate["replacement"],
                normalized,
                flags=re.IGNORECASE,
            )

    suppressed = _suppressed_intents(normalized)
    for item in suppressed:
        span = re.escape(item["source_span"])
        normalized = re.sub(span, " ", normalized, count=1, flags=re.IGNORECASE)
    normalized = re.sub(
        r"^\s*(?:but|instead|then)\b[\s,;:]*",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"\s+([,.;])", r"\1", " ".join(normalized.split()))

    unresolved_references = []
    explicit_anaphora = re.search(
        r"\b(?:that one|that vehicle|the former)\b",
        normalized,
        re.IGNORECASE,
    )
    referential_it = re.search(
        r"\b(?:follow|trail|approach|overtake|pass|avoid|stop (?:near|before|"
        r"behind|beside)|park (?:near|before|behind|beside)) it\b",
        normalized,
        re.IGNORECASE,
    )
    if explicit_anaphora or referential_it:
        if not re.search(
            r"\b(?:car|vehicle|truck|bus|van|pedestrian|cyclist)\b",
            normalized,
            re.IGNORECASE,
        ):
            unresolved_references.append("anaphoric_target")

    return {
        "normalized_text": normalized,
        "edits": edits,
        "suppressed_intents": suppressed,
        "unresolved_references": unresolved_references,
        "warnings": warnings,
        "requires_confirmation": requires_confirmation,
    }


def filter_suppressed_actions(
    commands: list[dict[str, Any]],
    suppressed_intents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove only the explicitly negated action-direction combination."""

    remaining = list(commands)
    for suppressed in suppressed_intents:
        expected_direction = (suppressed.get("parameters") or {}).get("direction")
        for index, command in enumerate(remaining):
            if command.get("action") != suppressed["action"]:
                continue
            if expected_direction is not None and command.get("direction") not in {
                None,
                expected_direction,
            }:
                continue
            remaining.pop(index)
            break
    return remaining
