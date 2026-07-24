from __future__ import annotations

from typing import Any
from uuid import uuid4

from .schema_tools import validate_document


def make_step(
    step_id: str,
    action: str,
    *,
    parameters: dict[str, Any] | None = None,
    trigger: dict[str, Any] | None = None,
    depends_on: list[str] | None = None,
    preconditions: list[str] | None = None,
    on_blocked: str = "SAFE_STOP",
    purpose: str | None = None,
    target: dict[str, Any] | None = None,
    completion: dict[str, Any] | None = None,
) -> dict[str, Any]:
    step: dict[str, Any] = {
        "step_id": step_id,
        "action": action,
        "parameters": parameters or {},
        "trigger": trigger or {"type": "IMMEDIATE"},
        "depends_on": depends_on or [],
        "preconditions": preconditions or [],
        "on_blocked": on_blocked,
    }
    if purpose is not None:
        step["purpose"] = purpose
    if target is not None:
        step["target"] = target
    if completion is not None:
        step["completion"] = completion
    return step


def make_document(
    *,
    raw_text: str,
    normalized_text: str,
    modality: str,
    category: str,
    urgency: str,
    steps: list[dict[str, Any]],
    status: str,
    method: str,
    model: str | None,
    confidence: float,
    latency_ms: float,
    request_id: str | None = None,
    missing_slots: list[str] | None = None,
    warnings: list[str] | None = None,
    clarification_question: str | None = None,
    driving_style: str = "NORMAL",
    max_speed_mps: float | None = None,
    language: str = "zh-CN",
) -> dict[str, Any]:
    constraints: dict[str, Any] = {
        "safety_first": True,
        "obey_traffic_rules": True,
        "driving_style": driving_style,
    }
    if max_speed_mps is not None:
        constraints["max_speed_mps"] = max_speed_mps

    parse_result: dict[str, Any] = {
        "status": status,
        "method": method,
        "model": model,
        "confidence": round(max(0.0, min(1.0, confidence)), 4),
        "missing_slots": missing_slots or [],
        "warnings": warnings or [],
        "latency_ms": round(max(0.0, latency_ms), 3),
    }
    if clarification_question:
        parse_result["clarification_question"] = clarification_question

    document = {
        "schema_version": "1.1.0",
        "request_id": request_id or f"cmd-{uuid4().hex[:16]}",
        "input": {
            "modality": modality,
            "language": language,
            "raw_text": raw_text,
            "normalized_text": normalized_text,
        },
        "intent": {
            "category": category,
            "urgency": urgency,
            "steps": steps,
            "constraints": constraints,
        },
        "parse_result": parse_result,
    }
    validate_document(document)
    return document
