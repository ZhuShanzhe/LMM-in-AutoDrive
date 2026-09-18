from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .contracts import ACTION_LABELS


TEACHER_SPECS = {
    "unidrivevla-base": {
        "repository": "https://github.com/xiaomi-research/unidrivevla",
        "checkpoint": "owl10/UniDriveVLA_B2D_Base_Stage3",
        "role": "primary_carla_trajectory_teacher",
    },
    "opendrivevla-0.5b": {
        "repository": "https://github.com/DriveVLA/OpenDriveVLA",
        "checkpoint": "OpenDriveVLA/OpenDriveVLA-0.5B",
        "role": "open_loop_multimodal_teacher_and_small_vla_baseline",
    },
    "simlingo": {
        "repository": "https://github.com/RenzKa/simlingo",
        "checkpoint": "RenzKa/simlingo",
        "role": "language_action_and_closed_loop_reference",
    },
}


@dataclass(frozen=True)
class TeacherPrediction:
    sample_id: str
    model: str
    action_logits: tuple[float, ...]
    target_speed_kmh: float
    latency_ms: float | None
    trajectory: tuple[tuple[float, float], ...]

    @property
    def action(self) -> str:
        index = max(range(len(self.action_logits)), key=self.action_logits.__getitem__)
        return ACTION_LABELS[index]


def parse_teacher_prediction(data: Any) -> TeacherPrediction:
    if not isinstance(data, dict):
        raise ValueError("teacher prediction must be an object")
    logits = data.get("action_logits")
    if (
        not isinstance(logits, list)
        or len(logits) != len(ACTION_LABELS)
        or any(not isinstance(value, (int, float)) for value in logits)
    ):
        raise ValueError(f"action_logits must contain {len(ACTION_LABELS)} numbers")
    trajectory_data = data.get("trajectory", [])
    if not isinstance(trajectory_data, list):
        raise ValueError("trajectory must be an array")
    trajectory: list[tuple[float, float]] = []
    for point in trajectory_data:
        if (
            not isinstance(point, (list, tuple))
            or len(point) != 2
            or any(not isinstance(value, (int, float)) for value in point)
        ):
            raise ValueError("trajectory points must be [x, y] pairs")
        trajectory.append((float(point[0]), float(point[1])))
    sample_id = data.get("sample_id")
    model = data.get("model")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("sample_id must be a non-empty string")
    if model not in TEACHER_SPECS:
        raise ValueError("unsupported teacher model")
    speed = data.get("target_speed_kmh")
    if not isinstance(speed, (int, float)) or not 0 <= float(speed) <= 100:
        raise ValueError("target_speed_kmh must be between 0 and 100")
    latency = data.get("latency_ms")
    if latency is not None and (
        not isinstance(latency, (int, float)) or float(latency) < 0
    ):
        raise ValueError("latency_ms must be null or non-negative")
    return TeacherPrediction(
        sample_id=sample_id,
        model=model,
        action_logits=tuple(float(value) for value in logits),
        target_speed_kmh=float(speed),
        latency_ms=None if latency is None else float(latency),
        trajectory=tuple(trajectory),
    )


class JsonlTeacherStore:
    """Read normalized teacher outputs exported by an official model repository."""

    def __init__(self, records: Iterable[TeacherPrediction]) -> None:
        self._records = {record.sample_id: record for record in records}
        if not self._records:
            raise ValueError("teacher store cannot be empty")

    @classmethod
    def from_path(cls, path: str | Path) -> "JsonlTeacherStore":
        records: list[TeacherPrediction] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    records.append(parse_teacher_prediction(json.loads(line)))
                except (ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
        return cls(records)

    def get(self, sample_id: str) -> TeacherPrediction:
        try:
            return self._records[sample_id]
        except KeyError as exc:
            raise KeyError(f"missing teacher prediction for {sample_id!r}") from exc


def compare_action_predictions(
    student_records: Iterable[dict[str, Any]],
    teacher_store: JsonlTeacherStore,
) -> dict[str, float | int]:
    total = 0
    matches = 0
    for record in student_records:
        sample_id = record.get("sample_id")
        action = record.get("action")
        if not isinstance(sample_id, str) or action not in ACTION_LABELS:
            raise ValueError("student record requires sample_id and a valid action")
        total += 1
        matches += int(action == teacher_store.get(sample_id).action)
    if total == 0:
        raise ValueError("student predictions cannot be empty")
    return {
        "samples": total,
        "teacher_action_agreement": round(matches / total, 6),
    }
