"""ByteTrack adapter. Tracking logic stays in the proven supervision backend."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .detector import Detection


class ByteTrackAdapter:
    def __init__(self, *, frame_rate: int = 10, track_activation_threshold: float = 0.25) -> None:
        try:
            import supervision as sv
        except ImportError as exc:
            raise RuntimeError(
                "ByteTrack requires supervision; install scene_understanding/requirements-realtime.txt"
            ) from exc
        self.sv = sv
        self.tracker = sv.ByteTrack(
            frame_rate=frame_rate,
            track_activation_threshold=track_activation_threshold,
        )
        self.ages: dict[int, int] = defaultdict(int)

    def reset(self) -> None:
        self.tracker.reset()
        self.ages.clear()

    def update(self, detections: Iterable[Detection]) -> list[dict]:
        import numpy as np

        items = list(detections)
        if items:
            sv_detections = self.sv.Detections(
                xyxy=np.asarray([item.bbox_xyxy for item in items], dtype=np.float32),
                confidence=np.asarray([item.confidence for item in items], dtype=np.float32),
                class_id=np.asarray([item.class_id for item in items], dtype=int),
                data={
                    "category": np.asarray([item.category for item in items], dtype=object),
                    "subtype": np.asarray([item.subtype for item in items], dtype=object),
                },
            )
        else:
            sv_detections = self.sv.Detections.empty()

        tracked = self.tracker.update_with_detections(sv_detections)
        output: list[dict] = []
        for index, xyxy in enumerate(tracked.xyxy):
            track_id = int(tracked.tracker_id[index])
            self.ages[track_id] += 1
            output.append(
                {
                    "track_id": f"track_{track_id}",
                    "category": str(tracked.data["category"][index]),
                    "subtype": str(tracked.data["subtype"][index]),
                    "bbox_xyxy": [float(value) for value in xyxy],
                    "confidence": float(tracked.confidence[index]),
                    "age_frames": self.ages[track_id],
                }
            )
        return output
