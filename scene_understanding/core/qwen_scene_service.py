"""Long-lived Qwen keyframe service with a non-blocking latest-frame queue."""

from __future__ import annotations

import queue
import threading
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from scene_understanding.core.run_qwen_scene_inference import (
    infer_one_record,
    load_backend,
)


@dataclass(frozen=True)
class QwenSceneConfig:
    model_path: Path
    max_new_tokens: int = 1536
    min_visual_tokens: int = 256
    max_visual_tokens: int = 1024

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if not 0 < self.min_visual_tokens <= self.max_visual_tokens:
            raise ValueError("visual token bounds are invalid")


class SceneService(Protocol):
    def infer(self, record: dict[str, Any]) -> dict[str, Any]: ...


class QwenSceneService:
    """Load Qwen once and reuse it for deterministic keyframe inference."""

    def __init__(self, config: QwenSceneConfig) -> None:
        self.config = config
        self._backend: tuple[Any, Any, Any] | None = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    @property
    def is_ready(self) -> bool:
        return self._backend is not None

    def warmup(self) -> None:
        if self._backend is not None:
            return
        with self._load_lock:
            if self._backend is None:
                if not self.config.model_path.is_dir():
                    raise FileNotFoundError(f"model directory not found: {self.config.model_path}")
                self._backend = load_backend(
                    self.config.model_path,
                    min_visual_tokens=self.config.min_visual_tokens,
                    max_visual_tokens=self.config.max_visual_tokens,
                )

    def infer(self, record: dict[str, Any]) -> dict[str, Any]:
        self.warmup()
        assert self._backend is not None
        model, processor, torch_module = self._backend
        with self._infer_lock:
            return infer_one_record(
                record,
                model=model,
                processor=processor,
                torch_module=torch_module,
                model_path=self.config.model_path,
                max_new_tokens=self.config.max_new_tokens,
                min_visual_tokens=self.config.min_visual_tokens,
                max_visual_tokens=self.config.max_visual_tokens,
            )


class LatestFrameWorker:
    """Process at most one queued keyframe and replace stale pending work."""

    _STOP = object()

    def __init__(
        self,
        service: SceneService,
        *,
        name: str = "qwen-scene-worker",
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.service = service
        self._monotonic_clock = monotonic_clock
        self._queue: queue.Queue[object] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._result_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._started = False
        self._closed = False
        self._latest: dict[str, Any] | None = None
        self._latest_completed_monotonic_s: float | None = None
        self._stats = {"submitted": 0, "dropped": 0, "completed": 0, "errors": 0}

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("worker is closed")
            if self._started:
                return
            self._started = True
            self._thread.start()

    def submit(self, record: dict[str, Any]) -> None:
        if not isinstance(record.get("frame_id"), str) or not record["frame_id"]:
            raise ValueError("record.frame_id is required")
        self.start()
        item = deepcopy(record)
        with self._lock:
            if self._closed:
                raise RuntimeError("worker is closed")
            self._stats["submitted"] += 1
            self._result_event.clear()
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                pass
            with self._lock:
                self._stats["dropped"] += 1
            self._queue.put_nowait(item)

    def latest(self, *, max_age_seconds: float | None = None) -> dict[str, Any] | None:
        if max_age_seconds is not None and max_age_seconds < 0:
            raise ValueError("max_age_seconds must be non-negative")
        with self._lock:
            if self._latest is None:
                return None
            if (
                max_age_seconds is not None
                and self._latest_completed_monotonic_s is not None
                and self._monotonic_clock() - self._latest_completed_monotonic_s
                > max_age_seconds
            ):
                return None
            return deepcopy(self._latest)

    def wait_for_result(self, timeout: float | None = None) -> dict[str, Any] | None:
        if not self._result_event.wait(timeout):
            return None
        return self.latest()

    def stats(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)

    def close(self, *, wait: bool = True, timeout: float | None = None) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            started = self._started
        if not started:
            return
        try:
            self._queue.put_nowait(self._STOP)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                pass
            self._queue.put_nowait(self._STOP)
        if wait:
            self._thread.join(timeout)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is self._STOP:
                    return
                assert isinstance(item, dict)
                started = time.perf_counter()
                try:
                    result = self.service.infer(item)
                except Exception as exc:
                    result = {
                        "frame_id": item["frame_id"],
                        "source": item.get("source"),
                        "camera_name": item.get("camera_name"),
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    with self._lock:
                        self._stats["errors"] += 1
                result["service_elapsed_seconds"] = round(time.perf_counter() - started, 6)
                with self._lock:
                    self._latest = result
                    self._latest_completed_monotonic_s = self._monotonic_clock()
                    self._stats["completed"] += 1
                    self._result_event.set()
            finally:
                self._queue.task_done()
