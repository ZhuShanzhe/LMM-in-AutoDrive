"""Lossless CARLA XYZI replay input, without a simulator or torch dependency."""

from pathlib import Path
import numpy as np


def load_xyzi(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.stat().st_size % 16:
        raise ValueError("invalid float32 XYZI byte length")
    points = np.fromfile(path, dtype="<f4").reshape(-1, 4)
    if not np.isfinite(points).all():
        raise ValueError("non-finite LiDAR values")
    return points
