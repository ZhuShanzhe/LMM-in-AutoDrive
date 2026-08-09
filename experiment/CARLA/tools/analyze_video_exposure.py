import json
import sys
from pathlib import Path

import cv2
import numpy as np


video_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
capture = cv2.VideoCapture(str(video_path))
if not capture.isOpened():
    raise RuntimeError(f"cannot open video: {video_path}")

frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
fps = float(capture.get(cv2.CAP_PROP_FPS))
# Sample at least 600 frames across the video.
sample_every = max(1, frame_count // 600)
samples = []
for frame_index in range(0, frame_count, sample_every):
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok:
        continue
    luma = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    samples.append(
        {
            "frame": frame_index,
            "time_s": round(frame_index / fps, 3),
            "mean_luma": float(np.mean(luma)),
            "p99_luma": float(np.percentile(luma, 99)),
            "pixels_ge_250_ratio": float(np.mean(luma >= 250)),
            "pixels_eq_255_ratio": float(np.mean(luma == 255)),
        }
    )
capture.release()
if len(samples) < 600:
    raise RuntimeError(
        f"expected at least 600 sampled frames, got {len(samples)}"
    )

result = {
    "video": str(video_path),
    "frame_count": frame_count,
    "fps": fps,
    "sample_every_frames": sample_every,
    "sample_count": len(samples),
    "aggregate": {
        "mean_luma": float(np.mean([row["mean_luma"] for row in samples])),
        "max_sample_mean_luma": float(max(row["mean_luma"] for row in samples)),
        "mean_pixels_ge_250_ratio": float(
            np.mean([row["pixels_ge_250_ratio"] for row in samples])
        ),
        "max_pixels_ge_250_ratio": float(
            max(row["pixels_ge_250_ratio"] for row in samples)
        ),
        "mean_pixels_eq_255_ratio": float(
            np.mean([row["pixels_eq_255_ratio"] for row in samples])
        ),
        "global_overexposure": bool(
            float(np.mean([row["mean_luma"] for row in samples])) > 200.0
        ),
    },
    "samples": samples,
}
output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result["aggregate"], indent=2))
