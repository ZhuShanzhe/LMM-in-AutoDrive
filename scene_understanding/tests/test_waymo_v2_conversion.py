from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene_understanding.scripts.data.prepare_waymo_v2 import convert_split


def test_convert_split_joins_camera_keys_and_writes_relative_image(tmp_path: Path):
    source = tmp_path / "source" / "training"
    image_dir = source / "camera_image"
    box_dir = source / "camera_box"
    image_dir.mkdir(parents=True)
    box_dir.mkdir(parents=True)
    encoded = io.BytesIO()
    Image.new("RGB", (64, 32), color=(10, 20, 30)).save(encoded, format="JPEG")
    keys = {
        "key.segment_context_name": ["segment"],
        "key.frame_timestamp_micros": [123],
        "key.camera_name": [1],
    }
    pq.write_table(
        pa.table({**keys, "[CameraImageComponent].image": [encoded.getvalue()]}),
        image_dir / "segment.parquet",
    )
    pq.write_table(
        pa.table(
            {
                **keys,
                "[CameraBoxComponent].box.center.x": [32.0],
                "[CameraBoxComponent].box.center.y": [16.0],
                "[CameraBoxComponent].box.size.x": [16.0],
                "[CameraBoxComponent].box.size.y": [8.0],
                "[CameraBoxComponent].type": [2],
                "[CameraBoxComponent].difficulty_level.detection": [1],
            }
        ),
        box_dir / "segment.parquet",
    )

    output = tmp_path / "converted"
    summary = convert_split(tmp_path / "source", output, "training")
    row = json.loads((output / "manifests" / "training.jsonl").read_text())

    assert summary["frames"] == 1
    assert summary["boxes:pedestrian"] == 1
    assert row["camera_name"] == "front"
    assert row["image_path"] == "../images/training/segment_123_1.jpg"
    assert row["annotations"][0]["category"] == "pedestrian"
    assert row["annotations"][0]["bbox_2d"] == [0.375, 0.375, 0.625, 0.625]
    assert (output / "manifests" / row["image_path"]).resolve().is_file()
