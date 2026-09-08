from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


CARLA_DIR = Path(__file__).resolve().parents[1]
if str(CARLA_DIR) not in sys.path:
    sys.path.insert(0, str(CARLA_DIR))

from evaluation.camera import ExperimentCamera


class _Image:
    def __init__(self, frame: int):
        self.frame = frame
        self.raw_data = b"frame"
        self.saved_paths: list[str] = []

    def save_to_disk(self, path: str) -> None:
        self.saved_paths.append(path)


class _Writer:
    def __init__(self):
        self.frames: list[bytes] = []

    def write_raw(self, frame: bytes) -> None:
        self.frames.append(frame)


class CameraVideoCadenceTests(unittest.TestCase):
    def test_sparse_image_sampling_does_not_reduce_video_fps(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            camera = ExperimentCamera(
                None,
                None,
                temp_dir,
                every_n_frames=200,
                save_images=True,
            )
            writer = _Writer()
            camera.video_writer = writer

            first = _Image(1)
            camera._images.put(first)
            self.assertTrue(camera.save_frame(1))
            self.assertEqual(writer.frames, [b"frame"])
            self.assertEqual(first.saved_paths, [])
            self.assertEqual(camera.saved_frames, 0)

            sampled = _Image(200)
            camera._images.put(sampled)
            self.assertTrue(camera.save_frame(200))
            self.assertEqual(writer.frames, [b"frame", b"frame"])
            self.assertEqual(len(sampled.saved_paths), 1)
            self.assertEqual(camera.saved_frames, 1)


if __name__ == "__main__":
    unittest.main()
