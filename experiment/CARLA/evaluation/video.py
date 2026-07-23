"""Direct CARLA-camera to H.264 recorder without temporary image files."""

import os
import shutil
import subprocess
import queue
import threading


class FfmpegVideoWriter:
    def __init__(self, output_path, width, height, fps, ffmpeg_path=None):
        self.output_path = output_path
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.ffmpeg_path = self._find_ffmpeg(ffmpeg_path)
        self.process = None
        self.frame_count = 0
        self.dropped_frames = 0
        self._frames = queue.Queue(maxsize=64)
        self._worker = None
        self._last_frame = None

    def start(self):
        output_dir = os.path.dirname(os.path.abspath(self.output_path))
        os.makedirs(output_dir, exist_ok=True)
        command = [
            self.ffmpeg_path,
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgra",
            "-video_size", "{0}x{1}".format(self.width, self.height),
            "-framerate", str(self.fps),
            "-i", "-",
            "-an",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "21",
            "-pix_fmt", "yuv420p",
            self.output_path,
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._worker = threading.Thread(target=self._write_frames, daemon=True)
        self._worker.start()

    def write(self, image):
        self.write_raw(bytes(image.raw_data))

    def write_raw(self, raw_frame):
        if self.process is None:
            return
        self._last_frame = bytes(raw_frame)
        try:
            self._frames.put_nowait(self._last_frame)
        except queue.Full:
            self.dropped_frames += 1

    def append_last_frame(self, frame_count):
        """Extend a terminal frame without changing simulation metrics."""
        if self.process is None or self._last_frame is None:
            return 0
        appended = max(0, int(frame_count))
        for _ in range(appended):
            self._frames.put(self._last_frame)
        return appended

    def _write_frames(self):
        while True:
            frame = self._frames.get()
            if frame is None:
                return
            if self.process is None or self.process.stdin is None:
                return
            self.process.stdin.write(frame)
            self.frame_count += 1

    def close(self):
        if self.process is None:
            return
        self._frames.put(None)
        self._worker.join(timeout=120)
        if self._worker.is_alive():
            raise RuntimeError("ffmpeg frame writer did not finish")
        if self.process.stdin is not None:
            self.process.stdin.close()
        return_code = self.process.wait(timeout=60)
        if return_code != 0:
            raise RuntimeError("ffmpeg exited with code {0}".format(return_code))
        self.process = None

    @staticmethod
    def _find_ffmpeg(explicit_path):
        candidate = explicit_path or shutil.which("ffmpeg")
        if candidate and os.path.isfile(candidate):
            return candidate
        raise RuntimeError("ffmpeg was not found; pass --ffmpeg <path-to-ffmpeg.exe>")
