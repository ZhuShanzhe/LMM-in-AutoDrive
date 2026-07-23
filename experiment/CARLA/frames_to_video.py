"""Convert images recorded by ExperimentCamera into an MP4 demo video."""

import argparse
import glob
import os
import shutil
import subprocess
import tempfile


def parse_args():
    parser = argparse.ArgumentParser(description="Create an MP4 video from CARLA camera frames")
    parser.add_argument("--frames", required=True, help="Directory containing PNG frames")
    parser.add_argument("--output", required=True, help="Output .mp4 path")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--ffmpeg", default=None, help="Optional path to ffmpeg.exe")
    parser.add_argument("--extension", default="png", help="Input frame extension, without a dot")
    return parser.parse_args()


def main():
    args = parse_args()
    extension = args.extension.lower().lstrip(".")
    frame_paths = sorted(glob.glob(os.path.join(args.frames, "*.{0}".format(extension))))
    if not frame_paths:
        raise RuntimeError("No {0} frames found in {1}".format(extension, args.frames))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    ffmpeg = find_ffmpeg(args.ffmpeg)
    manifest_path = write_concat_manifest(frame_paths, args.fps)
    try:
        command = [
            ffmpeg,
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", manifest_path,
            "-vsync", "vfr",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            args.output,
        ]
        subprocess.check_call(command)
    finally:
        if os.path.exists(manifest_path):
            os.remove(manifest_path)
    print("[Done] Wrote {0} from {1} frames".format(args.output, len(frame_paths)))


def find_ffmpeg(explicit_path):
    candidates = [explicit_path, shutil.which("ffmpeg")]
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(os.path.join(conda_prefix, "Library", "bin", "ffmpeg.exe"))
    candidates.append(r"D:\\anaconda3\\envs\\carla37\\Library\\bin\\ffmpeg.exe")
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    raise RuntimeError("ffmpeg was not found; pass --ffmpeg <path-to-ffmpeg.exe>")


def write_concat_manifest(frame_paths, fps):
    interval_s = 1.0 / float(fps)
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    try:
        for path in frame_paths:
            normalized_path = os.path.abspath(path).replace("\\", "/").replace("'", "\\\\'")
            handle.write("file '{0}'\n".format(normalized_path))
            handle.write("duration {0:.8f}\n".format(interval_s))
        normalized_path = os.path.abspath(frame_paths[-1]).replace("\\", "/").replace("'", "\\\\'")
        handle.write("file '{0}'\n".format(normalized_path))
    finally:
        handle.close()
    return handle.name


if __name__ == "__main__":
    main()
