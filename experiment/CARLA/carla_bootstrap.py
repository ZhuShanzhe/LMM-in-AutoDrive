"""Locate the locally installed CARLA Python API before importing ``carla``."""

import glob
import os
import sys


def setup_carla_api(carla_root=None):
    carla_root = carla_root or os.environ.get(
        "CARLA_ROOT", "D:\\CARLA\\carla-0-9-15-windows\\WindowsNoEditor"
    )
    dist_dir = os.path.join(carla_root, "PythonAPI", "carla", "dist")
    api_dir = os.path.join(carla_root, "PythonAPI", "carla")
    candidates = glob.glob(os.path.join(dist_dir, "carla-*.egg"))
    if not candidates:
        raise RuntimeError(
            "CARLA Python API was not found. Set CARLA_ROOT to the WindowsNoEditor directory."
        )
    for path in [candidates[0], api_dir]:
        if path not in sys.path:
            sys.path.insert(0, path)
    return carla_root
