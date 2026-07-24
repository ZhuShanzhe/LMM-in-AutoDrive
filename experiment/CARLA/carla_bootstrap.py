"""Locate the locally installed CARLA Python API before importing ``carla``."""

import glob
import importlib.util
import os
import sys


def setup_carla_api(carla_root=None):
    carla_root = carla_root or os.environ.get("CARLA_ROOT")
    if importlib.util.find_spec("carla") is not None:
        return carla_root
    if not carla_root:
        raise RuntimeError(
            "Install the matching CARLA Python wheel, or set CARLA_ROOT for an egg-based API."
        )
    dist_dir = os.path.join(carla_root, "PythonAPI", "carla", "dist")
    api_dir = os.path.join(carla_root, "PythonAPI", "carla")
    candidates = glob.glob(os.path.join(dist_dir, "carla-*.egg"))
    if not candidates:
        raise RuntimeError(
            "CARLA Python API is not installed. Install the matching wheel from "
            "CARLA_ROOT/PythonAPI/carla/dist before running this script."
        )
    for path in [candidates[0], api_dir]:
        if path not in sys.path:
            sys.path.insert(0, path)
    return carla_root
