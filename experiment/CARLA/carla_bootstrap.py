"""Locate the CARLA 0.9.16 Python API in the Linux deployment."""

from __future__ import annotations

import glob
import importlib
from importlib import metadata
import os
import sys
from pathlib import Path


EXPECTED_VERSION = "0.9.16"
DEFAULT_CARLA_ROOT = "/root/autodl-tmp/CARLA_0.9.16"


def _import_carla():
    try:
        return importlib.import_module("carla")
    except ModuleNotFoundError:
        return None


def _validate_version(carla_module) -> None:
    version = getattr(carla_module, "__version__", None)
    if not version:
        try:
            version = metadata.version("carla")
        except metadata.PackageNotFoundError:
            version = None
    if version and version != EXPECTED_VERSION:
        raise RuntimeError(
            f"CARLA Python API {version} is installed, but {EXPECTED_VERSION} is required."
        )


def setup_carla_api(carla_root=None):
    """Make the matching CARLA API importable and return its installation root."""

    configured_root = carla_root or os.environ.get("CARLA_ROOT")
    installed = _import_carla()
    if installed is not None:
        _validate_version(installed)
        return configured_root

    root = Path(configured_root or DEFAULT_CARLA_ROOT).expanduser().resolve()
    api_dir = root / "PythonAPI" / "carla"
    dist_dir = api_dir / "dist"
    candidates = sorted(
        glob.glob(str(dist_dir / "carla-*.whl"))
        + glob.glob(str(dist_dir / "carla-*.egg"))
    )
    for path in [*(Path(item) for item in candidates), api_dir]:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))

    importlib.invalidate_caches()
    installed = _import_carla()
    if installed is None:
        raise RuntimeError(
            "CARLA Python API was not found. Install the CARLA 0.9.16 wheel or "
            f"set CARLA_ROOT to the extracted CARLA directory (checked: {root})."
        )
    _validate_version(installed)
    return str(root)
