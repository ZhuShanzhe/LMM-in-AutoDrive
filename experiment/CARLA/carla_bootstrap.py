"""Locate the CARLA 0.9.16 Python API and navigation agents."""

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
    workspace_root = Path(__file__).resolve().parents[3]
    roots = []
    for candidate in (
        configured_root,
        workspace_root / "carla-0-9-16",
        DEFAULT_CARLA_ROOT,
    ):
        if candidate is None:
            continue
        root = Path(candidate).expanduser().resolve()
        if root not in roots:
            roots.append(root)

    installed = _import_carla()
    if installed is not None:
        _validate_version(installed)
        for root in roots:
            api_dir = root / "PythonAPI" / "carla"
            if api_dir.is_dir():
                if str(api_dir) not in sys.path:
                    sys.path.append(str(api_dir))
                return str(root)
        return configured_root

    selected_root = None
    for root in roots:
        api_dir = root / "PythonAPI" / "carla"
        dist_dir = api_dir / "dist"
        candidates = sorted(
            glob.glob(str(dist_dir / "carla-*.whl"))
            + glob.glob(str(dist_dir / "carla-*.egg"))
        )
        for path in [*(Path(item) for item in candidates), api_dir]:
            if path.exists() and str(path) not in sys.path:
                sys.path.insert(0, str(path))
        if api_dir.is_dir():
            selected_root = root
            break

    importlib.invalidate_caches()
    installed = _import_carla()
    if installed is None:
        checked = ", ".join(str(path) for path in roots)
        raise RuntimeError(
            "CARLA Python API was not found. Install the CARLA 0.9.16 wheel or "
            f"set CARLA_ROOT to the extracted CARLA directory (checked: {checked})."
        )
    _validate_version(installed)
    return str(selected_root) if selected_root is not None else configured_root
