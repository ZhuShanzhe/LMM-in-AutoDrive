"""Locate the CARLA 0.9.16 Python API without machine-specific paths."""

from __future__ import annotations

import glob
import importlib
from importlib import metadata
import os
import sys
from pathlib import Path
import zipfile


EXPECTED_VERSION = "0.9.16"


def _python_abi_tag() -> str:
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def _archive_cache_dir(repository_root: Path, archive: Path) -> Path:
    configured_cache = os.environ.get("CARLA_PYTHON_CACHE")
    cache_root = (
        Path(configured_cache).expanduser().resolve()
        if configured_cache
        else repository_root / ".runtime" / "carla-python-api"
    )
    return cache_root / _python_abi_tag() / archive.stem


def _extract_python_archive(repository_root: Path, archive: Path) -> Path:
    destination = _archive_cache_dir(repository_root, archive)
    extension_modules = list((destination / "carla").glob("libcarla*.so"))
    if extension_modules:
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(destination)
    extension_modules = list((destination / "carla").glob("libcarla*.so"))
    if not extension_modules:
        raise RuntimeError(
            f"CARLA archive does not contain libcarla for Linux: {archive}"
        )
    return destination


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
    repository_root = Path(__file__).resolve().parents[2]
    roots = []
    for candidate in (
        configured_root,
        repository_root / "CARLA_0.9.16",
        repository_root.parent / "CARLA_0.9.16",
        repository_root.parent / "carla-0-9-16",
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
                    sys.path.insert(0, str(api_dir))
                return str(root)
        return (
            str(Path(configured_root).expanduser().resolve())
            if configured_root
            else None
        )

    selected_root = None
    for root in roots:
        api_dir = root / "PythonAPI" / "carla"
        dist_dir = api_dir / "dist"
        archives = sorted(
            glob.glob(str(dist_dir / "carla-*.whl"))
            + glob.glob(str(dist_dir / "carla-*.egg"))
        )
        matching_archives = [
            Path(item)
            for item in archives
            if _python_abi_tag() in Path(item).name
        ]
        paths = [
            *(
                _extract_python_archive(repository_root, archive)
                for archive in matching_archives
            ),
            api_dir,
        ]
        for path in paths:
            if path.exists() and str(path) not in sys.path:
                sys.path.insert(0, str(path))
        importlib.invalidate_caches()
        installed = _import_carla()
        if installed is not None:
            selected_root = root
            break

    if installed is None:
        checked = ", ".join(str(path) for path in roots)
        raise RuntimeError(
            "CARLA Python API was not found. Install the CARLA 0.9.16 wheel or "
            f"set CARLA_ROOT to the extracted CARLA directory (checked: {checked})."
        )
    _validate_version(installed)
    return str(selected_root) if selected_root is not None else configured_root
