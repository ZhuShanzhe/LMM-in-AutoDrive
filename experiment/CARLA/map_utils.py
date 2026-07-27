"""Compatibility helpers for CARLA map identifiers."""

from __future__ import annotations

from collections.abc import Iterable


def resolve_carla_map_name(requested_map: str, available_maps: Iterable[str]) -> str:
    """Return the server-specific spelling for one configured CARLA map.

    CARLA servers have accepted both full asset paths and short map names
    across releases, but some builds expose only the latter to ``load_world``.
    Match the requested asset basename against the current server's advertised
    maps so existing scenario configurations remain portable.
    """

    requested = str(requested_map).strip()
    if not requested:
        raise ValueError("requested CARLA map must be a non-empty string")
    available = [str(item).strip() for item in available_maps if str(item).strip()]
    if requested in available:
        return requested

    basename = requested.rstrip("/").split("/")[-1]
    matches = [
        candidate
        for candidate in available
        if candidate.rstrip("/").split("/")[-1] == basename
    ]
    return matches[0] if len(matches) == 1 else requested
