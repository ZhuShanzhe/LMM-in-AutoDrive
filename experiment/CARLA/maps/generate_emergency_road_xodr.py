"""Generate the deterministic 6 km road for emergency-response scene 3.

The OpenDRIVE file contains a straight, junction-free urban expressway with
three driving lanes in each direction. Construction cones and event actors are
created at runtime; keeping them out of OpenDRIVE avoids relying on unsupported
conversion of arbitrary OpenDRIVE objects into Unreal assets.

The scored route remains 6 km long.  A short physical run-off section follows
the logical finish so CARLA vehicles cannot fall off the generated road mesh
before the client observes route completion.
"""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "scene_3_emergency_road/v1"
MAP_NAME = "VLA_EmergencyRoad_6km"
ROAD_ID = 1
SCENE_LENGTH_M = 6000.0
RUNOFF_LENGTH_M = 100.0
ROAD_LENGTH_M = (
    SCENE_LENGTH_M + RUNOFF_LENGTH_M
)
DRIVING_LANE_WIDTH_M = 3.5
SHOULDER_WIDTH_M = 2.5
SIDEWALK_WIDTH_M = 1.8
SPEED_LIMIT_MPS = 22.22

EGO_START_ROAD_ID = ROAD_ID
EGO_START_LANE_ID = -2
EGO_START_S_M = 50.0

EVENT_DISTANCES_M = {
    "cut_in": 1080.0,
    "advance_warning": 1550.0,
    "cone_taper_start": 1850.0,
    "right_lane_closed": 2400.0,
    "temporary_pedestrian": 3200.0,
    "blocked_lane": 4300.0,
    "work_zone_exit": 5050.0,
}

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "maps"
    / "output"
    / "VLA_EmergencyRoad_6km.xodr"
)


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _subelement(
    parent: ET.Element,
    tag: str,
    **attributes: Any,
) -> ET.Element:
    return ET.SubElement(
        parent,
        tag,
        {
            name: _format_value(value)
            for name, value in attributes.items()
        },
    )


def _add_width(
    lane: ET.Element,
    width_m: float,
) -> None:
    _subelement(
        lane,
        "width",
        sOffset=0.0,
        a=width_m,
        b=0.0,
        c=0.0,
        d=0.0,
    )


def _add_road_mark(
    lane: ET.Element,
    *,
    mark_type: str,
    color: str,
    lane_change: str,
    width_m: float,
) -> None:
    _subelement(
        lane,
        "roadMark",
        sOffset=0.0,
        type=mark_type,
        weight="standard",
        color=color,
        material="standard",
        width=width_m,
        laneChange=lane_change,
        height=0.0,
    )


def _add_center_lane(
    center: ET.Element,
) -> None:
    lane = _subelement(
        center,
        "lane",
        id=0,
        type="none",
        level=False,
    )
    _add_road_mark(
        lane,
        mark_type="solid solid",
        color="yellow",
        lane_change="none",
        width_m=0.25,
    )


def _add_driving_lane(
    side: ET.Element,
    lane_id: int,
) -> None:
    lane = _subelement(
        side,
        "lane",
        id=lane_id,
        type="driving",
        level=False,
    )
    _subelement(lane, "link")
    _add_width(
        lane,
        DRIVING_LANE_WIDTH_M,
    )
    _add_road_mark(
        lane,
        mark_type=(
            "solid"
            if abs(lane_id) == 3
            else "broken"
        ),
        color="white",
        lane_change=(
            "none"
            if abs(lane_id) == 3
            else "both"
        ),
        width_m=0.15,
    )


def _add_shoulder_lane(
    side: ET.Element,
    lane_id: int,
) -> None:
    lane = _subelement(
        side,
        "lane",
        id=lane_id,
        type="shoulder",
        level=False,
    )
    _add_width(
        lane,
        SHOULDER_WIDTH_M,
    )
    _add_road_mark(
        lane,
        mark_type="solid",
        color="white",
        lane_change="none",
        width_m=0.15,
    )


def _add_sidewalk_lane(
    side: ET.Element,
    lane_id: int,
) -> None:
    lane = _subelement(
        side,
        "lane",
        id=lane_id,
        type="sidewalk",
        level=False,
    )
    _add_width(
        lane,
        SIDEWALK_WIDTH_M,
    )
    _add_road_mark(
        lane,
        mark_type="curb",
        color="standard",
        lane_change="none",
        width_m=0.15,
    )


def _add_lane_side(
    lane_section: ET.Element,
    tag: str,
    sign: int,
) -> None:
    side = _subelement(
        lane_section,
        tag,
    )
    for index in range(1, 4):
        _add_driving_lane(
            side,
            sign * index,
        )
    _add_shoulder_lane(
        side,
        sign * 4,
    )
    _add_sidewalk_lane(
        side,
        sign * 5,
    )


def build_opendrive() -> ET.Element:
    root = ET.Element("OpenDRIVE")
    header = _subelement(
        root,
        "header",
        revMajor=1,
        revMinor=4,
        name=MAP_NAME,
        version="1.0",
        date="2026-07-29",
        north=20.0,
        south=-20.0,
        east=ROAD_LENGTH_M,
        west=0.0,
    )
    geo_reference = _subelement(
        header,
        "geoReference",
    )
    geo_reference.text = (
        "+proj=tmerc +lat_0=0 +lon_0=0 +k=1 "
        "+x_0=0 +y_0=0 +datum=WGS84 "
        "+units=m +no_defs"
    )

    road = _subelement(
        root,
        "road",
        name="emergency_expressway",
        length=ROAD_LENGTH_M,
        id=ROAD_ID,
        junction=-1,
        rule="RHT",
    )
    _subelement(road, "link")

    road_type = _subelement(
        road,
        "type",
        s=0.0,
        type="motorway",
    )
    _subelement(
        road_type,
        "speed",
        max=SPEED_LIMIT_MPS,
        unit="m/s",
    )

    plan_view = _subelement(
        road,
        "planView",
    )
    geometry = _subelement(
        plan_view,
        "geometry",
        s=0.0,
        x=0.0,
        y=0.0,
        hdg=0.0,
        length=ROAD_LENGTH_M,
    )
    _subelement(
        geometry,
        "line",
    )

    elevation_profile = _subelement(
        road,
        "elevationProfile",
    )
    _subelement(
        elevation_profile,
        "elevation",
        s=0.0,
        a=0.0,
        b=0.0,
        c=0.0,
        d=0.0,
    )

    lateral_profile = _subelement(
        road,
        "lateralProfile",
    )
    _subelement(
        lateral_profile,
        "superelevation",
        s=0.0,
        a=0.0,
        b=0.0,
        c=0.0,
        d=0.0,
    )

    lanes = _subelement(
        road,
        "lanes",
    )
    _subelement(
        lanes,
        "laneOffset",
        s=0.0,
        a=0.0,
        b=0.0,
        c=0.0,
        d=0.0,
    )
    lane_section = _subelement(
        lanes,
        "laneSection",
        s=0.0,
    )
    _add_lane_side(
        lane_section,
        "left",
        1,
    )
    center = _subelement(
        lane_section,
        "center",
    )
    _add_center_lane(center)
    _add_lane_side(
        lane_section,
        "right",
        -1,
    )

    return root


def validate_opendrive(
    root: ET.Element,
) -> list[str]:
    errors: list[str] = []

    if root.tag != "OpenDRIVE":
        errors.append(
            "root element must be OpenDRIVE"
        )

    header = root.find("header")
    if header is None:
        errors.append("header is missing")
    else:
        if header.get("name") != MAP_NAME:
            errors.append(
                f"header name must be {MAP_NAME}"
            )
        geo_reference = header.find(
            "geoReference"
        )
        if (
            geo_reference is None
            or not (geo_reference.text or "").strip()
        ):
            errors.append(
                "geoReference must not be empty"
            )

    roads = root.findall("road")
    if len(roads) != 1:
        errors.append(
            "scene 3 must contain exactly one "
            "junction-free expressway road"
        )
        return errors

    road = roads[0]
    if road.get("id") != str(ROAD_ID):
        errors.append(
            f"road id must be {ROAD_ID}"
        )
    if road.get("junction") != "-1":
        errors.append(
            "emergency route must not belong "
            "to a junction"
        )

    try:
        road_length = float(
            road.get("length", "nan")
        )
    except ValueError:
        road_length = math.nan
    if not math.isclose(
        road_length,
        ROAD_LENGTH_M,
        abs_tol=1e-6,
    ):
        errors.append(
            f"road length must be {ROAD_LENGTH_M}"
        )

    geometries = road.findall(
        "./planView/geometry"
    )
    if len(geometries) != 1:
        errors.append(
            "road must contain exactly one geometry"
        )
    elif geometries[0].find("line") is None:
        errors.append(
            "emergency expressway geometry "
            "must be straight"
        )

    if root.findall("junction"):
        errors.append(
            "scene 3 must not contain intersections"
        )

    lane_elements = road.findall(
        "./lanes/laneSection/*/lane"
    )
    lane_by_id: dict[int, ET.Element] = {}
    for lane in lane_elements:
        try:
            lane_id = int(
                lane.get("id", "")
            )
        except ValueError:
            errors.append(
                "lane id must be an integer"
            )
            continue
        if lane_id in lane_by_id:
            errors.append(
                f"duplicate lane id: {lane_id}"
            )
        lane_by_id[lane_id] = lane

    expected_lane_ids = set(range(-5, 6))
    if set(lane_by_id) != expected_lane_ids:
        errors.append(
            "lane ids must be exactly -5 through 5"
        )

    for lane_id in (-3, -2, -1, 1, 2, 3):
        lane = lane_by_id.get(lane_id)
        if lane is None:
            continue
        if lane.get("type") != "driving":
            errors.append(
                f"lane {lane_id} must be driving"
            )
        width = lane.find("width")
        if (
            width is None
            or not math.isclose(
                float(width.get("a", "nan")),
                DRIVING_LANE_WIDTH_M,
                abs_tol=1e-6,
            )
        ):
            errors.append(
                f"lane {lane_id} width must be "
                f"{DRIVING_LANE_WIDTH_M}"
            )

    for lane_id in (-4, 4):
        lane = lane_by_id.get(lane_id)
        if (
            lane is not None
            and lane.get("type") != "shoulder"
        ):
            errors.append(
                f"lane {lane_id} must be shoulder"
            )

    for lane_id in (-5, 5):
        lane = lane_by_id.get(lane_id)
        if (
            lane is not None
            and lane.get("type") != "sidewalk"
        ):
            errors.append(
                f"lane {lane_id} must be sidewalk"
            )

    center_lane = lane_by_id.get(0)
    if (
        center_lane is not None
        and center_lane.get("type") != "none"
    ):
        errors.append(
            "center lane must have type none"
        )

    for event_id, distance_m in (
        EVENT_DISTANCES_M.items()
    ):
        if not 0.0 <= distance_m <= SCENE_LENGTH_M:
            errors.append(
                f"event {event_id} is outside "
                "the 6 km route"
            )

    if not (
        EVENT_DISTANCES_M["cone_taper_start"]
        < EVENT_DISTANCES_M["right_lane_closed"]
        < EVENT_DISTANCES_M["blocked_lane"]
        < EVENT_DISTANCES_M["work_zone_exit"]
    ):
        errors.append(
            "work-zone event order is invalid"
        )

    if not (
        0.0
        < EGO_START_S_M
        < EVENT_DISTANCES_M["cut_in"]
    ):
        errors.append(
            "ego start must precede all hazards"
        )

    return errors


def write_opendrive(
    root: ET.Element,
    output_path: Path,
) -> None:
    ET.indent(
        root,
        space="  ",
    )
    content = ET.tostring(
        root,
        encoding="unicode",
        xml_declaration=True,
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        content + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Generated OpenDRIVE output path",
    )
    return parser


def main(
    argv: list[str] | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    root = build_opendrive()
    errors = validate_opendrive(root)
    if errors:
        raise ValueError(
            "Generated OpenDRIVE validation failed:\n- "
            + "\n- ".join(errors)
        )

    write_opendrive(
        root,
        args.output.expanduser().resolve(),
    )
    print(
        "Wrote emergency OpenDRIVE map to "
        f"{args.output.expanduser().resolve()}"
    )
    print(f"Schema: {SCHEMA_VERSION}")
    print(
        f"Scene route length: "
        f"{SCENE_LENGTH_M:.0f} m"
    )
    print(
        f"Physical road length: "
        f"{ROAD_LENGTH_M:.0f} m"
    )
    print(
        f"Terminal run-off buffer: "
        f"{RUNOFF_LENGTH_M:.0f} m"
    )
    print("Driving lanes: 3 per direction")
    print("Junctions: 0")
    print(
        "Ego start: "
        f"road={EGO_START_ROAD_ID}, "
        f"lane={EGO_START_LANE_ID}, "
        f"s={EGO_START_S_M:.0f} m"
    )
    print("Static validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
