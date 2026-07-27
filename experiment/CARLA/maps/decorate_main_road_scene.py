"""Load and visibly decorate the VLA main-road CARLA scene.

CARLA renders OpenDRIVE road geometry, but generic XODR objects are often not
materialized as visible meshes in a generated OpenDRIVE world.  This script
therefore draws persistent lane markings and spawns static city props with
CARLA blueprints after the XODR world is loaded.
"""

from __future__ import annotations

import math
import glob
import importlib.util
import os
import re
import sys
from pathlib import Path


def python_abi_tag():
    return "cp{0}{1}".format(sys.version_info.major, sys.version_info.minor)


def setup_carla_api():
    if importlib.util.find_spec("carla") is not None:
        return
    carla_root = os.environ.get("CARLA_ROOT") or r"D:\CARLA_0.9.16"
    if not carla_root:
        raise RuntimeError("Set CARLA_ROOT to your CARLA install directory before running this script.")
    api_dir = Path(carla_root) / "PythonAPI" / "carla"
    dist_dir = api_dir / "dist"
    candidates = []
    candidates.extend(glob.glob(str(dist_dir / "carla-*.egg")))
    candidates.extend(glob.glob(str(dist_dir / "carla-*.whl")))
    if not candidates:
        raise RuntimeError(
            "CARLA Python API was not found. Checked: {0}. "
            "Expected a carla-*.egg or carla-*.whl file there. "
            "Current Python: {1}".format(dist_dir, sys.version.split()[0])
        )
    expected_tag = python_abi_tag()
    matching = [path for path in candidates if expected_tag in Path(path).name]
    if not matching:
        package_tags = []
        for path in candidates:
            match = re.search(r"cp\d+", Path(path).name)
            package_tags.append(match.group(0) if match else Path(path).name)
        raise RuntimeError(
            "Found CARLA Python API packages, but none match current Python ABI {0}. "
            "Available packages: {1}. Current Python: {2}. "
            "Use a matching Python interpreter or install/build a CARLA API wheel for {0}."
            .format(expected_tag, ", ".join(package_tags), sys.version.split()[0])
        )
    sys.path.insert(0, matching[0])
    sys.path.insert(0, str(api_dir))


setup_carla_api()
try:
    import carla
except Exception as exc:
    raise RuntimeError(
        "Found a CARLA API package, but it could not be imported. "
        "This usually means the Python version does not match CARLA. "
        "Current Python: {0}. Try running this script with the Python "
        "version matching the file under CARLA_ROOT/PythonAPI/carla/dist."
        .format(sys.version.split()[0])
    ) from exc


ROAD_LENGTH_M = 5000.0
LANE_WIDTH_M = 3.5
LANES_PER_DIRECTION = 3
SHOULDER_WIDTH_M = 1.0
SIDEWALK_WIDTH_M = 2.0
RIGHT_TURN_START_S = 1200.0
RIGHT_TURN_RADIUS_M = 80.0
RIGHT_TURN_LENGTH_M = 125.66370614359172
XODR_PATH = Path(__file__).resolve().parent / "maps" / "output" / "VLA_MainRoad_5km.xodr"

WHITE = carla.Color(205, 205, 195)
YELLOW = carla.Color(220, 180, 55)
CURB = carla.Color(120, 120, 115)
SIDEWALK_GRAY = carla.Color(95, 95, 90)
ASPHALT_GRAY = carla.Color(45, 45, 45)
BUILDING_GRAY = carla.Color(95, 100, 105)


def pose_at_s(s):
    return float(s), 0.0, 0.0


def location_at(s, t, z=0.08):
    x, y, hdg = pose_at_s(float(s))
    nx = -math.sin(hdg)
    ny = math.cos(hdg)
    return carla.Location(x=x + nx * t, y=y + ny * t, z=z)


def rotation_at(s, yaw_offset=0.0):
    _, _, hdg = pose_at_s(float(s))
    return carla.Rotation(yaw=math.degrees(hdg + yaw_offset))


def draw_solid_line(world, t, color, width=0.055, step=6.0):
    s = 0.0
    while s < ROAD_LENGTH_M:
        end_s = min(s + step, ROAD_LENGTH_M)
        world.debug.draw_line(
            location_at(s, t),
            location_at(end_s, t),
            thickness=width,
            color=color,
            life_time=0.0,
        )
        s = end_s


def draw_broken_line(world, t, color=WHITE, dash=5.5, gap=8.5, width=0.045):
    s = 0.0
    while s < ROAD_LENGTH_M:
        end_s = min(s + dash, ROAD_LENGTH_M)
        world.debug.draw_line(
            location_at(s, t),
            location_at(end_s, t),
            thickness=width,
            color=color,
            life_time=0.0,
        )
        s += dash + gap


def draw_stop_line(world, s, t_center, width):
    world.debug.draw_line(
        location_at(s, t_center - width / 2.0, 0.1),
        location_at(s, t_center + width / 2.0, 0.1),
        thickness=0.12,
        color=WHITE,
        life_time=0.0,
    )


def draw_crosswalk(world, s, t_center, width):
    t0 = t_center - width / 2.0
    for i in range(8):
        t = t0 + i * width / 7.0
        world.debug.draw_line(
            location_at(s - 5.0, t, 0.11),
            location_at(s + 5.0, t, 0.11),
            thickness=0.08,
            color=WHITE,
            life_time=0.0,
        )


def first_blueprint(blueprints, patterns):
    for pattern in patterns:
        matches = blueprints.filter(pattern)
        if matches:
            return matches[0]
    return None


def try_spawn(world, blueprint, transform):
    if blueprint is None:
        return None
    try:
        return world.try_spawn_actor(blueprint, transform)
    except RuntimeError:
        return None


def draw_urban_surfaces(world, road_edge, sidewalk_edge):
    # Low-contrast surface hints: enough to read as sidewalks and a cross road,
    # without the bright debug-strip look.
    for t in (road_edge + SIDEWALK_WIDTH_M * 0.5, -(road_edge + SIDEWALK_WIDTH_M * 0.5)):
        s = 0.0
        while s < ROAD_LENGTH_M:
            world.debug.draw_line(
                location_at(s, t, 0.035),
                location_at(min(s + 14.0, ROAD_LENGTH_M), t, 0.035),
                thickness=1.35,
                color=SIDEWALK_GRAY,
                life_time=0.0,
            )
            s += 16.0

    for s in range(int(RIGHT_TURN_START_S - 18.0), int(RIGHT_TURN_START_S + 19.0), 6):
        world.debug.draw_line(
            location_at(float(s), -sidewalk_edge, 0.04),
            location_at(float(s), sidewalk_edge, 0.04),
            thickness=0.9,
            color=ASPHALT_GRAY,
            life_time=0.0,
        )


def draw_recessed_blocks(world, sidewalk_edge):
    # Neutral massing behind the sidewalks.  These are visual anchors only;
    # actual CARLA props are spawned below when the package has matching assets.
    for s in range(120, int(ROAD_LENGTH_M), 180):
        for row, setback in enumerate((7.0, 14.0)):
            for side in (1.0, -1.0):
                t = side * (sidewalk_edge + setback)
                yaw_offset = 0.0 if side > 0 else math.pi
                for offset, height, length, depth in (
                    (-55.0, 10.0 + row * 4.0, 24.0, 4.0),
                    (-18.0, 16.0 + row * 3.0, 30.0, 4.5),
                    (24.0, 12.5 + row * 5.0, 26.0, 4.0),
                    (58.0, 20.0 + row * 2.0, 20.0, 4.2),
                ):
                    block_s = max(0.0, min(ROAD_LENGTH_M, s + offset))
                    center = location_at(block_s, t, height * 0.5)
                    extent = carla.Vector3D(length * 0.5, depth, height * 0.5)
                    world.debug.draw_box(
                        carla.BoundingBox(center, extent),
                        rotation_at(block_s, yaw_offset),
                        thickness=0.07,
                        color=BUILDING_GRAY,
                        life_time=0.0,
                    )
                    for floor in range(1, int(height // 3.0)):
                        z = floor * 3.0
                        world.debug.draw_line(
                            location_at(block_s - length * 0.45, t, z),
                            location_at(block_s + length * 0.45, t, z),
                            thickness=0.025,
                            color=carla.Color(135, 140, 145),
                            life_time=0.0,
                        )


def decorate(world):
    half_drive = LANES_PER_DIRECTION * LANE_WIDTH_M
    road_edge = half_drive + SHOULDER_WIDTH_M
    sidewalk_edge = road_edge + SIDEWALK_WIDTH_M

    draw_urban_surfaces(world, road_edge, sidewalk_edge)

    # Visible road markings: center double yellow, lane dividers, road edges.
    draw_solid_line(world, -0.16, YELLOW, width=0.055)
    draw_solid_line(world, 0.16, YELLOW, width=0.055)
    for t in (-LANE_WIDTH_M, -2 * LANE_WIDTH_M, LANE_WIDTH_M, 2 * LANE_WIDTH_M):
        draw_broken_line(world, t)
    draw_solid_line(world, -half_drive, WHITE, width=0.045)
    draw_solid_line(world, half_drive, WHITE, width=0.045)
    draw_solid_line(world, -road_edge, CURB, width=0.075)
    draw_solid_line(world, road_edge, CURB, width=0.075)

    # Intersection markings at 1200 m.
    lane_area = half_drive
    draw_stop_line(world, RIGHT_TURN_START_S - 18.0, -lane_area / 2.0, lane_area)
    draw_stop_line(world, RIGHT_TURN_START_S - 18.0, lane_area / 2.0, lane_area)
    draw_crosswalk(world, RIGHT_TURN_START_S - 8.0, -lane_area / 2.0, lane_area)
    draw_crosswalk(world, RIGHT_TURN_START_S - 8.0, lane_area / 2.0, lane_area)
    blueprints = world.get_blueprint_library()
    light_bp = first_blueprint(blueprints, [
        "static.prop.streetlight*",
        "static.prop.light*",
        "static.prop.lamp*",
    ])
    sign_bp = first_blueprint(blueprints, [
        "static.prop.trafficwarning*",
        "static.prop.streetsign*",
        "static.prop.sign*",
    ])
    cone_bp = first_blueprint(blueprints, [
        "static.prop.trafficcone*",
        "static.prop.constructioncone*",
    ])
    building_bp = first_blueprint(blueprints, [
        "*building*",
        "*Building*",
        "*house*",
        "*House*",
        "*apartment*",
        "*shop*",
        "static.prop.building*",
        "static.prop.container*",
        "static.prop.kiosk*",
        "static.prop.busstop*",
    ])
    tree_bp = first_blueprint(blueprints, [
        "static.prop.tree*",
        "static.prop.plant*",
    ])
    bench_bp = first_blueprint(blueprints, [
        "static.prop.bench*",
        "static.prop.busstop*",
    ])

    draw_recessed_blocks(world, sidewalk_edge)

    spawned = []
    for s in range(120, int(ROAD_LENGTH_M), 120):
        for t, yaw_offset in ((sidewalk_edge + 0.8, 0.0), (-(sidewalk_edge + 0.8), math.pi)):
            actor = try_spawn(
                world,
                light_bp,
                carla.Transform(location_at(s, t, 0.0), rotation_at(s, yaw_offset)),
            )
            if actor:
                spawned.append(actor)

    for s in (300, 900, 1190, 1800, 3000, 4200):
        for t, yaw_offset in ((sidewalk_edge + 0.5, 0.0), (-(sidewalk_edge + 0.5), math.pi)):
            actor = try_spawn(
                world,
                sign_bp,
                carla.Transform(location_at(s, t, 0.0), rotation_at(s, yaw_offset)),
            )
            if actor:
                spawned.append(actor)

    for s in range(160, int(ROAD_LENGTH_M), 220):
        for t in (sidewalk_edge + 8.0, -(sidewalk_edge + 8.0)):
            actor = try_spawn(
                world,
                building_bp,
                carla.Transform(location_at(s, t, 0.0), rotation_at(s)),
            )
            if actor:
                spawned.append(actor)

    for s in range(160, int(ROAD_LENGTH_M), 160):
        for t in (sidewalk_edge + 1.7, -(sidewalk_edge + 1.7)):
            actor = try_spawn(
                world,
                tree_bp,
                carla.Transform(location_at(s, t, 0.0), rotation_at(s)),
            )
            if actor:
                spawned.append(actor)

    for s in range(500, int(ROAD_LENGTH_M), 900):
        for t, yaw_offset in ((sidewalk_edge + 0.4, 0.0), (-(sidewalk_edge + 0.4), math.pi)):
            actor = try_spawn(
                world,
                bench_bp,
                carla.Transform(location_at(s, t, 0.0), rotation_at(s, yaw_offset)),
            )
            if actor:
                spawned.append(actor)

    for t in (-road_edge, road_edge):
        for s in range(int(RIGHT_TURN_START_S - 40), int(RIGHT_TURN_START_S + 41), 10):
            actor = try_spawn(
                world,
                cone_bp,
                carla.Transform(location_at(s, t, 0.0), rotation_at(s)),
            )
            if actor:
                spawned.append(actor)

    return spawned


def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)
    world = client.generate_opendrive_world(XODR_PATH.read_text(encoding="utf-8"))
    actors = decorate(world)
    print("Loaded:", XODR_PATH)
    print("Visible scene decoration complete")
    print("Spawned static actors:", len(actors))
    print("Persistent debug markings: center lines, lane dividers, road edges, stop lines, crosswalks")


if __name__ == "__main__":
    main()
