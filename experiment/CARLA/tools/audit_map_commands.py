"""Read-only command/environment audit using the current diagnostic route geometry."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'experiment/CARLA')]


def audit_scene(world, scene, carla):
    from benchmark.catalog import load_catalog
    from benchmark.speed_limit_audit import capture_speed_signs
    catalog = load_catalog(scene)
    geometry_file = ROOT / 'experiment/CARLA/configs/diagnostic_sections' / scene / 'geometry.json'
    geometry = json.loads(geometry_file.read_text())
    route = [dict(p, distance_m=p['s_m']) for p in geometry]
    signs = capture_speed_signs(world, route, carla.Location)
    changes = sorted((start, sign['sign_speed_kmh'], sign['actor_id'])
        for sign in signs['signs'] for start, _ in sign['route_trigger_ranges_m'])
    rows = []
    world_map = world.get_map()
    for index, task in enumerate(catalog.tasks):
        later = [t.activate_m for t in catalog.tasks if t.activate_m > task.activate_m]
        end = task.end_m or (min(later) if later else catalog.route_length_m)
        start = task.activate_m
        active = [item for item in changes if item[0] <= start]
        initial_limit = active[-1][1] if active else 30.
        minimum_limit = min([initial_limit] + [speed for s, speed, _ in changes if start < s <= end])
        samples = [p for p in geometry if start <= p['s_m'] <= end and not p['is_junction']]
        opportunities = {'LEFT': [], 'RIGHT': []}
        for point in samples:
            wp = world_map.get_waypoint(carla.Location(**{k: point[k] for k in ('x','y','z')}))
            for direction, method in [('LEFT','get_left_lane'), ('RIGHT','get_right_lane')]:
                adjacent = getattr(wp, method)()
                permitted = str(wp.lane_change).lower()
                if (adjacent is not None and adjacent.lane_type == carla.LaneType.Driving
                        and adjacent.lane_id * wp.lane_id > 0
                        and (direction.lower() in permitted or 'both' in permitted)):
                    opportunities[direction].append(point['s_m'])
        source = task.source_command
        rows.append(dict(task_id=task.task_id, instruction=task.instruction,
            activate_m=start, end_m=end, initial_limit_kmh=initial_limit,
            conservative_interval_limit_kmh=minimum_limit,
            explicit_target_speed_kmh=source.get('target_speed_kmh'),
            lane_change_opportunity_m={k: ([min(v),max(v)] if v else None) for k,v in opportunities.items()},
            requested_steps=source.get('steps', source.get('structured_command', source.get('semantic_goal')))))
    return dict(scene_id=scene, source_sha256=catalog.source_sha256,
        geometry_sha256=hashlib.sha256(geometry_file.read_bytes()).hexdigest(),
        speed_sign_evidence=signs, speed_trigger_changes=changes, tasks=rows,
        scope='static_route_sign_and_lane_opportunities; runtime limit and dynamic hazards remain authoritative')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from carla_bootstrap import setup_carla_api
    setup_carla_api()
    import carla
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(60.)
    world = client.get_world()
    if world.get_map().name.split('/')[-1] != 'Town05_Opt':
        raise ValueError('Current route geometry requires Town05_Opt')
    reports = [audit_scene(world, scene, carla) for scene in ('scene_1','scene_2','scene_3')]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for report in reports:
        print(report['scene_id'])
        for row in report['tasks']:
            print(row['task_id'], row['activate_m'], row['conservative_interval_limit_kmh'],
                  row['lane_change_opportunity_m'], row['requested_steps'])


if __name__ == '__main__':
    main()
