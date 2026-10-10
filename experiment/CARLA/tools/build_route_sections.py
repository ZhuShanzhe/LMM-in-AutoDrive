"""Build task/event/geometry diagnostic sections from current CARLA routes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'experiment/CARLA'))

from benchmark.catalog import load_catalog, load_episode_catalog
from benchmark.route_sections import partition


def lane_count(waypoint):
    count = 1
    seen = {(waypoint.road_id, waypoint.lane_id)}
    for method in ('get_left_lane', 'get_right_lane'):
        current = waypoint
        for _ in range(12):
            candidate = getattr(current, method)()
            if candidate is None or 'driving' not in str(candidate.lane_type).lower():
                break
            key = (candidate.road_id, candidate.lane_id)
            if key in seen or candidate.lane_id * waypoint.lane_id <= 0:
                break
            seen.add(key)
            count += 1
            current = candidate
    return count


def sample_waypoint(waypoint, s):
    transform = waypoint.transform
    location = transform.location
    return dict(s_m=round(float(s), 3), x=float(location.x), y=float(location.y),
        z=float(location.z), yaw_deg=float(transform.rotation.yaw),
        road_id=int(waypoint.road_id), lane_id=int(waypoint.lane_id),
        is_junction=bool(waypoint.is_junction), lane_width_m=float(waypoint.lane_width),
        same_direction_lanes=lane_count(waypoint))


def coalesce_samples(samples):
    result = []
    for point in samples:
        if result and abs(point['s_m'] - result[-1]['s_m']) <= .001:
            previous = result[-1]
            distance2 = sum((point[k]-previous[k])**2 for k in ('x','y','z'))
            if distance2 > .1**2:
                raise ValueError('Same route progress has inconsistent physical positions')
            result[-1] = point
        else:
            result.append(point)
    return result


def collect_routes(world, output):
    import carla
    from continuous.task_geometry_binding import bind_task_geometry
    from scenarios.basic.voice_control_5km import BasicVoiceControl5KmScenario
    from scenarios.complex.town05_scene2 import build_configured_route
    from scene3_town05_route import build_town05_route_context
    from run_emergency_response_6km import build_ego_route_plan

    world_map = world.get_map()
    result = {}
    bound_path = output/'scene_1_bound_config.json'
    source = ROOT/'experiment/CARLA/configs/basic_voice_urban_5km.json'
    try:
        if not bound_path.is_file():
            raise ValueError('No cached binding')
        load_episode_catalog('scene_1', bound_path, world_map)
    except ValueError:
        bind_task_geometry(world, source, bound_path)
    scenario = BasicVoiceControl5KmScenario(world, config_path=str(bound_path))
    directives = [dict(c['route_directive'], id=c['id']) for c in scenario.config['commands']
                  if c.get('route_directive')]
    scenario._select_spawn_point_and_route(directives)
    samples = []
    for point in scenario.route:
        waypoint = world_map.get_waypoint(carla.Location(x=point['x'], y=point['y'], z=point['z']))
        sample = sample_waypoint(waypoint, point['distance_m'])
        sample.update(x=point['x'], y=point['y'], z=point['z'], yaw_deg=point['yaw'])
        samples.append(sample)
    result['scene_1'] = (load_episode_catalog('scene_1', bound_path, world_map), samples)

    catalog = load_catalog('scene_2')
    config = json.loads((ROOT/'experiment/CARLA/configs'/catalog.source_file).read_text())
    route, distances, _, _ = build_configured_route(world_map, config['route'])
    result['scene_2'] = (catalog, [sample_waypoint(wp, s) for (wp, _), s in zip(route, distances)])

    catalog = load_catalog('scene_3')
    config = json.loads((ROOT/'experiment/CARLA/configs'/catalog.source_file).read_text())
    context = build_town05_route_context(world_map, config['map']['route'])
    plan = build_ego_route_plan(context, [])
    result['scene_3'] = (catalog, [sample_waypoint(wp, s) for (wp, _), s in zip(plan, context.distances_m)])
    for scene, (catalog, samples) in result.items():
        # Route builders may include the finish lookahead; clip only diagnostic samples.
        result[scene] = (catalog, coalesce_samples([p for p in samples if p['s_m'] <= catalog.route_length_m]))
    stops_by_road = {}
    for light in world.get_actors().filter('traffic.traffic_light'):
        for waypoint in light.get_stop_waypoints():
            stops_by_road.setdefault(waypoint.road_id, []).append((light.id, waypoint))
    for _, samples in result.values():
        for point in samples:
            point['traffic_signal_ids'] = sorted({identity for identity, waypoint
                in stops_by_road.get(point['road_id'], [])
                if (point['x']-waypoint.transform.location.x)**2
                    + (point['y']-waypoint.transform.location.y)**2 <= 12.**2})
    return result


def historical_failures(scene):
    root = ROOT/f'test/10_02/whole_routes_001/{scene.replace("scene_", "scene")}/github_logs'
    result = []
    for path in sorted(root.glob('segment_*/interruption.json')):
        document = json.loads(path.read_text(encoding='utf-8'))
        summary = document.get('last_20_sim_seconds', {})
        result.append(dict(s_m=document.get('route_progress_m'),
            source=path.relative_to(ROOT).as_posix(), stop_reason=document.get('stop_reason'),
            decision_reasons=summary.get('decision_reasons', [])[:3],
            blocked_codes=summary.get('blocked_codes', [])[:3],
            route_equivalence='not_verified'))
    return result


def write_scene(output, report, samples):
    scene = report['scene_id']
    folder = output/scene
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'sections.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (folder/'geometry.json').write_text(json.dumps(samples, indent=2)+'\n', encoding='utf-8')
    fields = ['section_id','start_m','end_m','replay_start_m','stop_after_m','task_ids','event_ids',
              'geometry_kinds','historical_failure_count']
    with (folder/'sections.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for section in report['sections']:
            row = {key:section[key] for key in fields if key != 'historical_failure_count'}
            for key in ('task_ids','event_ids','geometry_kinds'):
                row[key] = ';'.join(row[key])
            row['historical_failure_count'] = len(section['historical_failures'])
            writer.writerow(row)
    lines = [f'# {scene} 分段索引', '',
        f'地图：`{report["map_name"]}`；路线：{report["route_length_m"]:.0f} 米；'
        f'{len(report["phases"])} 个任务主段，{len(report["sections"])} 个道路/事件细段。', '',
        '里程是沿本场景路线的累计里程，不是地图坐标。场景一使用当前配置重新绑定后的任务位置。', '',
        '## 任务主段', '', '| 主段 | 核心里程（米） | 包含细段 |', '| --- | --- | --- |']
    for phase in report['phases']:
        lines.append(f'| {phase["phase_id"]} | {phase["start_m"]:.1f}–{phase["end_m"]:.1f} | '
                     f'{", ".join(phase["section_ids"])} |')
    lines.extend(['', '## 道路与事件细段', '',
        '| 测试段 | 核心里程（米） | 带前置上下文的回放范围（米） | 任务 | 事件 | 道路特征 | 历史故障数 |',
        '| --- | --- | --- | --- | --- | --- | --- |'])
    for s in report['sections']:
        lines.append(f'| {s["section_id"]} | {s["start_m"]:.1f}–{s["end_m"]:.1f} | '
            f'{s["replay_start_m"]:.1f}–{s["stop_after_m"]:.1f} | '
            f'{", ".join(s["task_ids"]) or "巡航"} | {", ".join(s["event_ids"]) or "—"} | '
            f'{", ".join(s["geometry_kinds"]) or "普通路段"} | {len(s["historical_failures"])} |')
    lines.extend(['', '## 任务原文', ''])
    for task in report['tasks']:
        lines.append(f'- `{task["task_id"]}`：{task["instruction"]}（触发 {task["activate_m"]:.1f} 米）')
    (folder/'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--maximum-section-m', type=float, default=400.)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from carla_bootstrap import setup_carla_api
    setup_carla_api()
    import carla
    client = carla.Client(args.host, args.port)
    client.set_timeout(90.)
    world = client.get_world()
    if not world.get_map().name.endswith('Town05_Opt'):
        world = client.load_world('Town05_Opt')
    map_hash = hashlib.sha256(world.get_map().to_opendrive().encode()).hexdigest()
    routes = collect_routes(world, args.output)
    summary = dict(schema_version='diagnostic_route_index/1.0', map_name=world.get_map().name,
                   map_sha256=map_hash, scenes={}, scenario_modified=False, weights_modified=False)
    for scene, (catalog, samples) in routes.items():
        report = partition(catalog, samples, maximum_section_m=args.maximum_section_m,
                           failures=historical_failures(scene))
        report['map_sha256'] = map_hash
        report['route_geometry_sha256'] = hashlib.sha256(json.dumps(samples, sort_keys=True).encode()).hexdigest()
        report['geometry_source'] = 'CARLA official map via production route constructors'
        runtime_source = (args.output/'scene_1_bound_config.json' if scene=='scene_1'
                          else ROOT/'experiment/CARLA/configs'/catalog.source_file)
        runtime = json.loads(runtime_source.read_text(encoding='utf-8'))
        report['environment_context'] = {k:runtime[k] for k in
            ('environment', 'weather', 'surface_and_visibility', 'lane_contract') if k in runtime}
        if scene == 'scene_1':
            report['runtime_source_file'] = '../scene_1_bound_config.json'
        write_scene(args.output, report, samples)
        summary['scenes'][scene] = dict(route_length_m=catalog.route_length_m,
            phase_count=len(report['phases']), section_count=len(report['sections']), task_count=len(report['tasks']),
            event_count=len(report['events']), geometry_features=len(report['geometry_features']),
            source_sha256=catalog.source_sha256)
    (args.output/'index.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
