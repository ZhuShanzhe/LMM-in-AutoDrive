"""Benchmark planning, offline assessment and explicitly scoped runtime fixtures."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from .catalog import CONFIG_ROOT, SCENES, ConfigError, load_catalog, validate_catalog
from .planning import build_plan
from .task_oracle import TaskOracle
from .route_audit import audit_route


def main(argv=None):
    parser = argparse.ArgumentParser(description='CARLA benchmark task inventory and offline checks')
    parser.add_argument('command', choices=('list', 'validate', 'manifest', 'plan', 'evaluate', 'audit-route', 'run', 'preflight', 'overtake-geometry'))
    parser.add_argument('--scene', choices=tuple(SCENES))
    parser.add_argument('--task', default='all', help='activation-order number or stable source ID')
    parser.add_argument('--config-root', type=Path, default=CONFIG_ROOT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--require-ready', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--pre-roll-m', type=float, default=150)
    parser.add_argument('--spec', type=Path)
    parser.add_argument('--observations', type=Path)
    parser.add_argument('--route', type=Path)
    parser.add_argument('--traffic-config', type=Path)
    parser.add_argument('--map-snapshot', type=Path, help='Offline map/spawn snapshot for scene_3 preflight')
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--duration-s', type=float, default=15)
    parser.add_argument('--corridor-length-m', type=float, default=280,
                        help='Required pass-and-return geometry length, not a task completion threshold')
    args = parser.parse_args(argv)
    try:
        if args.command=='overtake-geometry':
            if not args.scene or args.route is None or args.map_snapshot is None or args.output is None:
                raise ConfigError('overtake-geometry requires scene, task, route, map-snapshot and output')
            catalog=load_catalog(args.scene,args.config_root)
            tasks=catalog.select(args.task)
            if len(tasks)!=1:
                raise ConfigError('select one overtake task')
            from .task_oracle import load_profile
            profile=load_profile(catalog,tasks[0])
            if not profile or not any(s['kind']=='overtake' for s in profile['steps']):
                raise ConfigError('selected task has no bound overtake criterion')
            if args.output.exists():
                raise ConfigError('output already exists')
            from carla_bootstrap import setup_carla_api
            setup_carla_api()
            import carla
            from .map_snapshot import load_map_snapshot
            from .overtake_geometry import build_overtake_geometry
            world_map=load_map_snapshot(args.map_snapshot,carla,catalog.map_name)
            payload=args.route.read_bytes()
            try:
                result=build_overtake_geometry(world_map,json.loads(payload),carla.Location,
                    preferred_m=profile['activate_m'],length_m=args.corridor_length_m)
            except ConfigError as error:
                result=dict(geometry_ready=False,execution_supported=False,reason=str(error))
            result['required_length_m']=args.corridor_length_m
            result.update(task_id=tasks[0].task_id,source_sha256=catalog.source_sha256,
                route_sha256=hashlib.sha256(payload).hexdigest(),
                map_sha256=hashlib.sha256(world_map.to_opendrive().encode('utf-8')).hexdigest())
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
            print(json.dumps({k:v for k,v in result.items() if k not in
                {'route','target_lane_route','lane_pairs','rejected_candidates'}},ensure_ascii=True))
            return 0 if result['geometry_ready'] else 1
        if args.map_snapshot is not None and (args.command!='preflight' or args.scene!='scene_3'):
            raise ConfigError('--map-snapshot currently requires preflight --scene scene_3')
        if args.command == 'preflight':
            if args.scene not in {'scene_1','scene_2','scene_3'} or args.output is None:
                raise ConfigError('preflight requires --scene and fresh --output directory')
            if args.scene=='scene_1':
                from .scene1_preflight import preflight_scene1
                result=preflight_scene1(args.output,args.host,args.port)
            elif args.scene=='scene_2':
                from .scene2_preflight import preflight_scene2
                result=preflight_scene2(args.output,args.host,args.port)
            else:
                from .scene3_preflight import preflight_scene3
                result=preflight_scene3(args.output,args.host,args.port,args.map_snapshot)
            print(json.dumps({k:v for k,v in result.items() if k!='route_geometry'},ensure_ascii=True))
            return 0 if result['ready'] else 1
        if args.command == 'run':
            if not args.scene or args.route is None or args.traffic_config is None or args.output is None:
                raise ConfigError('run requires --scene, --task, --route, --traffic-config and a fresh --output directory')
            if args.config_root != CONFIG_ROOT:
                raise ConfigError('run requires the registered config root')
            from .runtime import run_fixture
            result=run_fixture(args.scene,args.task,args.route,args.traffic_config,args.output,
                                     host=args.host,port=args.port,seconds=args.duration_s,seed=args.seed)
            print(json.dumps({k:result[k] for k in
                              ('task_feedback','run_valid','episode_safety','cleanup_completed')},ensure_ascii=True))
            return 0 if (result['run_valid'] and result['task_feedback']['status']=='SUCCESS'
                         and result['episode_safety']['status']=='NO_RECORDED_VIOLATION') else 1
        if args.command == 'audit-route':
            if args.route is None:
                raise ConfigError('audit-route requires --route')
            payload = args.route.read_bytes()
            result = audit_route(json.loads(payload))
            result['route_sha256'] = hashlib.sha256(payload).hexdigest()
        elif args.command == 'evaluate':
            if args.spec is None or args.observations is None:
                raise ConfigError('evaluate requires --spec and --observations')
            oracle = TaskOracle(json.loads(args.spec.read_text(encoding='utf-8')))
            with args.observations.open(encoding='utf-8') as stream:
                for line in stream:
                    oracle.update(json.loads(line))
                    if oracle.status in {'SUCCESS', 'FAILURE', 'SCENE_INVALID', 'TIMEOUT'}:
                        break
            result = oracle.end_of_stream()
            result['scope'] = 'offline_task_assessment_not_episode_safety'
            result['spec_sha256'] = hashlib.sha256(args.spec.read_bytes()).hexdigest()
            with args.observations.open('rb') as stream:
                result['observations_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
        elif args.scene is None:
            raise ConfigError('--scene required')
        else:
            catalog = load_catalog(args.scene, args.config_root)
            selected = catalog.select(args.task)
        if args.command == 'plan':
            result = build_plan(catalog, args.task, args.seed, args.pre_roll_m)
        elif args.command == 'validate':
            result = validate_catalog(catalog)
        elif args.command in {'list', 'manifest'}:
            result = catalog.to_dict()
            selected_ids = {t.task_id for t in selected}
            result['tasks'] = [t for t in result['tasks'] if t['task_id'] in selected_ids]
            result['selection'] = args.task
            result['execution_supported'] = False
        content = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(content, encoding='utf-8')
        else:
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8')
            print(content, end='')
        if args.command == 'validate':
            return 0 if result['config_valid'] and (not args.require_ready or result['benchmark_ready']) else 1
        if args.command == 'evaluate':
            return 0 if result['status'] == 'SUCCESS' else 1
        return 0
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
