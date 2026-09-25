"""Read-only actor-free geometry audit for the existing Scene 2 event setup."""
import hashlib
import json
from pathlib import Path

from .catalog import CONFIG_ROOT, ConfigError, load_catalog
from .event_fixture import crosswalk_fixture, crosswalk_candidates
from .planning import build_plan
from .route_audit import audit_route
from .truth_capture import prepare_lane_fixture
from .turn_fixture import waypoint_route


def preflight_scene2(output, host='127.0.0.1', port=2000):
    from carla_bootstrap import setup_carla_api
    setup_carla_api()
    import carla
    from scenarios.complex.town05_scene2 import build_repeated_route

    catalog=load_catalog('scene_2')
    path=CONFIG_ROOT/catalog.source_file
    raw=path.read_bytes()
    config=json.loads(raw)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    client=carla.Client(host,port)
    client.set_timeout(60)
    world=client.get_world()
    actors=world.get_actors()
    if actors.filter('vehicle.*') or actors.filter('walker.*') or actors.filter('sensor.*'):
        raise ConfigError('preflight refuses to replace a world containing runtime actors')
    if world.get_map().name.split('/')[-1]!=catalog.map_name:
        world=client.load_world(catalog.map_name)
    carla_map=world.get_map()
    route_config=config['route']
    pairs,distances=build_repeated_route(carla_map,route_config['start_spawn_index'],
        route_config['turnaround_spawn_index'],route_config['target_length_m'],route_config['route_sampling_m'])
    route=waypoint_route([w for w,_ in pairs])
    (output/'route.json').write_text(json.dumps(route,indent=2),encoding='utf-8')
    geometry=audit_route(route)
    plan=build_plan(catalog,'s2_t05_cmd_03')
    crossing=next(e for e in config['special_events'] if e['kind']=='crossing_pedestrian')
    report=dict(source_sha256=hashlib.sha256(raw).hexdigest(),map_name=catalog.map_name,
                route_geometry=geometry,event_dependency_audits=plan['event_dependency_audits'],
                scope='scene2_actor_free_preflight_not_runtime_acceptance',ready=False)
    try:
        fixture=crosswalk_fixture(carla_map,route,int(crossing['crosswalk_polygon_index']),
                                  float(crossing['anchor_progress_m']))
        report['crosswalk']=dict(valid=True,fixture=fixture)
    except ConfigError as error:
        report['crosswalk']=dict(valid=False,reason=str(error))
    report['crosswalk_candidates']=crosswalk_candidates(carla_map,route,float(crossing['anchor_progress_m']))[:8]
    # Check the post-crossing lane-change opportunity without asserting an actor is there.
    anchor=float(crossing['anchor_progress_m'])+25
    point=min(route,key=lambda p:abs(p['distance_m']-anchor))
    try:
        lane=prepare_lane_fixture(carla_map,carla.Location(x=point['x'],y=point['y'],z=point['z']),'LEFT')
        report['post_crossing_lane']=dict(valid=True,route_s_m=point['distance_m'],fixture=lane)
    except ConfigError as error:
        report['post_crossing_lane']=dict(valid=False,route_s_m=point['distance_m'],reason=str(error))
    (output/'preflight.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report
