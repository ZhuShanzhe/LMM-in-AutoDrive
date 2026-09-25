"""Actor-free checks using the formal Scene 1 route construction."""
import hashlib
import json
import math
from pathlib import Path

from .catalog import CONFIG_ROOT, ConfigError, load_catalog
from .lane_continuity import build_lane_corridor
from .turn_fixture import bind_route_turn
from .speed_limit_audit import capture_speed_signs


def preflight_scene1(output,host='127.0.0.1',port=2000):
    from carla_bootstrap import setup_carla_api
    setup_carla_api()
    import carla
    from scenarios.basic.voice_control_5km import BasicVoiceControl5KmScenario
    catalog=load_catalog('scene_1')
    client=carla.Client(host,port)
    client.set_timeout(60)
    world=client.get_world()
    def actor_count(world):
        actors=world.get_actors()
        return sum(len(actors.filter(pattern)) for pattern in ('vehicle.*','walker.*','sensor.*'))
    if actor_count(world):
        raise ConfigError('active actors present; refusing map replacement')
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    if world.get_map().name.split('/')[-1]!=catalog.map_name:
        world=client.load_world(catalog.map_name)
    world_map=world.get_map()
    xml=world_map.to_opendrive().encode('utf-8')
    (output/'map.xodr').write_bytes(xml)
    report=dict(scope='source_route_geometry_preflight_not_driving_acceptance',
                source_sha256=catalog.source_sha256,map_geometry_sha256=hashlib.sha256(xml).hexdigest(),
                issues=[],ready=False)
    scenario=None
    try:
        scenario=BasicVoiceControl5KmScenario(world,config_path=str(CONFIG_ROOT/catalog.source_file))
        directives=[dict(c['route_directive'],id=c['id']) for c in scenario.config['commands']
                    if c.get('route_directive')]
        scenario._select_spawn_point_and_route(directives)
        route=scenario.route
        payload=json.dumps(route,indent=2).encode('utf-8')
        (output/'route.json').write_bytes(payload)
        report['speed_sign_evidence']=capture_speed_signs(world,route,carla.Location)
        report.update(route_sha256=hashlib.sha256(payload).hexdigest(),
                      nominal_route_length_m=route[-1]['distance_m'],
                      geometric_route_length_m=sum(math.sqrt(sum((b[k]-a[k])**2 for k in ('x','y','z')))
                                                  for a,b in zip(route,route[1:])),
                      source_preflight=scenario.route_preflight)
        if min(report['nominal_route_length_m'],report['geometric_route_length_m'])<catalog.route_length_m:
            report['issues'].append('route_shorter_than_required_distance')
        report['task_turn_bindings']={}
        for task in catalog.tasks:
            action=task.source_command.get('action')
            if action not in ('turn_left','turn_right'):
                continue
            end=min((t.announce_m for t in catalog.tasks if t.announce_m>task.activate_m),
                    default=catalog.route_length_m)
            try:
                report['task_turn_bindings'][task.task_id]=bind_route_turn(
                    route,'LEFT' if action=='turn_left' else 'RIGHT',task.activate_m,end)
            except ConfigError as error:
                report['issues'].append(task.task_id+': '+str(error))
        task=next(t for t in catalog.tasks if t.task_id=='c15_keep_to_goal')
        try:
            report['destination_lane_corridor']=build_lane_corridor(world_map,route,task.activate_m,carla.Location)
        except ConfigError as error:
            report['issues'].append('destination_lane_corridor: '+str(error))
    except (ValueError,RuntimeError) as error:
        report['issues'].append('source_route_generation: '+str(error))
        if scenario is not None:
            manager=scenario.route_manager
            payload=json.dumps(manager.route,indent=2).encode('utf-8')
            (output/'rejected_route.json').write_bytes(payload)
            report['rejected_candidate']=dict(route_sha256=hashlib.sha256(payload).hexdigest(),
                route_length_m=manager.route_length_m,applied_directives=manager.applied_directives,
                unapplied_directives=manager.unapplied_directives,
                scope='last_rejected_candidate_not_accepted_route')
    report['remaining_dynamic_actor_count']=actor_count(world)
    if report['remaining_dynamic_actor_count']:
        report['issues'].append('unexpected_dynamic_actors')
    report['ready']=not report['issues']
    (output/'preflight.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report
