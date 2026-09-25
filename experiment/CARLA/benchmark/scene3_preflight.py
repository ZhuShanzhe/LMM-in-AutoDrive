"""Actor-free geometry preflight using the formal scene-three route adapter."""
import hashlib
import json
import math
from pathlib import Path

from .catalog import CONFIG_ROOT, ConfigError, load_catalog
from .route_audit import audit_route
from .turn_fixture import waypoint_route


def audit_worker_endpoints(adapter, event):
    results=[]
    for worker in event.get('workers', []):
        if worker['start_lane_id']==worker['destination_lane_id']:
            continue
        anchor=float(worker['start_s_m'])
        lane=adapter.route_waypoint(anchor)
        start=adapter.get_waypoint_xodr(1,int(worker['start_lane_id']),anchor)
        end=adapter.get_waypoint_xodr(1,int(worker['destination_lane_id']),anchor)
        if start is None or end is None or lane is None:
            results.append(dict(role=worker['role_name'],valid=False,reason='missing_endpoint_lane'))
            continue
        origin=lane.transform.location
        yaw=math.radians(lane.transform.rotation.yaw)
        def lateral(waypoint):
            point=waypoint.transform.location
            return -(point.x-origin.x)*math.sin(yaw)+(point.y-origin.y)*math.cos(yaw)
        a,b=lateral(start),lateral(end)
        half=float(lane.lane_width)/2
        valid=half>0 and all(math.isfinite(v) for v in (a,b,half)) and abs(a)>half and abs(b)>half and a*b<0
        results.append(dict(role=worker['role_name'],anchor_m=anchor,valid=valid,
            start_lateral_m=a,end_lateral_m=b,route_lane_width_m=half*2,
            reason=None if valid else 'endpoints_do_not_fully_cross_and_clear_route_lane',
            scope='endpoint centers only; actor extent, walkability and traffic not verified'))
    return results


def preflight_scene3(output, host='127.0.0.1', port=2000, map_snapshot=None):
    from carla_bootstrap import setup_carla_api
    setup_carla_api()
    import carla
    from scene3_town05_route import build_town05_route_context, validate_scene3_event_anchors
    catalog=load_catalog('scene_3')
    raw=(CONFIG_ROOT/catalog.source_file).read_bytes()
    config=json.loads(raw)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    from .map_snapshot import load_map_snapshot, save_map_snapshot
    if map_snapshot is not None:
        world_map=load_map_snapshot(map_snapshot,carla,catalog.map_name)
    else:
        client=carla.Client(host,port)
        client.set_timeout(60)
        world=client.get_world()
        actors=world.get_actors()
        if any(actors.filter(pattern) for pattern in ('vehicle.*','walker.*','sensor.*')):
            raise ConfigError('preflight refuses to replace a world containing runtime actors')
        if world.get_map().name.split('/')[-1]!=catalog.map_name:
            world=client.load_world(catalog.map_name)
        world_map=world.get_map()
    snapshot_manifest=save_map_snapshot(world_map,output/'map_snapshot')
    context=build_town05_route_context(world_map,config['map']['route'])
    route=waypoint_route([waypoint for waypoint,_ in context.route])
    (output/'route.json').write_text(json.dumps(route,indent=2),encoding='utf-8')
    geometry=audit_route(route)
    events=[]
    for event in catalog.events:
        row=dict(event_id=event['id'],valid=True)
        try:
            validate_scene3_event_anchors(context,[event])
            if event['scenario']=='temporary_worker_crossing':
                row['workers']=audit_worker_endpoints(context.adapter,event)
                row['valid']=all(worker['valid'] for worker in row['workers'])
        except (ValueError,RuntimeError,KeyError) as error:
            row.update(valid=False,reason=str(error))
        events.append(row)
    report=dict(source_sha256=hashlib.sha256(raw).hexdigest(),map_name=catalog.map_name,
        route_geometry=geometry,events=events,
        ready=not geometry['suspicious_gaps'] and all(event['valid'] for event in events),
        scope='actor_free_geometry_preflight_only; not runtime acceptance',
        map_geometry_sha256=snapshot_manifest['opendrive_sha256'],
        map_input_mode='offline_snapshot' if map_snapshot is not None else 'live_server',
        weather=config['weather'],weather_applied=False)
    (output/'preflight.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report
