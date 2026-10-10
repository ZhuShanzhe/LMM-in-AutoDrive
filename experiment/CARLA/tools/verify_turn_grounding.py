"""Audit recorded route topology and test isolated target/runtime/PID execution.

Physical checks deliberately exclude learned perception and traffic assessment.
They are not substitutes for the separate live VLA scene runner.
"""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'experiment/CARLA'))
from control.road_target_resolver import RoadTargetResolver
from lightweight_vla_adapter.src.driving_plan_runtime import DrivingPlanRuntime
from lightweight_vla_adapter.src.safety_bridge import gate_vla_proposal
from lightweight_vla_adapter.tests.fixtures import integration_documents


def load_routes():
    routes={}
    for scene in ('scene_1','scene_2','scene_3'):
        folder=ROOT/'experiment/CARLA/configs/diagnostic_sections'/scene
        points=json.loads((folder/'geometry.json').read_text())
        routes[scene]=[dict(distance_m=p['s_m'],yaw=p['yaw_deg'],section_id=0,
            **{k:p[k] for k in ('x','y','z','road_id','lane_id','is_junction')}) for p in points]
    return routes


def turn_document(identity,direction):
    doc,_,_,_=integration_documents(parser_action='TURN',direction=direction)
    doc['request_id']=identity
    text=f'Turn {direction.lower()} at the next junction.'
    doc['input'].update(raw_text=text,normalized_text=text)
    doc['parse_result'].update(source='isolated_geometry_contract_probe',model_prediction=False)
    doc['intent']['steps'][0].update(on_blocked='SAFE_STOP',completion=dict(type='JUNCTION_EXITED'))
    return doc


def geometry_audit(routes):
    results=[]
    for scene,route in routes.items():
        resolver=RoadTargetResolver(route)
        checks=[]
        for index,visit in enumerate(resolver.visits):
            for direction in ('LEFT','RIGHT'):
                doc=turn_document(f'{scene}-{index}-{direction}',direction)
                ready=resolver.resolve(doc,frame_id='geometry-probe',
                    progress_m=max(0.,visit['start_m']-1.))['step_1']
                expected=visit['complete'] and visit['distinct_road_turn_candidate'] and visit['direction']==direction
                checks.append(dict(junction=visit,requested_direction=direction,
                    readiness=ready,passed=ready['reachable']==expected))
        results.append(dict(scene=scene,checks=checks,passed=all(c['passed'] for c in checks),
            configured_turn_tasks=[t for t in json.loads((ROOT/'experiment/CARLA/configs/diagnostic_sections'/scene/'sections.json').read_text())['tasks']
                if '\u53f3\u8f6c' in t['instruction'] or '\u5de6\u8f6c' in t['instruction']]))
    return results


def physical_checks(routes):
    from carla_bootstrap import setup_carla_api
    setup_carla_api()
    import carla
    from continuous.route_manager import RouteManager
    from control.generic_route_pid import GenericRoutePID
    client=carla.Client('127.0.0.1',2000);client.set_timeout(60.)
    world=client.load_world('Town05_Opt')
    original=world.get_settings();settings=world.get_settings()
    settings.synchronous_mode=True;settings.fixed_delta_seconds=.05
    settings.no_rendering_mode=True
    world.apply_settings(settings)
    cases=[]
    for scene,direction in [('scene_1','RIGHT'),('scene_2','RIGHT'),('scene_2','LEFT'),('scene_3','STRAIGHT_OR_SHALLOW')]:
        visits=RoadTargetResolver(routes[scene]).visits
        visit=next(v for v in visits if v['direction']==direction and v['complete']
            and (scene!='scene_1' or v['start_m']>3000.))
        cases.append((scene,direction,visit))
    results=[]
    try:
        for scene,direction,visit in cases:
            route=routes[scene];start_m=max(0.,visit['start_m']-25.)
            manager=RouteManager(world);manager.route=deepcopy(route)
            manager.route_length_m=route[-1]['distance_m'];manager.seek(start_m)
            point=route[manager.current_index]
            pose=carla.Transform(carla.Location(x=point['x'],y=point['y'],z=point['z']+.5),carla.Rotation(yaw=point['yaw']))
            ego=world.spawn_actor(world.get_blueprint_library().find('vehicle.tesla.model3'),pose)
            sensor=None;collisions=[];rows=[]
            try:
                sensor=world.spawn_actor(world.get_blueprint_library().find('sensor.other.collision'),carla.Transform(),attach_to=ego)
                sensor.listen(lambda e:collisions.append(e.frame))
                for _ in range(20):
                    ego.apply_control(carla.VehicleControl(brake=1.));world.tick()
                pid=GenericRoutePID(world,ego,target_speed_kmh=20.,fixed_delta_seconds=.05,route_manager=manager)
                runtime=DrivingPlanRuntime()
                doc=turn_document(f'{scene}-{direction}-physical',direction)
                if direction=='STRAIGHT_OR_SHALLOW':
                    doc['intent']['steps'][0].update(action='PROCEED',parameters=dict(condition='STRAIGHT_THROUGH_JUNCTION'))
                # Establish cruising speed through actual observed execution,
                # not fabricated feedback or a speed override after gating.
                doc['intent']['steps'][0]['depends_on']=['speed_0']
                doc['intent']['steps'].insert(0,dict(step_id='speed_0',action='SET_SPEED',
                    parameters=dict(target_speed_mps=15./3.6),trigger=dict(type='IMMEDIATE'),
                    depends_on=[],preconditions=[],on_blocked='SAFE_STOP',
                    completion=dict(type='TARGET_SPEED_REACHED')))
                _,observation,_,risk=integration_documents()
                observation['source']='isolated_actuation_probe'
                observation['objects']=[]
                for _ in range(1200):
                    snapshot=world.get_snapshot();frame=f'carla_{snapshot.frame}'
                    velocity=ego.get_velocity();speed=math.sqrt(velocity.x**2+velocity.y**2+velocity.z**2)
                    manager.update(ego)
                    ready=pid.road_step_readiness(doc,frame_id=frame)
                    state=pid.execution_state();state.update(observation_frame_id=frame,ego_heading_deg=ego.get_transform().rotation.yaw)
                    step=runtime.prepare(doc,frame_id=frame,timestamp_s=snapshot.timestamp.elapsed_seconds,
                        speed_mps=speed,execution_state=state,readiness=ready)
                    if step is not None:pid.pin_road_step_target(step,ready.get(step['step_id']),frame_id=frame)
                    observation.update(frame_id=frame,timestamp=snapshot.timestamp.elapsed_seconds)
                    observation['ego']['speed_mps']=speed;risk['frame_id']=frame
                    decision=runtime.advance(observation,risk,readiness=ready)
                    decision=gate_vla_proposal(dict(schema_version='1.0.0',
                        request_id=doc['request_id'],frame_id=frame,action='accelerate',
                        target_speed_kmh=15.,target_lane=None,target_location=None,
                        target_entity_id=None,confidence=1.,model='contract_probe_fixed_speed',latency_ms=0.),
                        decision,risk)
                    pid.set_high_level_decision(decision);control=pid.run_step();ego.apply_control(control);world.tick()
                    rows.append(dict(frame=snapshot.frame,progress_m=manager.progress_m,speed_kmh=3.6*speed,
                        heading_deg=ego.get_transform().rotation.yaw,plan_status=runtime.state['plan_status'],
                        decision=decision,readiness=ready,pid=pid.execution_state()['pid']))
                    if runtime.state['plan_status']=='COMPLETED' or collisions:break
                results.append(dict(scene=scene,direction=direction,junction=visit,start_m=start_m,
                    end_m=manager.progress_m,ticks=len(rows),collisions=len(collisions),
                    plan_status=runtime.state['plan_status'],passed=runtime.state['plan_status']=='COMPLETED' and not collisions,
                    rows=rows,scope='isolated route/intent/runtime/PID; no learned risk or signal test'))
            finally:
                if sensor is not None:sensor.stop();sensor.destroy()
                ego.destroy();world.tick()
    finally:world.apply_settings(original)
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--physical',action='store_true')
    args=parser.parse_args();routes=load_routes()
    report=dict(scope='recorded actual map geometry; optional isolated actuation',
        geometry=geometry_audit(routes),physical=physical_checks(routes) if args.physical else [],
        end_to_end_validated=False)
    report['passed']=all(r['passed'] for r in report['geometry']+report['physical'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'],geometry_checks=sum(len(r['checks']) for r in report['geometry']),
        physical=[{k:v for k,v in r.items() if k!='rows'} for r in report['physical']]),indent=2))
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
