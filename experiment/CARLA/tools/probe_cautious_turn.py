"""Isolated live-camera/LiDAR/radar VLA diagnostic, never an official score."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiment/CARLA'))
sys.path.insert(0,str(ROOT/'experiment/CARLA/tools'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-steps',type=int,default=1800)
    parser.add_argument('--max-wall-s',type=float,default=650.)
    parser.add_argument('--obstacle',action='store_true')
    parser.add_argument('--persistent-medium-fixture',action='store_true',
        help='Counterfactual risk input for control-policy testing, not model accuracy')
    args=parser.parse_args()
    if min(args.max_steps,args.max_wall_s)<=0:parser.error('Positive bounds required')
    args.output.mkdir(parents=True,exist_ok=True)
    from carla_bootstrap import setup_carla_api
    setup_carla_api()
    import carla
    from scenarios.complex.town05_scene2 import build_configured_route
    from benchmark.turn_fixture import waypoint_route
    from continuous.route_manager import RouteManager
    from control.generic_route_pid import GenericRoutePID
    from universal_vla_controller import UniversalVLAController
    from verify_turn_grounding import turn_document
    client=carla.Client('127.0.0.1',2000);client.set_timeout(60.)
    world=client.load_world('Town05_Opt');original=world.get_settings();settings=world.get_settings()
    settings.synchronous_mode=True;settings.fixed_delta_seconds=.05;settings.no_rendering_mode=False
    world.apply_settings(settings)
    source=json.loads((ROOT/'experiment/CARLA/configs/scene_2_town05_runtime.json').read_text())
    from run_complex_avoidance_town05 import apply_weather
    apply_weather(world,source['weather'])
    route,distances,_,_=build_configured_route(world.get_map(),source['route'])
    manager=RouteManager(world);manager.route=waypoint_route([w for w,_ in route],distances)
    manager.route_length_m=float(distances[-1]);manager.seek(250.)
    point=manager.route[manager.current_index]
    pose=carla.Transform(carla.Location(x=point['x'],y=point['y'],z=point['z']+.5),carla.Rotation(yaw=point['yaw']))
    ego=controller=collision_sensor=obstacle=None;collisions=[];started=time.monotonic();steps=0
    record=dict(scope='isolated real-model sensor/control diagnostic; no ambient traffic or official benchmark',
        start_m=250.,map='Town05_Opt',weights_modified=False,teacher_force=False,
        configuration='same model/runtime assets; configured right-turn plus keep-lane intent',obstacle=args.obstacle)
    record['weather']=source['weather']
    record['risk_input_fixture']=bool(args.persistent_medium_fixture)
    try:
        ego=world.spawn_actor(world.get_blueprint_library().find('vehicle.audi.a2'),pose)
        collision_sensor=world.spawn_actor(world.get_blueprint_library().find('sensor.other.collision'),carla.Transform(),attach_to=ego)
        collision_sensor.listen(lambda event:collisions.append(event.frame))
        if args.obstacle:
            index=min(range(len(distances)),key=lambda i:abs(distances[i]-263.))
            point=route[index][0].transform
            obstacle_pose=carla.Transform(carla.Location(x=point.location.x,y=point.location.y,z=point.location.z+.5),point.rotation)
            obstacle=world.spawn_actor(world.get_blueprint_library().find('vehicle.tesla.model3'),obstacle_pose)
            obstacle.apply_control(carla.VehicleControl(brake=1.))
        for _ in range(20):ego.apply_control(carla.VehicleControl(brake=1.));world.tick()
        pid=GenericRoutePID(world,ego,target_speed_kmh=20.,fixed_delta_seconds=.05,route_manager=manager)
        doc=turn_document('sensor-risk-behavior-probe','RIGHT')
        doc['intent']['steps'].append(dict(step_id='step_2',action='KEEP_LANE',parameters={},
            trigger=dict(type='AFTER_STEP',step_id='step_1'),depends_on=['step_1'],preconditions=[],
            on_blocked='SAFE_STOP',completion=dict(type='ACTION_REACHED')))
        runtime=json.loads((args.prepared/'runtime.json').read_text())
        runtime['risk_behavior_debug_dir']=str(args.output/'sensor_evidence')
        runtime_path=args.output/'runtime.json';runtime_path.write_text(json.dumps(runtime,indent=2)+'\n')
        model_root=Path('/root/autodl-tmp/models/challenge-assets-pinned')
        controller=UniversalVLAController(world=world,ego=ego,route_controller=pid,
            commands=[dict(id='sensor-risk-behavior-probe',announce_at_m=250.,text=doc['input']['raw_text'],driving_intent=doc)],
            checkpoint_path=model_root/'lightweight_vla_adapter/universal_three_scene_v6_sensor_policy/model.pt',
            config_path=runtime_path,parser_model_path=model_root/'modernbert-drive-command-compositional',
            output_path=args.output/'vla_control_decisions.jsonl',precision='fp32',device='cuda',
            decision_interval_frames=1,fixed_delta_seconds=.05,enable_lidar=True)
        if args.persistent_medium_fixture:
            predict=controller.pipeline.predict_proposal
            def fixture(*values,**kwargs):
                proposal=predict(*values,**kwargs)
                raw=controller.pipeline._last_visual_risk_assessment
                if raw['risk_level']=='low':
                    revised=deepcopy(raw)
                    revised.update(risk_level='medium',recommended_action='decelerate',
                        probabilities=dict(low=.001,medium=.998,high=.001),raw_argmax_level='medium',
                        source='synthetic_medium_fixture_over_real_sensor_pipeline',
                        reason_codes=['synthetic_persistent_medium_fixture'],fixture_original_risk=raw)
                    controller.pipeline._last_visual_risk_assessment=revised
                return proposal
            controller.pipeline.predict_proposal=fixture
        for steps in range(1,args.max_steps+1):
            manager.update(ego);control=controller.run_step();ego.apply_control(control);world.tick()
            if collisions:record['stop_reason']='collision';break
            if controller.driving_plan and (controller.driving_plan.state or {}).get('plan_status')=='COMPLETED':
                record['stop_reason']='plan_completed';break
            if time.monotonic()-started>args.max_wall_s:record['stop_reason']='wall_bound';break
        record.setdefault('stop_reason','step_bound')
        record.update(steps=steps,progress_m=manager.progress_m,collisions=len(collisions),
            plan_state=deepcopy(controller.driving_plan.state) if controller.driving_plan else None,
            elapsed_wall_s=time.monotonic()-started)
    except Exception as error:
        record['initialization_or_runtime_error']=repr(error)
        raise
    finally:
        if controller is not None:record['controller_summary']=controller.close()
        for actor in (collision_sensor,obstacle,ego):
            if actor is not None:
                if hasattr(actor,'stop'):actor.stop()
                actor.destroy()
        world.tick();world.apply_settings(original)
        (args.output/'probe_execution.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ('plan_state','controller_summary')},indent=2))


if __name__=='__main__':main()
