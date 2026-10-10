"""Replay recorded interface defects and optionally verify their CARLA actuation.

This is a contract diagnostic, not an end-to-end model or route assessment.
Physical checks use the issued commands on the recorded route sections, with
no simulated task completion or replacement of recorded sensor evidence.
"""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiment/CARLA'))

from control.generic_instruction_fsm import GenericInstructionFSM
from control.generic_temporal_risk_supervisor import GenericTemporalRiskSupervisor
from lightweight_vla_adapter.src.longitudinal_contract import enforce_longitudinal_contract


def evidence(scene, segment):
    path = ROOT / f'test/10_02/whole_routes_001/{scene}/github_logs/segment_{segment:03d}/interruption.json'
    document = json.loads(path.read_text(encoding='utf-8'))
    return path, document


def replay():
    results = []
    cases = []
    for segment in (2, 11):
        path, document = evidence('scene3', segment)
        rows = []
        for sample in document['evidence_samples']:
            parsed = sample['decision_trace']['stages']['parsed_instruction']
            if parsed['parse_status'] == 'VALID':
                continue
            before = (sample.get('control_decision')
                      or sample['decision_trace']['stages']['final_decision'])
            after, blocked = GenericInstructionFSM.enforce_parse_status(
                before, SimpleNamespace(parse_status=parsed['parse_status']))
            assert blocked and after['target_speed_kmh'] == 0.
            assert isinstance(after['emergency'], bool)
            assert not after['allow_positive_acceleration']
            rows.append(dict(frame=sample['simulation_frame'], before=before, after=after))
        assert rows, f'No clarification defect in {path}'
        results.append(dict(issue='clarification_output', evidence=str(path.relative_to(ROOT)),
            route_progress_m=document['route_progress_m'], rows=rows, passed=True))
        if segment == 2:
            cases.append(('clarification_stop', document['route_progress_m'], rows[-1]['after']))

    path, document = evidence('scene3', 32)
    supervisor = GenericTemporalRiskSupervisor()
    rows = []
    for sample in document['evidence_samples']:
        stages = sample['decision_trace']['stages']
        risk = sample['risk_assessment']
        parsed = stages['parsed_instruction']
        initial = dict(sample['control_decision'], action='emergency_brake',
                       emergency=True, target_speed_kmh=0.)
        after, override = supervisor.apply(initial, stages['canonical_decision'], risk,
            parsed_intent=parsed['parsed_intent'], requested_lane_direction=None,
            target_lane_risk=None, stationary_elapsed_s=30., resume_active=False,
            resume_speed_kmh=40., frame=sample['simulation_frame'])
        speed = sample['longitudinal_contract']['reference_ego_speed_kmh']
        for _ in range(3):
            after, diag = enforce_longitudinal_contract(after, risk, speed)
        rows.append(dict(frame=sample['simulation_frame'], sensor_frame=sample['sensor_frame'],
            before=sample['control_decision'], after=after, override=override,
            diagnostics=diag))
    approved = [row for row in rows if row['diagnostics']['guarded_crawl_authorized']]
    assert approved and all(row['after']['target_speed_kmh'] == 6. for row in approved)
    results.append(dict(issue='guarded_crawl_contract', evidence=str(path.relative_to(ROOT)),
        route_progress_m=document['route_progress_m'], rows=rows, passed=True))
    cases.append(('guarded_crawl', document['route_progress_m'], approved[-1]['after']))
    return results, cases


def physical_checks(cases, host, port):
    from carla_bootstrap import setup_carla_api
    setup_carla_api()
    import carla
    from control.generic_route_pid import GenericRoutePID
    from run_emergency_response_6km import build_ego_route_plan
    from scene3_town05_route import build_town05_route_context

    client = carla.Client(host, port)
    client.set_timeout(60.)
    world = client.load_world('Town05_Opt')
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = .05
    world.apply_settings(settings)
    config = json.loads((ROOT / 'experiment/CARLA/configs/scene_3_emergency_6km_runtime.json').read_text())
    results = []
    try:
        for name, progress, command in cases:
            context = build_town05_route_context(world.get_map(), config['map']['route'])
            plan = build_ego_route_plan(context, [])
            index = min(range(len(context.distances_m)), key=lambda i: abs(context.distances_m[i] - progress))
            transform = plan[index][0].transform
            transform.location.z += .5
            blueprint = world.get_blueprint_library().find('vehicle.tesla.model3')
            ego = world.spawn_actor(blueprint, transform)
            sensor = None
            collisions = []
            try:
                sensor = world.spawn_actor(world.get_blueprint_library().find('sensor.other.collision'),
                                           carla.Transform(), attach_to=ego)
                sensor.listen(lambda event: collisions.append(event.frame))
                for _ in range(20):
                    ego.apply_control(carla.VehicleControl(brake=1.))
                    world.tick()
                context.tracker.index = index
                route = GenericRoutePID(world, ego, target_speed_kmh=40., fixed_delta_seconds=.05,
                    route_context=context, route_plan=plan, logical_lane_adapter=context.adapter)
                start = ego.get_location()
                rows = []
                for _ in range(120):
                    issued = deepcopy(command)
                    # Only the output/PID interface is under test; no live risk or VLA claim.
                    issued.pop('control_valid_until_s', None)
                    route.set_high_level_decision(issued)
                    control = route.run_step()
                    ego.apply_control(control)
                    world.tick()
                    velocity = ego.get_velocity()
                    speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                    rows.append(dict(frame=world.get_snapshot().frame, speed_kmh=speed,
                                     throttle=control.throttle, brake=control.brake))
                displacement = ego.get_location().distance(start)
                passed = (displacement > 2. and max(r['throttle'] for r in rows) > 0.
                          and max(r['speed_kmh'] for r in rows) <= command['target_speed_kmh'] + .8
                          if name == 'guarded_crawl' else
                          displacement < .5 and all(r['throttle'] == 0. for r in rows))
                passed = passed and not collisions
                results.append(dict(case=name, start_route_progress_m=progress, ticks=len(rows),
                    duration_s=6., displacement_m=displacement, final_speed_kmh=rows[-1]['speed_kmh'],
                    max_speed_kmh=max(r['speed_kmh'] for r in rows), collisions=len(collisions),
                    passed=passed, rows=rows))
            finally:
                if sensor is not None:
                    sensor.stop()
                    sensor.destroy()
                ego.destroy()
                world.tick()
    finally:
        world.apply_settings(original_settings)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--physical', action='store_true')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)
    args = parser.parse_args()
    replay_results, cases = replay()
    report = dict(scope='Recorded-input interface replay; optional isolated CARLA PID actuation',
                  end_to_end_validated=False, replay=replay_results,
                  physical=physical_checks(cases, args.host, args.port) if args.physical else [])
    report['passed'] = all(row['passed'] for row in report['replay'] + report['physical'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'], replay_cases=len(replay_results),
                         physical=[{k:v for k,v in row.items() if k != 'rows'} for row in report['physical']]),
                     ensure_ascii=False, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
