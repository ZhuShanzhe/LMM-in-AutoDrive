"""Summarize current diagnostics without replacing original assessments."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def summarize(folder):
    log = folder / 'run/vla_control_decisions.jsonl'
    result = dict(raw_directory=str(folder), metadata=read(folder/'fragment_execution.json'))
    counters = {key: Counter() for key in ('actions','risk_levels','dispositions','signals','stop_reasons')}
    count = 0
    bounded_violations = 0
    weather_violations = 0
    high_motion_rows = 0
    medium_motion_rows = 0
    missing_turn_target = 0
    max_actual_speed = 0.
    plans = {}
    for line in log.open():
        row = json.loads(line)
        if row.get('schema_version') != 'unified_vla_decision/1.0':
            continue
        count += 1
        if count == 1:
            result['start_progress_m'] = row['route_s_m']
            result['first_timestamp_s'] = row['risk_assessment']['risk_evidence']['decision_timestamp_s']
        control = row['control_decision']
        risk = row['risk_assessment']
        policy = risk['behavior_policy']
        target = control['target_speed_kmh']
        counters['actions'][control['action']] += 1
        counters['risk_levels'][risk['risk_level']] += 1
        counters['dispositions'][policy['disposition']] += 1
        counters['signals'][row['traffic_control_contract']['status']] += 1
        if target == 0:
            counters['stop_reasons'][control.get('reason')] += 1
        if policy['disposition'] == 'CAUTIOUS_PROCEED':
            bounded_violations += target > policy['max_speed_kmh'] + 1e-5
            high_motion_rows += risk['risk_level'] == 'high' and target > 0
            medium_motion_rows += risk['risk_level'] == 'medium' and target >= 20
        bound = row.get('environment_speed_bound')
        if bound:
            weather_violations += target > bound['max_speed_kmh'] + 1e-5
        missing_turn_target += control['action'] in ('turn_left','turn_right') and not control.get('target_location')
        max_actual_speed = max(max_actual_speed, row['execution_feedback']['speed_kmh'])
        plan = row.get('control_plan_state')
        if plan:
            plans[plan['request_id']] = plan
        result['final_progress_m'] = row['route_s_m']
        result['final_control'] = control
        result['final_environment_speed_bound'] = bound
        result['last_timestamp_s'] = risk['risk_evidence']['decision_timestamp_s']
    result.update(decision_rows=count, progress_delta_m=result['final_progress_m']-result['start_progress_m'],
        bounded_target_violations=bounded_violations, environment_target_violations=weather_violations,
        confirmed_visual_high_positive_target_rows=high_motion_rows,
        medium_target_at_least_20_rows=medium_motion_rows,
        missing_turn_target_rows=missing_turn_target, max_actual_speed_kmh=max_actual_speed,
        last_observed_plans=plans, decision_log_sha256=digest(log), **{k:dict(v) for k,v in counters.items()})
    summary = next((read(folder/'run'/name) for name in ('summary.json','scene_summary.json','metrics.json')
                    if (folder/'run'/name).exists()), {})
    result['safety'] = summary.get('safety', {k:summary[k] for k in ('collision_count','collisions',
        'collision_events','illegal_lane_invasions','illegal_lane_invasion_events',
        'restricted_lane_invasion_count','speeding_events','speeding_frames') if k in summary})
    result['native_benchmark'] = read(folder/'run/benchmark/summary.json')
    result['observed_duration_s'] = result['last_timestamp_s'] - result['first_timestamp_s']
    if result['native_benchmark']:
        result['benchmark_safety'] = result['native_benchmark'].get('episode_safety')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    probes = {name:summarize(args.raw/name) for name in
        ('scene1_default','scene2_default','scene3_default','scene2_aligned','scene2_resumed','scene3_aligned')
        if (args.raw/name/'fragment_execution.json').exists()}
    result = dict(schema_version='adaptive_speed_diagnostic/1.0', weights_modified=False,
        scene_instructions_modified=True, full_route_acceptance_claimed=False, probes=probes)
    (args.output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for name in ('map_commands_before.json','map_commands_after.json','full_regression.log','junction_bindings_after.json'):
        if (args.raw/name).exists():
            shutil.copy2(args.raw/name,args.output/name)
    for name, report in probes.items():
        print(name, report['decision_rows'], round(report['progress_delta_m'],3),
            report['metadata']['native_assessment_outcome'], report['safety'])


if __name__ == '__main__':
    main()
