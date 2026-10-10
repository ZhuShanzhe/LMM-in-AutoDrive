"""Preserve native failures and summarize bounded real-traffic diagnostics."""
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
    metadata_path = folder / 'fragment_execution.json'
    if not metadata_path.exists():
        metadata_path = folder / 'probe_execution.json'
    metadata = read(metadata_path)
    if metadata is None:
        return {'completed_diagnostic': False, 'raw_evidence': str(folder)}
    log = folder / 'run/vla_control_decisions.jsonl'
    entries = [json.loads(line) for line in log.open()] if log.exists() else []
    rows = [r for r in entries if r.get('schema_version') == 'unified_vla_decision/1.0']
    policies = [r['risk_assessment']['behavior_policy'] for r in rows]
    controls = [r['control_decision'] for r in rows]
    plans = {}
    for row in rows:
        plan = row.get('control_plan_state')
        if plan:
            plans[plan['request_id']] = {k: plan.get(k) for k in
                ('plan_status', 'active_step_id', 'step_states', 'last_frame_id', 'reason_codes')}
    summary = next((read(folder / 'run' / name) for name in
                    ('summary.json', 'scene_summary.json', 'metrics.json')
                    if (folder / 'run' / name).exists()), {})
    benchmark = read(folder / 'run/benchmark/summary.json') or {}
    result = dict(metadata)
    result.update(
        completed_diagnostic=True, decision_rows=len(rows),
        metadata_source=str(metadata_path), metadata_sha256=digest(metadata_path),
        final_progress_m=metadata.get('final_progress_m', metadata.get('last_logged_progress_m')),
        fallback_rows=dict(Counter(r.get('status', 'unknown') for r in entries
                                   if r.get('schema_version') != 'unified_vla_decision/1.0')),
        risk_levels=dict(Counter(r['risk_assessment']['risk_level'] for r in rows)),
        behavior_dispositions=dict(Counter(p['disposition'] for p in policies)),
        stop_reasons=dict(Counter(c.get('reason') for c in controls
                                 if c['action'] in ('stop', 'emergency_brake'))),
        last_100_stop_reasons=dict(Counter(c.get('reason') for c in controls[-100:])),
        unobserved_road_rows=sum(p['lidar'].get('reason') == 'unobserved_forward_road'
                                for p in policies),
        insufficient_observation_rows_with_measured_clearance=sum(
            p['disposition'] == 'OBSERVATION_INSUFFICIENT'
            and p['lidar'].get('clear_distance_m', 0) >= p['required_clearance_m']
            for p in policies),
        cautious_cap_violations=sum(c['target_speed_kmh'] > 6.000001
            for p, c in zip(policies, controls) if p['disposition'] == 'CAUTIOUS_PROCEED'),
        stop_authority_violations=sum(c['action'] not in ('stop', 'emergency_brake')
            or c['target_speed_kmh'] > 0 for p, c in zip(policies, controls)
            if p['disposition'] in ('MUST_STOP', 'OBSERVATION_INSUFFICIENT')),
        turns_missing_navigation_target=sum(not c.get('target_location') for c in controls
            if c['action'] in ('turn_left', 'turn_right')),
        policy_truth_access_rows=sum(r.get('policy_truth_access', False) for r in rows),
        max_observed_speed_kmh=max((r['execution_feedback']['speed_kmh'] for r in rows), default=None),
        final_control=controls[-1] if controls else None,
        final_plan_state=rows[-1].get('control_plan_state') if rows else None,
        last_observed_plan_states=plans,
        safety={k: summary[k] for k in ('collision_count', 'collisions', 'collision_events',
            'restricted_lane_invasion_count', 'illegal_lane_invasions',
            'illegal_lane_invasion_events', 'speeding_events', 'speeding_frames')
            if k in summary},
        native_benchmark={k: benchmark[k] for k in ('tasks', 'total_tasks', 'success_count',
            'verified_completion_rate_lower_bound', 'benchmark_ready', 'captured_frames')
            if k in benchmark},
        raw_evidence=str(folder),
        decision_log_sha256=digest(log) if log.exists() else None,
        scope='bounded real-background-traffic diagnostics; not full-route acceptance or live ASR',
        full_route_pass_claimed=False,
    )
    if 'safety' in summary:
        result['safety'] = summary['safety']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    probes = {}
    for name in ('scene1_initial', 'scene1_long', 'scene2_start', 'scene2_complete',
                 'scene2_dense', 'scene3_initial', 'scene3_cutin', 'scene3_cutin_fixed', 'scene3_diagnostic_final'):
        folder = args.raw / name
        if not folder.exists():
            continue
        probes[name] = summarize(folder)
        if probes[name]['completed_diagnostic']:
            target = args.output / name
            target.mkdir(exist_ok=True)
            source = Path(probes[name]['metadata_source'])
            shutil.copy2(source, target / source.name)
    regression = (args.raw / 'regression_all.log').read_text().strip().splitlines()[-1]
    result = dict(schema_version='traffic_recovery_diagnostic/1.0', date='2026-10-04',
                  weights_modified=False, full_three_route_acceptance=False,
                  regression_result=regression, probes=probes)
    (args.output / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    shutil.copy2(args.raw / 'regression_all.log', args.output / 'regression.log')
    print(json.dumps({k: {f: v.get(f) for f in ('decision_rows', 'final_progress_m',
        'unobserved_road_rows', 'native_assessment_outcome', 'safety')} for k, v in probes.items()},
        ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
