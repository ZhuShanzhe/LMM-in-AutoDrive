"""Summarize bounded risk-policy diagnostics without treating fixtures as model scores."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def summarize(folder):
    metadata_path = folder / 'probe_execution.json'
    metadata = json.loads(metadata_path.read_text())
    log = folder / 'vla_control_decisions.jsonl'
    entries = [json.loads(line) for line in log.open()]
    rows = [row for row in entries if row.get('schema_version') == 'unified_vla_decision/1.0']
    cautious = [row for row in rows if row['risk_assessment']['behavior_policy']['disposition'] == 'CAUTIOUS_PROCEED']
    stopping = [row for row in rows if row['risk_assessment']['behavior_policy']['disposition'] in ('MUST_STOP', 'OBSERVATION_INSUFFICIENT')]
    moving_turns = [row for row in rows if row['control_decision']['action'] in ('turn_left', 'turn_right')]
    end = rows[-100:]
    return {
        'scope': metadata['scope'],
        'synthetic_risk_fixture': metadata.get('risk_input_fixture', False),
        'model_accuracy_measurement': False,
        'ambient_traffic': False,
        'stop_reason': metadata.get('stop_reason'),
        'steps': metadata.get('steps'),
        'decision_rows': len(rows),
        'fallback_rows': dict(Counter(row.get('status', 'unknown') for row in entries if row.get('schema_version') != 'unified_vla_decision/1.0')),
        'start_progress_m': metadata['start_m'],
        'final_progress_m': metadata['progress_m'],
        'collisions': metadata['collisions'],
        'plan_status': (metadata.get('plan_state') or {}).get('plan_status'),
        'elapsed_wall_s': metadata['elapsed_wall_s'],
        'risk_levels': dict(Counter(row['risk_assessment']['risk_level'] for row in rows)),
        'behavior_dispositions': dict(Counter(row['risk_assessment']['behavior_policy']['disposition'] for row in rows)),
        'control_actions': dict(Counter(row['control_decision']['action'] for row in rows)),
        'stop_reasons': dict(Counter(row['control_decision'].get('reason') for row in rows if row['control_decision']['action'] in ('stop', 'emergency_brake'))),
        'cautious_target_speed_max_kmh': max((row['control_decision']['target_speed_kmh'] for row in cautious), default=None),
        'cautious_target_cap_violations': sum(row['control_decision']['target_speed_kmh'] > 6.000001 for row in cautious),
        'stop_authority_violations': sum(row['control_decision']['action'] not in ('stop', 'emergency_brake') or row['control_decision']['target_speed_kmh'] > 0 for row in stopping),
        'turns_missing_navigation_target': sum(not row['control_decision'].get('target_location') for row in moving_turns),
        'observed_speed_max_kmh': max((row['execution_feedback']['speed_kmh'] for row in rows), default=None),
        'last_100_observed_speed_max_kmh': max((row['execution_feedback']['speed_kmh'] for row in end), default=None),
        'last_control_action': rows[-1]['control_decision']['action'] if rows else None,
        'classification_changed_rows': sum(row['risk_assessment']['behavior_policy']['classification_changed'] for row in rows),
        'policy_truth_access_rows': sum(row['policy_truth_access'] for row in rows),
        'raw_evidence': str(folder),
        'raw_log_sha256': sha256(log),
        'metadata_sha256': sha256(metadata_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    probes = {}
    for name in ('isolated_turn', 'isolated_turn_evening', 'real_evening_final', 'persistent_medium_final', 'obstacle_final'):
        probes[name] = summarize(args.raw / name)
        target = args.output / name
        target.mkdir(exist_ok=True)
        shutil.copy2(args.raw / name / 'probe_execution.json', target / 'probe_execution.json')
    regression = (args.raw / 'regression_current.log').read_text()
    result = {
        'schema_version': 'risk_behavior_diagnostic/1.0',
        'date': '2026-10-04',
        'full_three_route_acceptance': False,
        'weights_modified': False,
        'regression_result': regression.strip().splitlines()[-1],
        'probes': probes,
    }
    (args.output / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    shutil.copy2(args.raw / 'regression_current.log', args.output / 'regression.log')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
