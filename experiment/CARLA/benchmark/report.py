"""Descriptive aggregation of independent episode assessments, not certification."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def summarize(paths):
    runs = []
    seen = set()
    for raw in paths:
        path = Path(raw).resolve()
        if path in seen:
            raise ValueError('duplicate assessment path: ' + str(path))
        seen.add(path)
        try:
            payload = path.read_bytes()
            data = json.loads(payload)
            if not isinstance(data, dict):
                raise ValueError('assessment must be an object')
            tasks = data.get('tasks')
            if not isinstance(tasks, dict) or not tasks:
                raise ValueError(data.get('reason', 'missing task assessment'))
            if not isinstance(data.get('episode_safety', {}), dict):
                raise ValueError('invalid episode safety result')
            counts = Counter()
            instruction = Counter()
            details = []
            for identity, result in tasks.items():
                if not isinstance(result, dict) or not isinstance(result.get('status'), str):
                    raise ValueError('invalid task result: ' + identity)
                status = result['status']
                counts[status] += 1
                if status != 'NOT_RUN':
                    instruction[result.get('instruction_status', 'UNVERIFIED')] += 1
                if status != 'SUCCESS' or result.get('instruction_status') != 'SUCCESS':
                    details.append(dict(task_id=identity, status=status,
                        instruction_status=result.get('instruction_status', 'UNVERIFIED'),
                        reason=result.get('reason')))
            denominator = len(tasks) - counts['NOT_RUN']
            verified = sum(r.get('status') == 'SUCCESS' and r.get('instruction_status') == 'SUCCESS'
                           for r in tasks.values())
            metadata = data.get('run_metadata') or {}
            runner = metadata.get('runner', {}) if isinstance(metadata, dict) else {}
            source = runner.get('policy_source', 'UNDECLARED') if isinstance(runner, dict) else 'UNDECLARED'
            if source not in {'VLA_MODEL', 'NON_VLA_CONTROL', 'EXTERNAL'}:
                source = 'UNDECLARED'
            runs.append(dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest(),
                evidence_status='READABLE_ASSESSMENT', scene_id=data.get('scene_id'),
                policy_source=source,
                source_sha256=data.get('source_sha256'), map_geometry_sha256=data.get('map_geometry_sha256'),
                route_sha256=data.get('route_sha256'),run_metadata=data.get('run_metadata'),
                assessment_profile_sha256=data.get('assessment_profile_sha256'),
                captured_frames=data.get('captured_frames'), final_route_s_m=data.get('final_route_s_m'),
                traffic_density=data.get('traffic_density'),
                task_status_counts=dict(counts), instruction_status_counts=dict(instruction),
                eligible_tasks=denominator,
                recorded_criteria_pass_rate=counts['SUCCESS']/denominator if denominator else None,
                verified_completion_lower_bound=verified/denominator if denominator else None,
                episode_safety=data.get('episode_safety', {'status':'UNKNOWN'}),
                unresolved_tasks=details))
        except (OSError, ValueError, TypeError) as error:
            runs.append(dict(path=str(path), evidence_status='INVALID_OR_MISSING', reason=str(error)))
    return dict(schema_version='assessment_report/1.0', runs=runs,
        requested_runs=len(runs), unreadable_runs=sum(r['evidence_status']!='READABLE_ASSESSMENT' for r in runs),
        full_benchmark_acceptance=False,
        limitations=[
            'Descriptive per-run results only; differing policies, seeds and task definitions are not pooled.',
            'NOT_RUN excluded; NOT_REACHED, unsupported and invalid tasks remain in the denominator.',
            'Task success and safety are separate; no recorded violation is not proof of full traffic compliance.',
            'Reading summaries does not revalidate their underlying observations or coverage reviews.'])


def markdown(report):
    lines=['# Independent Assessment Report', '',
           '| Run | Scene | Policy | Eligible tasks | Criteria pass | Verified lower bound | Safety |',
           '|---|---|---|---:|---:|---:|---|']
    def cell(value):
        return str(value).replace('|', '\\|').replace('\n', ' ')
    def rate(value):
        return 'N/A' if value is None else f'{value:.1%}'
    for index, run in enumerate(report['runs'], 1):
        if run['evidence_status']!='READABLE_ASSESSMENT':
            lines.append(f'| {index} | INVALID/MISSING | UNDECLARED | N/A | N/A | N/A | UNKNOWN |')
        else:
            lines.append(f"| {index} | {cell(run['scene_id'])} | {cell(run['policy_source'])} | {run['eligible_tasks']} | "
                         f"{rate(run['recorded_criteria_pass_rate'])} | {rate(run['verified_completion_lower_bound'])} | "
                         f"{cell(run['episode_safety'].get('status', 'UNKNOWN'))} |")
    lines.extend(['', '## Evidence'])
    for index, run in enumerate(report['runs'], 1):
        lines.extend(['', f'### Run {index}', '', cell(run['path'])])
        if 'reason' in run:
            lines.append(cell(run['reason']))
        density=run.get('traffic_density') or {}
        lines.append('Traffic geometry: '+cell(density.get('status','UNAVAILABLE')))
        for segment in density.get('route_segments',[]):
            lines.append(f"- {segment['from_m']}-{segment['until_m']} m: "
                f"front vehicles mean={segment['mean'].get('front_cone',0):.2f}; "
                f"below 3={rate(segment['below_three_front_actor_fraction'])}; "
                f"empty route lane={rate(segment.get('empty_route_lane_fraction'))}; "
                f"empty exact-key ego lane={rate(segment['empty_ego_lane_fraction'])}")
        for task in run.get('unresolved_tasks', []):
            lines.append(f"- {cell(task['task_id'])}: {cell(task['status'])}; instruction "
                         f"{cell(task['instruction_status'])}; {cell(task['reason'])}")
    lines.extend(['', '## Interpretation', ''] + ['- '+x for x in report['limitations']])
    return '\n'.join(lines)+'\n'


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summaries', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path, help='new report directory')
    args=parser.parse_args(argv)
    result=summarize(args.summaries)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (args.output/'report.md').write_text(markdown(result),encoding='utf-8')
    return 1 if result['unreadable_runs'] else 0


if __name__=='__main__':
    raise SystemExit(main())
