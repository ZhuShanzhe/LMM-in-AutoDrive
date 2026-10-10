"""Resolve a diagnostic section by id or route progress without CARLA/GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'experiment/CARLA'))
from benchmark.route_sections import validate_sections
from benchmark.catalog import CONFIG_ROOT, SCENES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', choices=['scene1','scene2','scene3'], required=True)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument('--section')
    choice.add_argument('--phase')
    choice.add_argument('--at-m', type=float)
    parser.add_argument('--root', type=Path, default=ROOT/'experiment/CARLA/configs/diagnostic_sections')
    parser.add_argument('--list-sections', action='store_true')
    args = parser.parse_args()
    name = args.scene.replace('scene','scene_')
    folder = args.root/name
    report = json.loads((folder/'sections.json').read_text(encoding='utf-8'))
    validate_sections(report)
    source = ((folder/report['runtime_source_file']).resolve() if report.get('runtime_source_file')
              else ROOT/'experiment/CARLA/configs'/report['source_file'])
    payload = source.read_bytes().replace(b'\r\n', b'\n')
    if hashlib.sha256(payload).hexdigest() != report['source_sha256']:
        parser.error('Source configuration changed; rebuild sections before using them')
    binding = json.loads(payload).get('geometry_binding') or {}
    if binding:
        original = (CONFIG_ROOT/SCENES[name][0]).read_bytes().replace(b'\r\n', b'\n')
        if hashlib.sha256(original).hexdigest() != binding.get('source_sha256'):
            parser.error('Original scene configuration changed; rebuild the map binding and sections')
    rows = report['sections']
    if args.section:
        rows = [s for s in rows if s['section_id']==args.section]
    elif args.phase:
        rows = [p for p in report['phases'] if p['phase_id']==args.phase]
    elif args.at_m is not None:
        rows = [s for s in rows if s['start_m'] <= args.at_m < s['end_m']
                or args.at_m==report['route_length_m']==s['end_m']]
    if not rows:
        parser.error('No matching section')
    if args.section or args.phase or args.at_m is not None:
        section = dict(rows[0])
        section['instructions'] = [t for t in report['tasks'] if t['task_id'] in section['task_ids']]
        section['native_runner_options'] = (
            {'--resume-route-progress-m':section['replay_start_m'],
             '--stop-at-route-progress-m':section['stop_after_m'], '--scenario-config':str(source)}
            if name=='scene_1' else {'--start-progress-m':section['replay_start_m']}
            if name=='scene_2' else {})
        section['runner_limitations'] = (
            'Native pose seek; task/event/temporal state is not restored automatically.' if name=='scene_1' else
            'Native cold seek omits earlier commands; dependency tasks need full replay/state restoration. '
            'End progress requires monitoring; native runner has no distance-stop flag.' if name=='scene_2' else
            'Native runner has no progress-start flag; use recorded-input replay or isolated pose-based '
            'interface checks. Full VLA/event subroute execution requires a separate stateful entry.')
        print(json.dumps(section, ensure_ascii=False, indent=2))
    else:
        for s in rows if args.list_sections else report['phases']:
            identity = s.get('section_id', s.get('phase_id'))
            print(f'{identity}: {s["start_m"]:.1f}-{s["end_m"]:.1f} m; '
                  f'replay {s["replay_start_m"]:.1f}-{s["stop_after_m"]:.1f}; '
                  f'tasks={",".join(s["task_ids"])}')


if __name__ == '__main__':
    main()
