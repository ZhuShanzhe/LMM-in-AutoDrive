"""Prepare a pinned x86 CARLA run; execute only when explicitly requested."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
SCENES = {'scene1': 'Town04', 'scene2': 'Town05', 'scene3': 'Town05'}


def runner_result(output, returncode):
    """Keep task failure distinct from an entry-point or model-load failure."""
    outcome_path = Path(output)/'run/benchmark/run_outcome.json'
    decisions = Path(output)/'run/vla_control_decisions.jsonl'
    outcome = json.loads(outcome_path.read_text()) if outcome_path.is_file() else None
    valid_frames = 0
    if decisions.is_file():
        with decisions.open() as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if 'simulation_frame' in row and 'error' not in row:
                    valid_frames += 1
    started = bool(outcome is not None and valid_frames)
    statuses = {0: 'recorded_checks_passed', 2: 'assessment_failed',
                3: 'assessment_incomplete', 4: 'assessment_error'}
    return dict(runner_exit_code=returncode, closed_loop_started=started,
                valid_control_frames=valid_frames,
                status=statuses.get(returncode, 'runner_failed') if started else 'runner_failed',
                assessment_outcome=outcome)


def build_command(scene, model_root, config, output, host, port, device, seconds):
    if scene not in SCENES or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('A supported scene and positive bounded duration are required')
    common = ['--host', host, '--port', str(port), '--output-dir', str(output),
              '--benchmark-assessment', '--vla-record-sensors',
              '--vla-checkpoint', str(model_root/'lightweight_vla_adapter/universal_three_scene_v6_sensor_policy/model.pt'),
              '--vla-config', str(config), '--vla-device', device, '--vla-precision', 'fp32']
    parser_model = str(model_root/'modernbert-drive-command-compositional')
    if scene == 'scene1':
        runner = 'run_control_experiment.py'
        options = ['basic_voice_urban_5km', '--map', 'Town04_Opt', '--bind-task-geometry', '--scenario-config',
                   str(ROOT/'experiment/CARLA/configs/basic_voice_urban_5km.json'),
                   '--duration-s', str(seconds), '--decision-source', 'vla_scene_bridge',
                   '--command-parser-model', parser_model, '--command-parser-device', device]
    elif scene == 'scene2':
        runner = 'run_complex_avoidance_town05.py'
        options = ['--duration', str(seconds), '--variant-index', '0', '--external-ego-control',
                   '--record-ground-truth', '--ground-truth-every-n', '1', '--record-multimodal',
                   '--command-parser-model', parser_model, '--vla-decision-every-n', '1']
    else:
        runner = 'run_emergency_response_6km.py'
        options = ['--duration', str(seconds), '--event-variant', 'auto', '--seed', '42',
                   '--record-ground-truth', '--ground-truth-every-n', '1',
                   '--ego-controller', 'vla-route-pid', '--vla-parser-model', parser_model,
                   '--vla-decision-every-n', '1']
    return [sys.executable, str(ROOT/'experiment/CARLA'/runner), *options, *common]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', choices=SCENES, required=True)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--duration-s', type=float, default=60)
    parser.add_argument('--check-server', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--fuse-conv-bn', action='store_true')
    args = parser.parse_args()
    if not math.isfinite(args.duration_s) or args.duration_s <= 0:
        parser.error('--duration-s must be finite and positive')
    if args.output.exists():
        parser.error('Output already exists; use a fresh directory')
    output = args.output.resolve()
    output.mkdir(parents=True)
    report = dict(status='checking', closed_loop_validated=False, j6p_test=False,
                  scope='x86 text-to-control diagnostic; not full ASR or competition acceptance',
                  device=args.device, scene=args.scene, blockers=[])
    try:
        import torch
        import carla
        from lightweight_vla_adapter.scripts.prepare_challenge_runtime import verify_assets, resolve_config
        sys.path.insert(0, str(ROOT/'experiment/CARLA'))
        from benchmark.catalog import load_catalog, validate_catalog
        model_root = args.model_root.resolve()
        manifest = json.loads((ROOT/'lightweight_vla_adapter/configs/challenge_assets.json').read_text())
        report['verified_assets'] = len(verify_assets(model_root, manifest))
        report['versions'] = {name: importlib.metadata.version(name) for name in
                              ['torch', 'torchvision', 'transformers', 'carla', 'numpy', 'scipy']}
        if report['versions']['carla'] != '0.9.16':
            raise ValueError('CARLA Python API must be 0.9.16')
        if args.device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('Requested CUDA is unavailable; no silent CPU fallback')
        config = resolve_config(ROOT, model_root,
                                ROOT/f'lightweight_vla_adapter/configs/maps/{SCENES[args.scene]}_traffic_heads.json')
        config['fuse_conv_bn'] = args.fuse_conv_bn
        config_path = output/'runtime.json'
        config_path.write_text(json.dumps(config, indent=2)+'\n')
        catalog = validate_catalog(load_catalog(args.scene.replace('scene', 'scene_')))
        report['catalog'] = catalog
        if not catalog['config_valid']:
            raise ValueError('Invalid scene catalog')
        command = build_command(args.scene, model_root, config_path, output/'run',
                                args.host, args.port, args.device, args.duration_s)
        report['command'] = command
        report['fuse_conv_bn'] = args.fuse_conv_bn
        # --help verifies imports without connecting to or changing the simulation.
        probe = subprocess.run(command[:2]+['--help'], cwd=ROOT, capture_output=True, text=True)
        (output/'runner_help.log').write_text(probe.stdout+probe.stderr)
        if probe.returncode:
            raise RuntimeError('Runner import/help failed; see runner_help.log')
        report['server_checked'] = False
        if args.check_server or args.execute:
            client = carla.Client(args.host, args.port)
            client.set_timeout(5.0)
            version = client.get_server_version()
            report['server_version'] = version
            if not version.startswith('0.9.16'):
                raise ValueError('CARLA server version must match 0.9.16')
            report['server_checked'] = True
        report['status'] = 'prepared'
        if args.execute:
            report['status'] = 'running'
            (output/'preflight.json').write_text(json.dumps(report, indent=2)+'\n')
            env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
            env['PYTHONPATH'] = str(ROOT) + (os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
            with (output/'runner.log').open('w') as log:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            report.update(runner_result(output, result.returncode))
            # Process completion alone is not a successful driving assessment.
            if report['status'] == 'runner_failed':
                raise RuntimeError('Runner failed; inspect runner.log and partial run artifacts')
    except Exception as error:
        report['status'] = 'blocked'
        report['blockers'].append(str(error))
    (output/'preflight.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'catalog'}, indent=2))
    return 1 if report['blockers'] else report.get('runner_exit_code', 0)


if __name__ == '__main__':
    raise SystemExit(main())
