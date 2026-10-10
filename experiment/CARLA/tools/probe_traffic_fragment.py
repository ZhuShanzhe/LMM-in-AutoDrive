"""Run bounded native traffic diagnostics; preserve native assessment outcomes."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from run_segment_probe import latest_progress


def fragment_command(report, start, task, seconds):
    command = list(report['command'])
    scene = report['scene']
    for flag in ('--vla-record-sensors', '--record-multimodal'):
        if flag in command:
            command.remove(flag)
    duration = '--duration-s' if scene == 'scene1' else '--duration'
    command[command.index(duration) + 1] = str(seconds)
    if scene == 'scene1':
        command.extend(['--resume-route-progress-m', str(start)])
    elif scene == 'scene2':
        command.extend(['--start-progress-m', str(start), '--no-video'])
    elif scene == 'scene3':
        command.extend(['--diagnostic-start-progress-m', str(start)])
    else:
        raise ValueError('Unsupported scene')
    command.extend(['--benchmark-task', task])
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--start-m', type=float, default=0.)
    parser.add_argument('--end-m', type=float, required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--duration-s', type=float, default=80.)
    parser.add_argument('--max-wall-s', type=float, default=800.)
    parser.add_argument('--lidar-points-per-second', type=int)
    parser.add_argument('--lidar-channels', type=int)
    parser.add_argument('--lidar-range-m', type=float)
    args = parser.parse_args()
    if (not all(math.isfinite(v) for v in (args.start_m, args.end_m, args.duration_s, args.max_wall_s))
            or args.start_m < 0 or args.end_m <= args.start_m or min(args.duration_s, args.max_wall_s) <= 0):
        parser.error('Finite positive bounds and end greater than start required')
    if args.lidar_points_per_second is not None and args.lidar_points_per_second <= 0:
        parser.error('LiDAR points per second must be positive')
    if args.lidar_channels is not None and args.lidar_channels <= 0:
        parser.error('LiDAR channels must be positive')
    if args.lidar_range_m is not None and (not math.isfinite(args.lidar_range_m) or args.lidar_range_m <= 0):
        parser.error('LiDAR range must be finite and positive')
    report = json.loads((args.prepared / 'preflight.json').read_text())
    if report['status'] != 'prepared':
        parser.error('A successful preflight is required')
    command = fragment_command(report, args.start_m, args.task, args.duration_s)
    config_path = Path(command[command.index('--vla-config') + 1])
    runtime = json.loads(config_path.read_text())
    runtime['risk_behavior_debug_dir'] = str(args.prepared / 'physical_evidence')
    sampling = dict(runtime.get('physical_lidar_sampling', {}))
    for key, value in (('points_per_second', args.lidar_points_per_second),
                       ('channels', args.lidar_channels), ('range_m', args.lidar_range_m)):
        if value is not None:
            sampling[key] = value
    if sampling:
        runtime['physical_lidar_sampling'] = sampling
    config_path.write_text(json.dumps(runtime, indent=2) + '\n')
    root = Path(command[1]).resolve().parents[2]
    metadata = dict(scope='native scene traffic fragment; not full-route acceptance or live ASR',
        scene=report['scene'], command=command, start_m=args.start_m, end_m=args.end_m,
        task_selector=args.task, state_restored=False, weights_modified=False,
        route_modified=False, physical_evidence_capture=True, native_assessment_success=None)
    from carla_multiview_sensor import resolve_physical_lidar_sampling
    metadata['physical_lidar_sampling'] = resolve_physical_lidar_sampling(runtime)
    metadata['lidar_points_per_second'] = metadata['physical_lidar_sampling']['lidar_points_per_second']
    metadata['sensor_sampling_override'] = any(value is not None for value in
        (args.lidar_points_per_second, args.lidar_channels, args.lidar_range_m))
    progress_path = args.prepared / 'run/vla_control_decisions.jsonl'
    started = time.monotonic()
    environment = dict(os.environ, PYTHONUNBUFFERED='1')
    with (args.prepared / 'runner.log').open('w') as stream:
        process = subprocess.Popen(command, cwd=root, env=environment, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        metadata['pid'] = process.pid
        stop_at = None
        while process.poll() is None:
            progress = latest_progress(progress_path)
            if stop_at is None and progress is not None and progress >= args.end_m:
                process.send_signal(signal.SIGINT)
                metadata['stop_reason'] = 'diagnostic_end_progress'
                stop_at = time.monotonic()
            elif stop_at is None and time.monotonic() - started >= args.max_wall_s:
                process.send_signal(signal.SIGINT)
                metadata['stop_reason'] = 'diagnostic_wall_bound'
                stop_at = time.monotonic()
            if stop_at is not None and time.monotonic() - stop_at >= 60:
                os.killpg(process.pid, signal.SIGTERM)
                metadata['forced_termination'] = True
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            time.sleep(.5)
    metadata.update(runner_exit_code=process.returncode,
        elapsed_wall_s=time.monotonic()-started, final_progress_m=latest_progress(progress_path))
    metadata.setdefault('stop_reason', 'native_runner_finished')
    outcome = args.prepared / 'run/benchmark/run_outcome.json'
    if outcome.exists():
        metadata['native_assessment_outcome'] = json.loads(outcome.read_text())
    (args.prepared / 'fragment_execution.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0 if process.returncode in (0, 2, 3, 130, -signal.SIGINT) else 1


if __name__ == '__main__':
    sys.exit(main())
