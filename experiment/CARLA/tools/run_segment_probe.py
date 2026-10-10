"""Bound an existing prepared scene-2 runner without altering its policy."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def latest_progress(path):
    if not path.is_file():
        return None
    with path.open('rb') as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size - 1024 * 1024))
        payload = stream.read()
    for line in reversed(payload.splitlines()):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        value = row.get('route_s_m', row.get('progress_m'))
        if isinstance(value, (int, float)):
            return value
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--duration-s', type=float, default=150.)
    parser.add_argument('--end-progress-m', type=float, default=420.)
    parser.add_argument('--max-wall-s', type=float, default=1200.)
    parser.add_argument('--record-sensors', action='store_true')
    args = parser.parse_args()
    report = json.loads((args.prepared/'preflight.json').read_text())
    if min(args.duration_s, args.end_progress_m, args.max_wall_s) <= 0:
        parser.error('Positive time and progress bounds required')
    command = list(report['command'])
    if report['scene'] != 'scene2':
        parser.error('This probe only wraps the existing scene-2 entry')
    command[command.index('--duration')+1] = str(args.duration_s)
    if '--record-multimodal' in command:
        command.remove('--record-multimodal')
    if not args.record_sensors and '--vla-record-sensors' in command:
        command.remove('--vla-record-sensors')
    command.extend(['--no-video', '--benchmark-task', 's2_t05_cmd_01'])
    root = Path(command[1]).resolve().parents[2]
    record = dict(scope='Scene-2 task-1 closed-loop diagnostic; configured DrivingIntent, not live ASR',
        command=command, start_progress_m=0., monitor_end_progress_m=args.end_progress_m,
        duration_limit_s=args.duration_s, wall_time_limit_s=args.max_wall_s,
        route_modified=False, weights_modified=False)
    (args.prepared/'probe_execution.json').write_text(json.dumps(record, indent=2)+'\n')
    environment = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                       CARLA_ROOT='/root/autodl-tmp/CARLA_0.9.16',
                       PYTHONPATH=str(root)+':'+str(root/'experiment/CARLA'))
    progress_path = args.prepared/'run/vla_control_decisions.jsonl'
    with (args.prepared/'runner.log').open('w') as stream:
        process = subprocess.Popen(command, cwd=root, env=environment, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        record['pid'] = process.pid
        started = time.monotonic()
        stop_sent = False
        stop_at = None
        while process.poll() is None:
            progress = latest_progress(progress_path)
            if not stop_sent and progress is not None and progress >= args.end_progress_m:
                process.send_signal(signal.SIGINT)
                record['probe_stop_reason'] = 'diagnostic_end_progress_reached'
                stop_sent = True
                stop_at = time.monotonic()
            elif not stop_sent and time.monotonic()-started >= args.max_wall_s:
                process.send_signal(signal.SIGINT)
                record['probe_stop_reason'] = 'diagnostic_wall_time_limit'
                stop_sent = True
                stop_at = time.monotonic()
            if stop_at is not None and time.monotonic()-stop_at >= 60.:
                os.killpg(process.pid, signal.SIGTERM)
                record['forced_termination'] = True
                try:
                    process.wait(timeout=15.)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            time.sleep(.5)
        record['runner_exit_code'] = process.returncode
        record['elapsed_wall_s'] = time.monotonic()-started
    record['last_logged_progress_m'] = latest_progress(progress_path)
    record.setdefault('probe_stop_reason', 'native_runner_finished')
    (args.prepared/'probe_execution.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))
    return 0 if process.returncode in (0, 2, 3, 130) else process.returncode


if __name__ == '__main__':
    raise SystemExit(main())
