"""Run one CARLA job in an owned, map-specific Linux server process."""

import argparse
import errno
import json
import os
from pathlib import Path
import pwd
import re
import signal
import socket
import subprocess
import time

import carla


def server_command(root, town, port, user, engine_ini):
    if not re.fullmatch(r'Town[0-9A-Za-z_]+', town):
        raise ValueError('Expected an installed CARLA Town map name')
    binary = root / 'CarlaUE4.sh'
    asset = root / 'CarlaUE4/Content/Carla/Maps' / (town + '.umap')
    if not binary.is_file() or not asset.is_file():
        raise FileNotFoundError('CARLA launcher or requested map is missing')
    account = pwd.getpwnam(user)
    if account.pw_uid == 0:
        raise ValueError('CARLA engine must run as a non-root user')
    runtime = Path('/tmp') / ('runtime-' + user)
    runtime.mkdir(mode=0o700, exist_ok=True)
    if os.geteuid() == 0:
        os.chown(runtime, account.pw_uid, account.pw_gid)
    environment = ['env', 'HOME=' + account.pw_dir, 'XDG_RUNTIME_DIR=' + str(runtime),
                   'VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json']
    prefix = ['runuser', '-u', user, '--'] if os.geteuid() == 0 else []
    if not prefix and os.geteuid() != account.pw_uid:
        raise PermissionError('Run as root or the requested CARLA user')
    return prefix + environment + [str(binary), '-RenderOffScreen', '-nosound',
        '-quality-level=Low', f'-carla-rpc-port={port}', '-unattended', '-stdout',
        '-FullStdOutLogOutput', '-nowrite', '-ENGINEINI=' + str(engine_ini.resolve())]


def stop_owned(process):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=15.)
    except subprocess.TimeoutExpired:
        remaining = True
    else:
        # runuser can exit before an Unreal child that ignored SIGTERM.
        try:
            os.killpg(process.pid, 0)
            remaining = True
        except ProcessLookupError:
            remaining = False
    if remaining:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10.)


def terminate_request(*_):
    raise KeyboardInterrupt('Termination requested')


def wait_for_port(port,timeout=20.):
    started=time.monotonic()
    while True:
        try:
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                probe.bind(('127.0.0.1',port))
            return time.monotonic()-started
        except OSError as error:
            if error.errno!=errno.EADDRINUSE:raise
            if time.monotonic()-started>=timeout:
                raise TimeoutError(f'Port {port} remains in use; no unrelated process was terminated') from error
            time.sleep(.2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--carla-root', type=Path, required=True)
    parser.add_argument('--town', required=True)
    parser.add_argument('--user', default='carla')
    parser.add_argument('--port', type=int, default=2300)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--startup-timeout', type=float, default=120.)
    parser.add_argument('--command-timeout', type=float, default=3600.)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError('Use a new output directory')
    port_wait_seconds=sum(wait_for_port(port) for port in range(args.port,args.port+3))
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    engine_ini = args.output_dir / 'Engine.ini'
    launch = server_command(args.carla_root.resolve(), args.town, args.port, args.user, engine_ini)
    args.output_dir.mkdir(parents=True)
    account = pwd.getpwnam(args.user)
    user_ini = Path(account.pw_dir) / '.config/Epic/CarlaUE4/Saved/Config/LinuxNoEditor/Engine.ini'
    inherited = user_ini.read_text() if user_ini.is_file() else ''
    engine_ini.write_text(inherited + '\n[/Script/EngineSettings.GameMapsSettings]\n'
        + f'GameDefaultMap=/Game/Carla/Maps/{args.town}\n'
        + f'ServerDefaultMap=/Game/Carla/Maps/{args.town}\n')
    report = dict(schema_version='carla_isolated_job/1.0', status='starting',
                  port_wait_seconds=port_wait_seconds,
                  requested_town=args.town, command=command, server_command=launch,
                  rendering='offscreen RGB enabled; no layers deliberately unloaded')
    server = job = None
    signal.signal(signal.SIGTERM, terminate_request)
    try:
        with (args.output_dir / 'server.log').open('w') as server_log, \
             (args.output_dir / 'job.log').open('w') as job_log:
            started = time.monotonic()
            server = subprocess.Popen(launch, stdout=server_log, stderr=subprocess.STDOUT,
                                      stdin=subprocess.DEVNULL, start_new_session=True)
            last_error = None
            while time.monotonic() - started < args.startup_timeout:
                if server.poll() is not None:
                    raise RuntimeError(f'CARLA exited during startup: {server.returncode}')
                try:
                    # Recreate failed RPC clients; an early connection can stay stale.
                    client = carla.Client('127.0.0.1', args.port)
                    client.set_timeout(5.)
                    world = client.get_world()
                    name = world.get_map().name.rsplit('/', 1)[-1]
                    if name != args.town:
                        raise ValueError(f'Unexpected startup map {name}; refusing hot reload')
                    first = world.get_snapshot().frame
                    second = world.wait_for_tick(2.).frame
                    if second <= first:
                        raise RuntimeError('CARLA frame clock is not advancing')
                    break
                except RuntimeError as error:
                    last_error = repr(error)
                    with (args.output_dir / 'startup_attempts.jsonl').open('a') as attempts:
                        attempts.write(json.dumps(dict(elapsed_s=time.monotonic() - started,
                                                       error=last_error)) + '\n')
                    time.sleep(1.)
            else:
                raise TimeoutError(f'CARLA did not become ready: {last_error}')
            report.update(status='ready', actual_town=name, server_version=client.get_server_version(),
                          startup_seconds=time.monotonic() - started, first_frames=[first, second])
            print(json.dumps(report), flush=True)
            if command:
                job = subprocess.Popen(command, stdout=job_log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
                job_started = time.monotonic()
                while job.poll() is None:
                    if server.poll() is not None:
                        raise RuntimeError('CARLA exited while the job was running')
                    if time.monotonic() - job_started > args.command_timeout:
                        raise TimeoutError('CARLA job exceeded its wall-clock limit')
                    time.sleep(1.)
                report['job_exit_code'] = job.returncode
                if job.returncode:
                    raise RuntimeError(f'CARLA job failed: {job.returncode}; see job.log')
            report['status'] = 'completed'
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        stop_owned(job)
        stop_owned(server)
        report['owned_processes_stopped'] = True
        (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
