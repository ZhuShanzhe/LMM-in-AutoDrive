from pathlib import Path
import signal
import socket
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from lightweight_vla_adapter.scripts import run_carla_isolated as runner


@pytest.fixture
def installation(tmp_path, monkeypatch):
    (tmp_path / 'CarlaUE4.sh').touch()
    maps = tmp_path / 'CarlaUE4/Content/Carla/Maps'
    maps.mkdir(parents=True)
    (maps / 'Town03_Opt.umap').touch()
    monkeypatch.setattr(runner.pwd, 'getpwnam', lambda _: SimpleNamespace(
        pw_uid=1001, pw_gid=1001, pw_dir='/home/carla'))
    monkeypatch.setattr(runner.os, 'geteuid', lambda: 0)
    monkeypatch.setattr(runner.os, 'chown', lambda *_: None)
    monkeypatch.setattr(Path, 'mkdir', lambda *_, **__: None)
    return tmp_path


def test_map_is_selected_before_start_without_disabling_rgb(installation):
    command = runner.server_command(installation, 'Town03_Opt', 2300, 'carla', installation / 'Engine.ini')
    assert command[:4] == ['runuser', '-u', 'carla', '--']
    assert '-RenderOffScreen' in command
    assert '-nowrite' in command
    assert '-no-rendering' not in command
    assert '-ENGINEINI=' + str((installation / 'Engine.ini').resolve()) in command


def test_invalid_map_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        runner.server_command(tmp_path, '../Town03;true', 2300, 'carla', tmp_path / 'Engine.ini')


def test_missing_assets_are_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        runner.server_command(tmp_path, 'Town03_Opt', 2300, 'carla', tmp_path / 'Engine.ini')


def test_engine_root_user_is_rejected(installation, monkeypatch):
    monkeypatch.setattr(runner.pwd, 'getpwnam', lambda _: SimpleNamespace(pw_uid=0))
    with pytest.raises(ValueError, match='non-root'):
        runner.server_command(installation, 'Town03_Opt', 2300, 'root', installation / 'Engine.ini')


def test_cleanup_targets_only_owned_group(monkeypatch):
    kill = Mock(side_effect=[None,ProcessLookupError()])
    monkeypatch.setattr(runner.os, 'killpg', kill)
    process = Mock(pid=12345)
    runner.stop_owned(process)
    assert kill.call_args_list == [call(12345, signal.SIGTERM),call(12345,0)]
    process.wait.assert_called_once_with(timeout=15.)


def test_cleanup_escalates_only_after_timeout(monkeypatch):
    kill = Mock()
    monkeypatch.setattr(runner.os, 'killpg', kill)
    process = Mock(pid=12345)
    process.wait.side_effect = [subprocess.TimeoutExpired('owned', 15), 0]
    runner.stop_owned(process)
    assert kill.call_args_list == [call(12345, signal.SIGTERM), call(12345, signal.SIGKILL)]


def test_cleanup_accepts_already_exited_group(monkeypatch):
    monkeypatch.setattr(runner.os, 'killpg', Mock(side_effect=ProcessLookupError))
    runner.stop_owned(Mock(pid=12345))
    runner.stop_owned(None)


def test_cleanup_kills_child_group_even_when_wrapper_has_exited(monkeypatch):
    kill=Mock()
    monkeypatch.setattr(runner.os,'killpg',kill)
    process=Mock(pid=12345)
    runner.stop_owned(process)
    assert kill.call_args_list==[call(12345,signal.SIGTERM),call(12345,0),call(12345,signal.SIGKILL)]


def test_busy_port_never_launches_or_kills_other_server(tmp_path, monkeypatch):
    with socket.socket() as existing:
        existing.bind(('127.0.0.1', 0))
        existing.listen(1)
        port = existing.getsockname()[1]
        monkeypatch.setattr('sys.argv', ['runner', '--carla-root', str(tmp_path),
            '--town', 'Town03_Opt', '--port', str(port), '--output-dir', str(tmp_path / 'out')])
        launch, kill = Mock(), Mock()
        monkeypatch.setattr(runner.subprocess, 'Popen', launch)
        monkeypatch.setattr(runner.os, 'killpg', kill)
        with pytest.raises(OSError):
            runner.main()
        launch.assert_not_called()
        kill.assert_not_called()
