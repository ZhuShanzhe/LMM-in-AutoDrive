import runpy
from pathlib import Path

import pytest

MODULE = runpy.run_path(str(Path(__file__).resolve().parents[1]/'tools/run_challenge_x86.py'))


@pytest.mark.parametrize('scene', ['scene1', 'scene2', 'scene3'])
def test_challenge_command_keeps_model_and_independent_assessment(scene, tmp_path):
    command = MODULE['build_command'](scene, tmp_path/'models', tmp_path/'runtime.json',
                                     tmp_path/'run', '127.0.0.1', 2000, 'cpu', 60)
    assert '--benchmark-assessment' in command
    assert '--vla-record-sensors' in command
    assert command[command.index('--vla-precision')+1] == 'fp32'
    assert command[command.index('--vla-config')+1] == str(tmp_path/'runtime.json')
    assert 'route-pid' not in command
    assert '--benchmark-compound-driver' not in command
    assert '--competition-run' not in command
    if scene == 'scene3':
        assert command[command.index('--ego-controller')+1] == 'vla-route-pid'


@pytest.mark.parametrize('seconds', [0, -1, float('inf'), float('nan')])
def test_invalid_duration_rejected(seconds, tmp_path):
    with pytest.raises(ValueError):
        MODULE['build_command']('scene1', tmp_path, tmp_path, tmp_path, 'localhost', 2000, 'cpu', seconds)
