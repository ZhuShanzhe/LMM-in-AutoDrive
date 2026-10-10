import importlib.util
from pathlib import Path
import sys

import pytest

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location('probe_traffic_fragment', TOOLS / 'probe_traffic_fragment.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize('scene,flag', [('scene1', '--resume-route-progress-m'),
    ('scene2', '--start-progress-m'), ('scene3', '--diagnostic-start-progress-m')])
def test_fragment_command_retains_model_and_native_assessment(scene, flag):
    duration = '--duration-s' if scene == 'scene1' else '--duration'
    command = ['python', 'runner.py', duration, '1', '--benchmark-assessment',
        '--vla-checkpoint', 'model.pt', '--vla-record-sensors', '--record-multimodal']
    result = module.fragment_command(dict(scene=scene, command=command), 100, 'task1', 80)
    assert result[result.index(flag)+1] == '100'
    assert result[result.index(duration)+1] == '80'
    assert '--benchmark-assessment' in result and 'model.pt' in result
    assert '--vla-record-sensors' not in result
    assert command[command.index(duration)+1] == '1'
