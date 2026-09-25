"""Static wiring checks avoid importing GPU models or connecting to CARLA."""
import ast
from pathlib import Path
import pytest


@pytest.mark.parametrize('name', ['run_control_experiment.py', 'run_complex_avoidance_town05.py',
                                'run_emergency_response_6km.py'])
def test_recording_option_passed_only_to_universal_controller(name):
    root=Path(__file__).resolve().parents[1]
    tree=ast.parse((root/name).read_text(encoding='utf-8'))
    calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call)
           and any(k.arg=='sensor_recording_dir' for k in node.keywords)]
    assert len(calls)==1
    assert isinstance(calls[0].func,ast.Name) and calls[0].func.id=='UniversalVLAController'
    value=next(k.value for k in calls[0].keywords if k.arg=='sensor_recording_dir')
    assert isinstance(value,ast.IfExp)
    assert isinstance(value.test,ast.Attribute) and value.test.attr=='vla_record_sensors'


@pytest.mark.parametrize('name', ['run_control_experiment.py', 'run_emergency_response_6km.py'])
def test_non_model_multimodal_capture_uses_independent_assessment(name):
    root=Path(__file__).resolve().parents[1]
    source=(root/name).read_text(encoding='utf-8')
    assert "--record-multimodal requires --benchmark-assessment" in source
    assert 'multimodal_capture.observe(' in source
    assert 'multimodal_capture.close()' in source
