import subprocess
import sys
from pathlib import Path


def test_model_import_does_not_load_control_integration():
    code = """
import sys
import lightweight_vla_adapter
from lightweight_vla_adapter.scripts.run_offline_inference import build_model
assert not any(x == 'scene_understanding' or x.startswith('scene_understanding.') for x in sys.modules)
try:
    lightweight_vla_adapter.nonexistent_attribute
except AttributeError:
    pass
else:
    raise AssertionError('Unknown export must raise AttributeError')
"""
    subprocess.run([sys.executable, '-c', code], check=True,
                   cwd=Path(__file__).resolve().parents[2])


def test_public_control_exports_are_preserved():
    from lightweight_vla_adapter import LightweightVLAPipeline, gate_vla_proposal, advance_vla_control_plan
    from lightweight_vla_adapter.src.pipeline import LightweightVLAPipeline as pipeline
    from lightweight_vla_adapter.src.safety_bridge import gate_vla_proposal as gate, advance_vla_control_plan as advance
    assert LightweightVLAPipeline is pipeline
    assert gate_vla_proposal is gate
    assert advance_vla_control_plan is advance
