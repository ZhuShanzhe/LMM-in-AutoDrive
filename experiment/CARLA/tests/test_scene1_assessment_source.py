import ast
from pathlib import Path


def test_independent_assessment_prefers_actual_map_bound_execution_config():
    tree=ast.parse((Path(__file__).resolve().parents[1]/'run_control_experiment.py').read_text())
    calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call)
        and isinstance(node.func,ast.Name) and node.func.id=='attach_episode']
    assert len(calls)==1
    source=calls[0].args[5]
    assert isinstance(source,ast.BoolOp) and isinstance(source.op,ast.Or)
    assert isinstance(source.values[0],ast.Name) and source.values[0].id=='scenario_config_path'
