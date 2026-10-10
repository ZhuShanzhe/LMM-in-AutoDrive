import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


tree=ast.parse((Path(__file__).resolve().parents[1]/'run_complex_avoidance_town05.py').read_text())
value=next(keyword.value for node in ast.walk(tree) if isinstance(node,ast.Call)
    for keyword in node.keywords if keyword.arg=='policy_source')
expression=compile(ast.Expression(value),'<policy-source>','eval')


@pytest.mark.parametrize('vla,external,source',[
    (True,True,'VLA_MODEL'),(True,False,'VLA_MODEL'),
    (False,True,'EXTERNAL'),(False,False,'NON_VLA_CONTROL'),
])
def test_recorded_policy_source_matches_actual_controller_priority(vla,external,source):
    assert eval(expression,{'vla_enabled':vla,'args':SimpleNamespace(external_ego_control=external)})==source
