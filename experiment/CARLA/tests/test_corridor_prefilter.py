import ast
import inspect
import numpy as np
import pytest
from lightweight_vla_adapter.src import lane_risk_contract as new_lane
from lightweight_vla_adapter.src import risk_behavior as new_risk


def without_rectangle(module,name):
    tree=ast.parse(inspect.getsource(getattr(module,name)))
    body=tree.body[0].body
    start=next(i for i,node in enumerate(body) if isinstance(node,ast.Assign)
        and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='radius')
    assert [node.targets[0].id for node in body[start:start+4]]==['radius','lower','upper','xyz']
    del body[start:start+4]
    namespace=dict(vars(module))
    exec(compile(tree,'<projection_without_rectangle>','exec'),namespace)
    return namespace[name]


old_risk=without_rectangle(new_risk,'lidar_corridor_evidence')
old_lane=without_rectangle(new_lane,'measured_lane_clearance')


@pytest.mark.parametrize('seed',range(50))
@pytest.mark.parametrize('curve',[False,True])
def test_projection_rectangle_preserves_complete_safety_evidence(seed,curve):
    rng=np.random.default_rng(seed)
    x=np.linspace(-10.,60.,36)
    road=np.column_stack([x,np.sin(x/20.)*8. if curve else np.zeros_like(x),x*.08])
    points=rng.uniform([-70.,-70.,-5.],[70.,70.,10.],(5000,3))
    kwargs=dict(half_width_m=1.4,ego_front_m=2.2,sensor_frame=8,observed_at_s=1.)
    assert new_risk.lidar_corridor_evidence(points,road,**kwargs)==old_risk(points,road,**kwargs)
    assert new_lane.measured_lane_clearance(points,road,[0.,0.,0.])==old_lane(points,road,[0.,0.,0.])
