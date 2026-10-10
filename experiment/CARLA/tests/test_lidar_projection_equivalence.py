"""Reference equivalence for the allocation-reduced corridor calculation."""
import ast
from pathlib import Path
import numpy as np
from lightweight_vla_adapter.src import risk_behavior as module


def reference(points, route, **kwargs):
    source = Path(module.__file__).read_text()
    source = source.replace('from .route_projection_compiled import project_points_to_route, projection_available', '')
    source = source.replace(
        'dx, dy = x-first[0], y-first[1]\n            fraction = np.clip((dx*delta[0]+dy*delta[1]) / length**2, 0., 1.)\n            rx, ry = dx-fraction*delta[0], dy-fraction*delta[1]\n            distance = np.sqrt(rx*rx+ry*ry)',
        'fraction = np.clip(((xyz[:, :2] - first[:2])*delta).sum(axis=1) / length**2, 0., 1.)\n            distance = np.linalg.norm(xyz[:, :2] - first[:2] - fraction[:, None]*delta, axis=1)')
    assert 'np.linalg.norm(xyz[:, :2] - first[:2]' in source
    source = source.replace(
        "lower, upper = np.searchsorted(ground_progress, [start, start+1.5], side='left')\n        count = int(upper-lower); counts.append(count)",
        'support = ground & (progress-ego_front_m >= start) & (progress-ego_front_m < start+1.5)\n        count = int(support.sum()); counts.append(count)')
    namespace = dict(projection_available=lambda:False)
    exec(compile(source, '<reference>', 'exec'), namespace)
    return namespace['lidar_corridor_evidence'](points, route, **kwargs)


def test_corridor_equivalence_random_geometry_and_boundary_points():
    rng = np.random.default_rng(20261006)
    for case in range(45):
        x = np.linspace(0., 60., 32)
        road = np.column_stack((x, 2.*np.sin(x/20+case), .03*x))
        cloud = rng.uniform([-10,-10,-1],[65,10,4],(3000,3))
        ground = np.repeat(road,5,axis=0)
        ground[:,1] += rng.uniform(-1.,1.,len(ground))
        cloud = np.concatenate((cloud,ground))
        kwargs = dict(half_width_m=1.25,ego_front_m=2.,sensor_frame=case,observed_at_s=case*.05)
        assert module.lidar_corridor_evidence(cloud,road,**kwargs)==reference(cloud,road,**kwargs)
    road = np.array([[0.,0.,0.],[60.,0.,0.]])
    values = np.r_[np.arange(5.,61.,.25),[5.,6.5,6.5-1e-10,6.5+1e-10]]
    cloud = np.column_stack((values,np.zeros(len(values)),np.zeros(len(values))))
    kwargs = dict(half_width_m=1.,ego_front_m=2.,sensor_frame=0,observed_at_s=0.)
    assert module.lidar_corridor_evidence(cloud,road,**kwargs)==reference(cloud,road,**kwargs)


def test_compact_identity_preserves_command_semantics():
    path = Path(__file__).resolve().parents[1]/'universal_vla_controller.py'
    source = path.read_text()
    fn = next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)
              and n.name=='command_memory_identity')
    import hashlib,json
    namespace=dict(hashlib=hashlib,json=json,Mapping=dict,Any=object)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),namespace)
    identity=namespace['command_memory_identity']
    first=dict(id='x',target_speed_kmh=30.,driving_intent={'steps':['WAIT','PROCEED']})
    assert identity(first)==identity(dict(reversed(list(first.items()))))
    assert identity(first)!=identity(dict(first,target_speed_kmh=40.))
    assert len(identity(first))==79
