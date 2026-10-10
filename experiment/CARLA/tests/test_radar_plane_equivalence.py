import ast
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
import radar_ground_filter as optimized


def reference_module():
    source=Path(optimized.__file__).read_text()
    tree=ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='cKDTree':node.keywords=[]
    namespace={};exec(compile(tree,'<reference-ground-filter>','exec'),namespace)
    def reference_fit(a,z,point,matrix):
        fit,_,rank,singular=np.linalg.lstsq(a,z,rcond=None)
        return fit,rank,singular
    namespace['_fit_local_plane']=reference_fit
    return namespace


@pytest.mark.parametrize('seed',range(20))
def test_small_plane_solve_preserves_ground_rejection(seed):
    random=np.random.default_rng(seed)
    x,y=np.meshgrid(np.linspace(-10.,35.,101),np.linspace(-6.,6.,51))
    slope=random.uniform(-.2,.2)
    cloud=np.column_stack((x.ravel(),y.ravel(),(slope*x+random.normal(0.,.003,x.shape)).ravel()))
    matrix=np.eye(4);matrix[2,3]=1.6
    points=[]
    for distance in random.uniform(5.,30.,30):
        altitude=np.degrees(np.arctan((slope*distance-1.6+random.choice([0.,.019,.021,.3]))/distance))
        points.append(dict(distance_m=float(distance),azimuth_deg=0.,altitude_deg=float(altitude),
            relative_velocity_mps=-5.,closing_speed_mps=5.,relative_height_m=0.))
    obs=dict(schema_version='physical_front_radar/1.0',_ground_candidates=points)
    expected=reference_module()['filter_ground'](deepcopy(obs),cloud,matrix)
    assert optimized.filter_ground(deepcopy(obs),cloud,matrix)==expected


@pytest.mark.parametrize('boundary',[.02,.015,.04])
def test_reference_fallback_preserves_plane_boundary_values(boundary):
    random=np.random.default_rng(42)
    a=np.column_stack((random.uniform(-2.,2.,(100,2)),np.ones(100)))
    z=np.zeros(100);point=np.array([0.,0.,boundary]);matrix=np.eye(4);matrix[2,3]=1.6
    actual=optimized._fit_local_plane(a,z,point,matrix)
    expected=np.linalg.lstsq(a,z,rcond=None)
    assert np.allclose(actual[0],expected[0]) and actual[1]==expected[2]
