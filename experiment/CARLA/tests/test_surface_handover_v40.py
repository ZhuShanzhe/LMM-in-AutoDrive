import importlib.util
from pathlib import Path
import numpy as np
import pytest

from control import passing_conditions as module


def fixtures():
    x,y=np.meshgrid(np.linspace(-2,2,10),np.linspace(-1.5,1.5,10))
    old=np.stack([x.ravel(),y.ravel(),np.ones(x.size)],axis=1)
    a,b=np.meshgrid(np.linspace(-2,2,10),np.linspace(1.5,2.5,10))
    new=np.stack([a.ravel(),b.ravel(),np.ones(a.size)],axis=1)
    track=dict(category='vehicle',position=np.array([0.,0.]),velocity=np.zeros(2),last_seen_s=10.,surface=old)
    measurement=dict(category='vehicle',position=np.median(new,axis=0),surface=new,box_cropped=True)
    return track,measurement


def test_fresh_partial_surface_handover():
    t,m=fixtures()
    assert module.PassingMemory.surface_association(t,m,10.1,np.array([1.,0.]),np.array([0.,1.])) is not None


@pytest.mark.parametrize('change',['stale','uncropped','separate_lane','unknown_surface','nonfinite','oversized','category','reverse_time'])
def test_no_unsupported_handover(change):
    t,m=fixtures();now=10.1
    if change=='stale':now=10.3
    if change=='uncropped':m['box_cropped']=False
    if change=='separate_lane':m['surface'][:,1]+=1.5;m['position']=np.median(m['surface'],axis=0)
    if change=='unknown_surface':t['surface']=None
    if change=='nonfinite':m['surface'][0,0]=np.nan
    if change=='oversized':t['surface'][:,0]*=5.
    if change=='category':m['category']='cyclist'
    if change=='reverse_time':now=9.9
    assert module.PassingMemory.surface_association(t,m,now,np.array([1.,0.]),np.array([0.,1.])) is None


def test_old_gate_unchanged_without_surfaces():
    t,m=fixtures();m['position']=np.array([1.,.5]);t['surface']=None
    assert module.PassingMemory.surface_association(t,m,10.1,np.array([1.,0.]),np.array([0.,1.])) is not None
