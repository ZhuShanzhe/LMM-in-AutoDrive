import numpy as np
import torch
from control.signal_lamp_state import LampStateNet,LampStateClassifier,project_head,head_crop,STATES


def vertices(x=10.):
    return [[a,b,c] for a in (x-.1,x+.1) for b in (-.2,.2) for c in (1.,1.8)]


def test_projection_works_for_different_camera_head_distances():
    close=project_head(vertices(10.),np.eye(4),1280,1280,100.)
    far=project_head(vertices(40.),np.eye(4),1280,1280,100.)
    assert close and far and close[3]-close[1]>far[3]-far[1]
    assert head_crop(np.zeros((1280,1280,3),np.uint8),close).shape==(64,48,3)


def test_invisible_nonfinite_and_tiny_projections_reject():
    assert project_head(vertices(-10.),np.eye(4),1280,1280,100.) is None
    assert project_head(vertices(400.),np.eye(4),1280,1280,100.) is None
    bad=vertices();bad[0][0]=float('nan')
    assert project_head(bad,np.eye(4),1280,1280,100.) is None


def test_green_prediction_requires_current_green_pixels(tmp_path):
    net=LampStateNet()
    with torch.no_grad():
        for p in net.parameters():p.zero_()
        net.head[-1].bias[3]=12.
    checkpoint=tmp_path/'lamp.pt'
    torch.save(dict(schema_version='lamp_state/1.0',classes=list(STATES),state_dict=net.state_dict()),checkpoint)
    classifier=LampStateClassifier(checkpoint,'cpu')
    black=np.zeros((64,48,3),np.uint8)
    green=black.copy();green[40:50,20:30]=[0,255,0]
    red=black.copy();red[10:20,20:30]=[255,0,0]
    uniform_green=np.full_like(black,[0,255,0])
    results=classifier.predict([black,green,red,uniform_green])
    assert [r['state'] for r in results]==['UNKNOWN','GREEN','UNKNOWN','UNKNOWN']


def test_static_head_conflict_is_not_resolved_by_highest_confidence():
    from control.map_signal_observer import MapSignalObserver
    from types import SimpleNamespace
    observer=MapSignalObserver.__new__(MapSignalObserver)
    observer.camera=SimpleNamespace(attributes={'fov':'100'})
    observer.static_heads={'signal-a':[vertices(10.),vertices(12.)]}
    observer.state_classifier=SimpleNamespace(predict=lambda crops:[
        dict(state='RED',color_confidence=.95),dict(state='GREEN',color_confidence=.99)])
    state,confidence,items=observer._observe_static_heads('signal-a',np.zeros((1280,1280,3),np.uint8),np.eye(4))
    assert state=='UNKNOWN' and confidence==0 and len(items)==2
    assert observer._observe_static_heads('unmapped',np.zeros((1280,1280,3),np.uint8),np.eye(4))[:2]==('UNKNOWN',0.)
