from pathlib import Path
import json
import pytest
from scene_understanding.core.target_history import TargetHistory
from scene_understanding.realtime_perception.pipeline import RealtimePerceptionPipeline
from lightweight_vla_adapter.src.tracked_observation import TrackedObservation


def item(identity='a'):
    return dict(track_id=identity,category='vehicle',subtype='car',bbox_2d=[.1,.2,.3,.4],confidence=.9,age_frames=3)


def test_last_seen_retained_30_seconds_then_deleted():
    h=TargetHistory();h.update('front',[item()],10.)
    out=h.update('front',[],20.)['entries'][0]
    assert out['status']=='HISTORICAL' and not out['usable_as_current_observation']
    assert out['last_seen_s']==10. and out['expires_at_s']==40.
    assert h.snapshot('front',40.)['entries']
    assert not h.snapshot('front',40.01)['entries']


def test_same_identity_refreshes_expiry_but_new_id_not_auto_merged():
    h=TargetHistory();h.update('front',[item()],0.)
    h.update('front',[],10.)
    out=h.update('front',[item()],20.)['entries'][0]
    assert out['first_seen_s']==0. and out['expires_at_s']==50.
    out=h.update('front',[item('b')],21.)['entries']
    assert len(out)==2
    assert [e for e in out if e['track_id']=='a'][0]['status']=='HISTORICAL'


def test_camera_namespaces_and_reset():
    h=TargetHistory();h.update('front',[item()],1.);h.update('rear',[item()],1.)
    h.update('front',[],2.)
    assert len(h.entries)==2
    h.reset();assert not h.entries


def test_lagged_camera_does_not_reset_other_camera_history():
    h=TargetHistory();h.update('front',[item()],2.)
    h.update('rear',[item()],1.9)
    assert len(h.entries)==2
    assert h.snapshot('front',2.)['entries'][0]['last_seen_s']==2.
    h.update('front',[item()],1.8)
    assert h.snapshot('front',2.)['entries'][0]['last_seen_s']==2.


def test_clock_rewind_clears_previous_scene():
    h=TargetHistory();h.update('front',[item()],20.)
    assert not h.snapshot('front',1.)['entries']


def test_snapshot_copy_does_not_mutate_evidence():
    h=TargetHistory();out=h.update('front',[item()],0.)
    out['entries'][0]['last_observation']['bbox_2d'][0]=.8
    assert h.snapshot('front',1.)['entries'][0]['last_observation']['bbox_2d'][0]==.1


def test_absent_at_same_timestamp_not_live():
    h=TargetHistory();h.update('front',[item()],0.)
    assert h.update('front',[],0.)['entries'][0]['status']=='HISTORICAL'


def test_expired_identity_starts_new_lifecycle():
    h=TargetHistory();h.update('front',[item()],0.)
    assert h.update('front',[item()],31.)['entries'][0]['first_seen_s']==31.


def test_limit_not_silent_early_forgetting():
    h=TargetHistory(maximum_entries=1);h.update('front',[item()],0.)
    with pytest.raises(RuntimeError):h.update('front',[item('b')],1.)


def test_pipeline_history_not_inserted_into_current_tracks(tmp_path):
    from PIL import Image
    from jsonschema import Draft202012Validator
    class Detector:
        def detect(self,image):return [],0.
    class Tracker:
        def __init__(self):self.count=0
        def reset(self):self.count=0
        def update(self,detections):
            self.count+=1
            if self.count!=1:return []
            x=item();x.pop('bbox_2d');x['bbox_xyxy']=[1.,2.,3.,4.];return [x]
    path=tmp_path/'image.png';Image.new('RGB',(10,10)).save(path)
    p=RealtimePerceptionPipeline(Detector(),Tracker())
    def run(t):return p.process(image_path=path,frame_id='f',source='carla',camera_name='front',timestamp_s=t)
    first=run(0.);second=run(20.);third=run(31.)
    assert first['tracks'] and not second['tracks']
    assert second['target_history']['entries'][0]['status']=='HISTORICAL'
    assert not third['target_history']['entries']
    schema=json.loads((Path(__file__).resolve().parents[1]/'schemas/perception_frame.schema.json').read_text())
    for value in (first,second,third,run(None)):Draft202012Validator(schema).validate(value)


def test_radar_dynamic_prediction_expires_but_history_survives():
    t=TrackedObservation()
    ego=dict(x=0.,y=0.,vx_mps=0.,vy_mps=0.)
    for i in range(1,5):
        r=dict(sensor_frame=i,measurement_timestamp_s=i*.1,measurement_pose=dict(x=0.,y=0.,yaw_deg=0.),
            tracking_points=[dict(distance_m=20.+d,azimuth_deg=a,altitude_deg=0.,relative_velocity_mps=0.) for d,a in [(-.05,-.2),(0.,0.),(.05,.2)]])
        out=t.update(r,ego=ego,route_points=[[0,0],[80,0]],now_s=i*.1,frame=i)
    for now in (.6,2.,20.,30.4):
        out=t.update({},ego=ego,route_points=[[0,0],[80,0]],now_s=now,frame=100+int(now*10))
        assert out['target_history']['entries']
    assert out['selected'] is None and not out['entities']
    out=t.update({},ego=ego,route_points=[[0,0],[80,0]],now_s=30.5,frame=500)
    assert not out['target_history']['entries']
