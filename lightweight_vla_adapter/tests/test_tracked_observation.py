import math
import numpy as np
import pytest
from lightweight_vla_adapter.src.tracked_observation import RouteGeometry, TrackedObservation
from lightweight_vla_adapter.src.task_event_memory import TaskEventMemory


def radar(frame, time, position=20., relative=-2., angle=0., pose_x=0.):
    return dict(sensor_frame=frame,measurement_timestamp_s=time,measurement_pose=dict(x=pose_x,y=0.,yaw_deg=0.),
        tracking_points=[dict(distance_m=position+d,azimuth_deg=angle+a,altitude_deg=0.,relative_velocity_mps=relative)
            for d,a in [(-.05,-.2),(0.,0.),(.05,.2)]])


def update(tracker, r, time, frame, **kw):
    return tracker.update(r,ego=dict(x=0.,y=0.,vx_mps=5.,vy_mps=0.,half_length_m=2.,corridor_half_width_m=1.6),
        route_points=kw.pop('route',[[0.,0.],[100.,0.]]),now_s=time,frame=frame,**kw)


def test_bend_uses_arc_length_not_heading_projection():
    route=RouteGeometry([[0,0],[10,0],[10,10]])
    p=route.project([10,8])
    assert p['s']==pytest.approx(18.)
    assert p['distance']==0


def test_degenerate_route_rejected():
    with pytest.raises(ValueError):RouteGeometry([[0,0],[0,0]])


def test_empty_detection_is_not_clear_road():
    r=radar(1,.1);r['tracking_points']=[]
    assert update(TrackedObservation(),r,.1,1)['status']=='UNKNOWN'
    assert update(TrackedObservation(),r,.1,1,coverage_clear=True)['status']=='CLEAR'


def test_timestamp_invalid_or_future_is_not_fresh():
    for time in (-1.,2.,float('nan')):
        out=update(TrackedObservation(),radar(1,time),1.,1)
        assert not out['radar_fresh'] and out['status']=='UNKNOWN'


def test_malformed_points_do_not_become_tracks():
    r=radar(1,.1);r['tracking_points']=[{},None,dict(distance_m='bad')]
    out=update(TrackedObservation(),r,.1,1)
    assert out['status']=='UNKNOWN' and not out['entities']


def test_target_continuity_and_velocity():
    t=TrackedObservation(); ids=[]
    for i in range(1,41):
        # Ego is stationary here; radar relative velocity must match world motion.
        r=radar(i,i*.1,position=20.+.3*i,relative=3.)
        out=t.update(r,ego=dict(x=0.,y=0.,vx_mps=0.,vy_mps=0.),route_points=[[0,0],[100,0]],now_s=i*.1,frame=i)
        if out['selected']:ids.append(out['selected_track_id'])
    assert len(set(ids))==1
    assert out['selected']['velocity_xy_mps'][0]==pytest.approx(3.,abs=.2)


def test_duplicate_frames_do_not_confirm_track():
    t=TrackedObservation()
    for _ in range(6):out=update(t,radar(1,.1),.1,1)
    assert out['selected'] is None
    assert t.tracks[0].hits==1


def test_short_occlusion_predicted_then_expires():
    t=TrackedObservation()
    for i in range(1,5):update(t,radar(i,.1*i,position=20+.3*i,relative=-2.),i*.1,i)
    out=update(t,{},.65,7)
    assert out['status']=='PREDICTED'
    out=update(t,{},1.,10)
    assert out['status']=='UNKNOWN' and not out['entities']


def test_immediate_missing_measurement_is_labeled_prediction():
    t=TrackedObservation()
    for i in range(1,5):update(t,radar(i,.1*i),i*.1,i)
    out=update(t,{},.45,5)
    assert out['status']=='PREDICTED'
    assert out['selected']['measured_at_s']==pytest.approx(.4)


def test_new_frame_with_old_timestamp_does_not_rewind_filter():
    t=TrackedObservation()
    for i in range(1,5):update(t,radar(i,.1*i),i*.1,i)
    out=update(t,radar(5,.35),.45,5)
    assert not out['radar_fresh']
    assert t.tracks[0].time==pytest.approx(.4)


def test_adjacent_track_kept_but_not_selected():
    t=TrackedObservation()
    for i in range(1,5):out=update(t,radar(i,i*.1,angle=12.),i*.1,i)
    assert out['entities'] and out['selected'] is None


def test_full_fov_bend_target_selected():
    t=TrackedObservation()
    angle=12.;y=20*math.sin(math.radians(angle));x=20*math.cos(math.radians(angle))
    for i in range(1,5):out=update(t,radar(i,i*.1,angle=angle),i*.1,i,route=[[0,0],[x,y],[x+40,y+8]])
    assert out['selected'] is not None


def test_reset_on_time_reversal():
    t=TrackedObservation()
    for i in range(1,5):update(t,radar(i,i*.1),i*.1,i)
    out=update(t,{},.05,0)
    assert not out['entities']


def test_missing_route_never_generates_ttc_or_lead():
    t=TrackedObservation()
    for i in range(1,5):out=update(t,radar(i,i*.1),i*.1,i,route=[])
    assert out['status']=='UNKNOWN' and out['selected'] is None
    assert all(x['closing_ttc_s'] is None for x in out['entities'])


def test_new_obstacle_does_not_erase_target_but_is_reported():
    t=TrackedObservation()
    for i in range(1,5):out=update(t,radar(i,i*.1,position=25.,relative=-5.),i*.1,i)
    original=out['selected_track_id']
    for i in range(5,9):
        r=radar(i,i*.1,position=25.,relative=-5.)
        r['tracking_points']+=radar(i,i*.1,position=10.,relative=-5.)['tracking_points']
        out=update(t,r,i*.1,i)
    assert out['selected_track_id']==original
    assert out['nearest_obstacle']['route_gap_m']<out['selected']['route_gap_m']-10
    obstacle=out['nearest_obstacle']['track_id']
    out=update(t,r,.8,8,requested_track_id=obstacle)
    assert out['selected_track_id']==obstacle
    out=update(t,r,.8,8,requested_track_id='missing-target')
    assert out['selected'] is None and not out['requested_track_resolved']


def test_event_retained_when_acceleration_superseded():
    m=TaskEventMemory();m.activate('follow-1','FOLLOW',dict(target='car-a'),0.)
    m.behavior_changed('follow-1','ACCELERATE',dict(gap=35.),0.)
    assert not m.behavior_changed('follow-1','DECELERATE',dict(gap=15.),.2)
    m.behavior_changed('follow-1','DECELERATE',dict(gap=15.),1.)
    event=m.decision_context()['active_events'][0]
    assert event['kind']=='FOLLOW' and event['current_behavior']=='DECELERATE'
    assert m.decision_context()['recent_transitions'][-1]['previous_behavior_status']=='SUPERSEDED'
    assert m.decision_context()['authority']=='DECISION_CONTEXT_ONLY'


def test_delayed_real_measurement_is_cached_without_renewing_on_prediction():
    t=TrackedObservation()
    for i in range(1,6):
        stamp=i*.1
        out=update(t,radar(i,stamp,position=20.+.3*i),stamp+.1,i+1)
    assert out['status']=='PREDICTED'
    entry=out['target_history']['entries'][0]
    assert entry['last_seen_s']==pytest.approx(.5)
    assert entry['status']=='HISTORICAL' and not entry['usable_as_current_observation']
    assert entry['last_observation']['position_xy_m'][0] < out['selected']['position_xy_m'][0]
    out=update(t,radar(5,.5),.7,7)
    assert out['target_history']['entries'][0]['last_seen_s']==pytest.approx(.5)
    assert update(t,{},30.5,305)['target_history']['entries']
    assert not update(t,{},30.51,306)['target_history']['entries']


def test_urgent_memory_update_and_explicit_cancel():
    m=TaskEventMemory();m.activate('x','FOLLOW',{},0.)
    m.behavior_changed('x','ACCELERATE',{},0.)
    assert m.behavior_changed('x','EMERGENCY_BRAKE',{},.1)
    m.finish('x',reason='CANCELLED',timestamp_s=.2)
    assert not m.decision_context()['active_events']


def test_memory_never_silently_drops_unfinished_event():
    m=TaskEventMemory(maximum_events=1);m.activate('x','FOLLOW',{},0.)
    with pytest.raises(RuntimeError):m.activate('y','TURN',{},1.)
    with pytest.raises(ValueError):m.activate('x','TURN',{},1.)


def test_event_survives_expired_target_without_retaining_physical_snapshot():
    m=TaskEventMemory();m.activate('x','FOLLOW',{},0.)
    m.behavior_changed('x','ACCELERATE',dict(status='TRACKED',selected=dict(track_id='a',measured_at_s=0.,position_xy_m=[1,2])),0.)
    entry=m.decision_context(31.)['active_events'][0]
    assert entry['kind']=='FOLLOW'
    assert entry['boundary_observation']['target_ref'] is None
    assert 'position_xy_m' not in entry['boundary_observation']


def test_event_reference_stays_when_observer_has_recently_seen_target():
    m=TaskEventMemory();m.activate('x','FOLLOW',{},0.)
    m.behavior_changed('x','ACCELERATE',dict(status='TRACKED',selected=dict(track_id='a',measured_at_s=0.)),0.)
    context=m.decision_context(31.,dict(entries=[dict(track_id='a',last_seen_s=30.)]))
    assert context['active_events'][0]['boundary_observation']['target_ref']=='a'
