from copy import deepcopy
import pytest
from control.risk_continuity import RiskContinuityConfig, RiskContinuityFilter


def risk(level='low',gap=None,relative=0.):
    p={k:.025 for k in ('low','medium','high')};p[level]=.95
    return dict(risk_level=level,recommended_action=dict(low='keep_lane',medium='decelerate',high='emergency_brake')[level],
        reason_codes=[] if level=='low' else ['learned_visual_risk_'+level],probabilities=p,
        risk_evidence=dict(is_fresh=True,age_s=.05),
        physical_forward_radar=dict(nearest_distance_m=gap,nearest_relative_velocity_mps=relative,
            nearest_closing_distance_m=gap,nearest_closing_velocity_mps=max(0.,-relative)),
        lane_change=dict(left=dict(is_safe=False),right=dict(is_safe=False)))


def update(filt,raw,i,tracked=None):
    return filt.apply(raw,timestamp_s=i*.05,frame_id=f'f{i}',tracked_observation=tracked)


@pytest.mark.parametrize('gap,relative',[(None,0.),(40.,-.2),(25.,0.)])
def test_ordinary_following_and_grade_noise_does_not_chatter(gap,relative):
    filt=RiskContinuityFilter();update(filt,risk(gap=gap,relative=relative),0)
    for i in range(1,101):
        raw=risk('medium' if i%5 in (1,2) else 'low',gap=gap,relative=relative)
        original=deepcopy(raw)
        result,diag=update(filt,raw,i)
        assert result['risk_level']=='low'
        assert raw==original and result['lane_change']==raw['lane_change']
        assert not diag['external_event']


def test_persistent_visual_medium_is_not_hidden_indefinitely():
    filt=RiskContinuityFilter();update(filt,risk(),0)
    result=None
    for i in range(1,21):result,_=update(filt,risk('medium'),i)
    assert result['risk_level']=='medium' and result['recommended_action']=='decelerate'


@pytest.mark.parametrize('kind',['high','physical_caution','physical_emergency','ttc'])
def test_new_danger_bypasses_filter_on_the_first_frame(kind):
    filt=RiskContinuityFilter();update(filt,risk(gap=40.),0)
    raw=risk('high' if kind=='high' else 'medium' if kind=='physical_caution' else 'low',gap=40.)
    if kind=='physical_caution':raw['reason_codes']=['physical_forward_radar_caution_distance']
    if kind=='physical_emergency':raw['recommended_action']='emergency_brake'
    if kind=='ttc':raw=risk(gap=15.,relative=-6.)
    result,diag=update(filt,raw,1)
    assert diag['reason']=='immediate_safety_evidence'
    assert result['recommended_action']==('decelerate' if kind=='physical_caution' else 'emergency_brake')


@pytest.mark.parametrize('kind',['presence','gap_drop','closing_change','new_nearest'])
def test_external_event_reassesses_without_waiting(kind):
    filt=RiskContinuityFilter()
    tracked=dict(selected_track_id=1,nearest_obstacle=dict(track_id=1))
    update(filt,risk(gap=40. if kind!='presence' else None),0,tracked)
    raw=risk('medium',gap=35. if kind=='gap_drop' else 40.,relative=-2. if kind=='closing_change' else 0.)
    if kind=='new_nearest':tracked=dict(selected_track_id=1,nearest_obstacle=dict(track_id=2))
    result,diag=update(filt,raw,1,tracked)
    assert diag['external_event'] and result['risk_level']=='medium'


@pytest.mark.parametrize('fault',['stale','invalid_probabilities','time_gap','backwards_time'])
def test_unknown_or_discontinuous_observation_cannot_reuse_low_state(fault):
    filt=RiskContinuityFilter();update(filt,risk(),2)
    raw=risk('medium');i=3
    if fault=='stale':raw['risk_evidence']['is_fresh']=False
    if fault=='invalid_probabilities':raw['probabilities']['medium']=float('nan')
    if fault=='time_gap':i=30
    if fault=='backwards_time':i=1
    result,_=update(filt,raw,i)
    assert result['risk_level']=='medium'


def test_duplicate_frames_cannot_complete_confirmation():
    filt=RiskContinuityFilter();update(filt,risk(),0)
    for _ in range(100):result,_=update(filt,risk('medium'),1)
    assert result['risk_level']=='low'


def test_live_radar_fusion_contract_actually_enables_continuity():
    from universal_vla_controller import fuse_forward_radar_risk
    filt=RiskContinuityFilter()
    for i in range(3):
        raw=risk('low' if i==0 else 'medium')
        raw.pop('risk_evidence');raw.pop('physical_forward_radar')
        fused=fuse_forward_radar_risk(raw,dict(sensor_frame=i,measurement_timestamp_s=i*.05-.05,
            nearest_distance_m=None,obstacle_candidate_count=0),ego_speed_kmh=20.,decision_timestamp_s=i*.05)
        result,diag=update(filt,fused,i)
        assert result['risk_level']=='low'
        assert diag['reason']!='freshness_or_probability_unavailable'


def test_new_event_cannot_immediately_erase_established_danger():
    filt=RiskContinuityFilter();update(filt,risk('high',gap=40.),0)
    result,_=update(filt,risk(gap=60.),1)
    assert result['risk_level']=='high'
    for i in range(2,31):result,_=update(filt,risk(gap=60.),i)
    assert result['risk_level']=='low'


@pytest.mark.parametrize('name',list(RiskContinuityConfig.__dataclass_fields__))
def test_invalid_configuration_is_rejected(name):
    with pytest.raises(ValueError):RiskContinuityConfig(**{name:0.})
