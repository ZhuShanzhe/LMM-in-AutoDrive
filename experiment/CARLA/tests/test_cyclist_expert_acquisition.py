import numpy as np
import pytest
from scene_understanding.realtime_perception.detector import Detection
from control.passing_conditions import (merge_cyclist_expert, pending_vehicle_passing_lane,
    require_acquired_vehicle_before_lane_change, PassingConditionObserver,
    cyclist_expert_view_names)


def detection(category,confidence=.5,box=(10,10,30,60)):
    return Detection(box,confidence,category,category,2)


def steps():
    return [dict(step_id='slow',action='ADJUST_SPEED'),
        dict(step_id='merge',action='CHANGE_LANE',parameters={'target_ref':'bike'},depends_on=['slow']),
        dict(step_id='pass',action='PASS_BY',parameters={'condition':'CYCLIST'},depends_on=['merge'])]


def test_cyclist_merge_requires_same_frame_referent():
    assert pending_vehicle_passing_lane(steps(),'slow')=='merge'
    proof={'pass':dict(valid=True,target_id='bike',target_observation={'measured_in_current_frame':False})}
    result=require_acquired_vehicle_before_lane_change({'can_execute':True},steps(),'merge',proof,maneuver_started=False)
    assert result['can_execute'] is False
    proof['pass']['target_observation']['measured_in_current_frame']=True
    proof['pass']['target_observation'].update(ahead_m=23.2,lateral_m=3.4,
        observed_categories=['cyclist'],current_history_samples=1)
    assert require_acquired_vehicle_before_lane_change({'can_execute':True},steps(),'merge',proof,
        maneuver_started=False)['can_execute'] is True


def test_cyclist_target_requires_current_measured_plausible_geometry():
    proof={'pass':dict(valid=True,target_id='bike',target_observation={
        'measured_in_current_frame':True,'motion_history_established':False,'current_history_samples':1,
        'ahead_m':6.4,'lateral_m':12.5,'observed_categories':['cyclist']})}
    result=require_acquired_vehicle_before_lane_change({'can_execute':True},steps(),'merge',proof,
        maneuver_started=False)
    assert result['can_execute'] is False
    assert 'passing_target_not_acquired' in result['reason_codes']


def test_cyclist_acquisition_uses_road_local_geometry_on_curves():
    proof={'pass':dict(valid=True,target_id='bike',target_observation={
        'measured_in_current_frame':True,'ahead_m':6.4,'lateral_m':12.5,
        'acquisition_geometry_valid':True,'road_ahead_m':23.2,'road_lateral_m':3.4,
        'observed_categories':['cyclist']})}
    result=require_acquired_vehicle_before_lane_change({'can_execute':True},steps(),'merge',proof,
        maneuver_started=False)
    assert result['can_execute'] is True


@pytest.mark.parametrize('observation',[
    {'measured_in_current_frame':True,'ahead_m':23.2,'lateral_m':3.4,'observed_categories':['cyclist']},
    {'measured_in_current_frame':True,'ahead_m':-1.,'lateral_m':0.,'observed_categories':['cyclist']},
    {'measured_in_current_frame':True,'ahead_m':36.,'lateral_m':0.,'observed_categories':['cyclist']},
    {'measured_in_current_frame':True,'ahead_m':23.2,'lateral_m':3.4,'observed_categories':['pedestrian']},
])
def test_cyclist_target_geometry_and_category_gate(observation):
    proof={'pass':dict(valid=True,target_id='bike',target_observation=observation)}
    result=require_acquired_vehicle_before_lane_change({'can_execute':True},steps(),'merge',proof,
        maneuver_started=False)
    expected=observation['ahead_m']==23.2 and observation['observed_categories']==['cyclist']
    assert result['can_execute'] is expected


@pytest.mark.parametrize('label',['pedestrian','motorcycle','cyclist'])
def test_explicit_expert_cyclist_resolves_duplicate_rider_only(label):
    existing=[detection(label,.8),detection('vehicle'),detection('pedestrian',box=(90,10,110,60))]
    expert=[detection('cyclist',.3)]
    merged=merge_cyclist_expert(existing,expert)
    assert [d.category for d in merged]==['vehicle','pedestrian','cyclist']
    assert merged[-1].confidence==.3


@pytest.mark.parametrize('label,confidence',[('pedestrian',.9),('motorcycle',.9),('cyclist',.24)])
def test_expert_cannot_promote_generic_or_low_confidence_labels(label,confidence):
    existing=[detection('motorcycle')]
    assert merge_cyclist_expert(existing,[detection(label,confidence)])==existing


def test_disabled_expert_is_not_called_or_cached_as_current():
    class Detector:
        def __init__(self):self.calls=0
        def detect(self,image):self.calls+=1;return [detection('cyclist')],0.
    primary=Detector();expert=Detector()
    observer=PassingConditionObserver(primary,cyclist_detector=expert)
    usable=[('front',np.zeros((80,120,3),dtype=np.uint8),np.eye(4),90)]
    observer.detect_views(usable,cyclist_expert=False)
    assert expert.calls==0
    observer.detect_views(usable,cyclist_expert=True)
    assert expert.calls==1
    observer.detect_views(usable,cyclist_expert=False)
    assert expert.calls==1


def test_specialist_runs_only_selected_views_without_erasing_other_view_detections():
    class Detector:
        def __init__(self,category):self.category=category;self.calls=0
        def detect(self,image):
            self.calls+=1
            return ([detection(self.category)] if self.category else []),0.
    primary=Detector('cyclist');expert=Detector(None)
    observer=PassingConditionObserver(primary,cyclist_detector=expert)
    usable=[(name,np.zeros((80,120,3),dtype=np.uint8),np.eye(4),90)
            for name in ('front','left','right','rear')]
    outputs=observer.detect_views(usable,cyclist_expert=True,
        cyclist_expert_views=('front',))
    assert primary.calls==4 and expert.calls==1
    assert [d.category for d in outputs[0]]==[]
    assert all([d.category for d in output]==['cyclist'] for output in outputs[1:])


def test_cyclist_specialist_uses_current_target_view_and_falls_back_when_unknown():
    doc=dict(request_id='r',intent=dict(steps=[dict(step_id='pass',action='PASS_BY',
        parameters=dict(condition='CYCLIST'))]))
    views=('front','left','right','rear')
    assert cyclist_expert_view_names(doc,views,{}, {},timestamp_s=1.,ego_rear_m=1.8)==views
    bound={('r','pass'):dict(target_id='bike')}
    tracks={'bike':dict(last_seen_s=1.,ahead_m=12.,image_aliases={('front','track_1'):1.})}
    assert cyclist_expert_view_names(doc,views,bound,tracks,
        timestamp_s=1.,ego_rear_m=1.8)==('front',)
    tracks['bike'].update(ahead_m=-8.,image_aliases={('rear','track_2'):1.})
    assert cyclist_expert_view_names(doc,views,bound,tracks,
        timestamp_s=1.,ego_rear_m=1.8)==('rear',)
    tracks['bike'].update(ahead_m=1.,image_aliases={('front','track_1'):1.})
    assert cyclist_expert_view_names(doc,views,bound,tracks,
        timestamp_s=1.,ego_rear_m=1.8)==views
    tracks['bike']['last_seen_s']=.5
    assert cyclist_expert_view_names(doc,views,bound,tracks,
        timestamp_s=1.,ego_rear_m=1.8)==views


def test_non_cyclist_pass_skips_specialist_head():
    doc=dict(request_id='r',intent=dict(steps=[dict(step_id='pass',action='OVERTAKE',
        parameters=dict(condition='SLOW_VEHICLE'))]))
    assert cyclist_expert_view_names(doc,('front','rear'),{}, {},
        timestamp_s=1.,ego_rear_m=1.8)==()


def test_partial_coco_bicycle_is_not_a_second_cyclist_identity():
    base=[detection('cyclist',.8,box=(10,40,30,60))]
    expert=[detection('cyclist',.7)]
    assert len(merge_cyclist_expert(base,expert))==1


def test_adjacent_bicycle_is_not_suppressed_by_containment_rule():
    base=[detection('cyclist',.8,box=(25,40,45,60))]
    expert=[detection('cyclist',.7)]
    assert len(merge_cyclist_expert(base,expert))==2


def test_specialized_bicycle_head_does_not_union_unconfirmed_base_bicycles():
    base=[detection('cyclist'),detection('vehicle'),detection('pedestrian',box=(90,10,110,60))]
    assert [d.category for d in merge_cyclist_expert(base,[],authoritative=True)]==['vehicle','pedestrian']


def test_dependent_cyclist_pass_is_an_explicit_referent_without_lane_target_ref():
    graph=steps();graph[1]['parameters']={'direction':'LEFT'}
    assert pending_vehicle_passing_lane(graph,'slow')=='merge'
    contract={'can_execute':True,'road_legal':True}
    result=require_acquired_vehicle_before_lane_change(contract,graph,'merge',{},maneuver_started=False)
    assert result['can_execute'] is False
    assert 'passing_target_not_acquired' in result['reason_codes']


def test_explicit_cyclist_overtake_plan_acquires_target_before_issuing_left_merge():
    import json
    from pathlib import Path
    from scene2_runtime_interface import build_scheduled_driving_intent
    root=Path(__file__).resolve().parents[1]
    command=next(c for c in json.loads((root/'configs/scene_2_town05_runtime.json').read_text())['commands']
        if c['id']=='s2_t05_cmd_07')
    command=dict(command,steps=['ADJUST_SPEED:DECREASE','CHANGE_LANE:LEFT_WHEN_SAFE',
                               'PASS_BY:CYCLIST','CHANGE_LANE:RIGHT_WHEN_SAFE'])
    document=build_scheduled_driving_intent(command,1,1761.,1.)
    intent_steps=document['intent']['steps']
    merge=intent_steps[1]
    assert pending_vehicle_passing_lane(intent_steps,intent_steps[0]['step_id'])==merge['step_id']
    result=require_acquired_vehicle_before_lane_change({'can_execute':True},intent_steps,merge['step_id'],
        {intent_steps[2]['step_id']:dict(valid=True,target_id='bike',target_observation={
            'measured_in_current_frame':True,'motion_history_established':False,'current_history_samples':1,
            'ahead_m':23.2,'lateral_m':3.4,'observed_categories':['cyclist']})},
        maneuver_started=False)
    assert result['can_execute'] is True
