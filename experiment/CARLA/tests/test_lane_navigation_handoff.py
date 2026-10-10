import math
from types import SimpleNamespace as NS
import pytest
from control.generic_route_pid import GenericRoutePID
from control.pid_controller import EgoPIDController


@pytest.mark.parametrize('backend',['route_manager','route_context'])
@pytest.mark.parametrize('yaw',[0.,90.,180.])
@pytest.mark.parametrize('mode',['completed_lane','turn_left','turn_right'])
def test_completed_lane_is_not_pulled_to_the_original_route(yaw,backend,mode):
    radians=math.radians(yaw)
    def waypoint(distance):
        return NS(transform=NS(location=NS(x=distance*math.cos(radians),
            y=distance*math.sin(radians),z=2.),rotation=NS(yaw=yaw)),
            next=lambda d:[waypoint(distance+d)])
    current=waypoint(0.)
    controller=GenericRoutePID.__new__(GenericRoutePID)
    controller._corridor_road_map=NS(get_waypoint=lambda *a,**kw:current)
    controller.ego=NS(get_location=lambda:current.transform.location)
    controller._current_speed_kmh=lambda:8.
    controller._junction_or_road_transition_ahead=lambda:False
    controller.route_context=None
    controller.route_manager=NS(target_point=lambda d:dict(
        x=d*math.cos(radians)-3.5*math.sin(radians),
        y=d*math.sin(radians)+3.5*math.cos(radians),yaw=yaw))
    if backend=='route_context':
        controller.route_context=NS()
        controller.route_plan=[(current,None)]
        controller.route_manager=None
    controller._pid=EgoPIDController.__new__(EgoPIDController)
    if mode == 'completed_lane':
        controller.handoff_completed_lane()
    else:
        controller._default_speed_kmh=30.
        controller._requested_cruise_speed_kmh=30.
        controller.set_high_level_decision(dict(action=mode,target_speed_kmh=8.))
    target=controller._route_target()
    assert target['reference']['x']==0. and target['reference']['y']==0.
    assert target['z']==2.
    assert abs(-math.sin(radians)*target['x']+math.cos(radians)*target['y'])<1e-6


@pytest.mark.parametrize('action',['turn_left','turn_right'])
def test_turn_lane_continuity_survives_a_temporary_stop_but_not_a_new_merge(action):
    controller=GenericRoutePID(NS(get_map=lambda:NS()), NS(get_speed_limit=lambda:60.),
        target_speed_kmh=30., fixed_delta_seconds=.05)
    controller.set_high_level_decision(dict(action=action,target_speed_kmh=10.))
    assert controller._completed_lane_navigation is True
    controller.set_high_level_decision(dict(action='stop',target_speed_kmh=0.))
    assert controller._completed_lane_navigation is True
    controller.set_high_level_decision(dict(action='lane_change_right',target_speed_kmh=8.))
    assert controller._completed_lane_navigation is False


@pytest.mark.parametrize('action', ['turn_left', 'turn_right'])
@pytest.mark.parametrize('temporary_stop', [False, True])
def test_same_turn_step_retains_selected_branch_across_decision_updates(action, temporary_stop):
    controller = GenericRoutePID(NS(get_map=lambda:NS()), NS(get_speed_limit=lambda:60.),
        target_speed_kmh=30., fixed_delta_seconds=.05)
    decision = dict(action=action, target_speed_kmh=10., source_step_id='turn_1', command_id='command_1')
    controller.set_high_level_decision(decision)
    selected_branch = [object()]
    controller._navigation_branch_samples = selected_branch
    if temporary_stop:
        controller.set_high_level_decision(dict(action='stop', target_speed_kmh=0.,
            source_step_id='turn_1', command_id='command_1'))
    controller.set_high_level_decision(decision)
    assert controller._navigation_branch_samples is selected_branch
    controller.set_high_level_decision({**decision, 'source_step_id': 'turn_2'})
    assert controller._navigation_branch_samples is None
    controller._navigation_branch_samples = selected_branch
    opposite = 'turn_left' if action == 'turn_right' else 'turn_right'
    controller.set_high_level_decision({**decision, 'source_step_id': 'turn_2', 'action': opposite})
    assert controller._navigation_branch_samples is None


def test_new_turn_is_not_masked_by_the_previous_merge_settling():
    controller=EgoPIDController.__new__(EgoPIDController)
    controller._last_lane_change_intent=dict(action='lane_change_left',target_speed_kmh=8.)
    controller._lane_change_settle_frames=20
    intent=dict(action='turn_right',target_speed_kmh=10.)
    assert controller._lateral_intent_for_control(intent) is intent
    assert controller._lane_change_settle_frames==0


@pytest.mark.parametrize('legal',[True,False])
@pytest.mark.parametrize('current_reaches',[True,False])
def test_recovery_uses_legal_route_topology_not_original_lane(legal,current_reaches):
    def waypoint(road,lane):
        return NS(road_id=road,lane_id=lane,is_junction=False,
            lane_type=__import__('carla').LaneType.Driving,
            transform=NS(location=NS(x=0.,y=0.,z=0.),rotation=NS(yaw=0.)))
    current,adjacent,planned,wrong=[waypoint(*pair) for pair in [(1,-1),(1,-2),(2,-1),(3,-1)]]
    adjacent.transform.location.y=3.5
    current.next=lambda d:[planned if current_reaches else wrong]
    adjacent.next=lambda d:[planned]
    current.get_left_lane=lambda:None
    current.get_right_lane=lambda:adjacent
    controller=GenericRoutePID.__new__(GenericRoutePID)
    controller._completed_lane_navigation=True
    controller._corridor_road_map=NS(get_waypoint=lambda *a,**kw:current)
    controller.ego=NS(get_location=lambda:current.transform.location)
    controller.route_manager=None
    controller.route_context=NS(distances_m=[0.,30.,45.,60.])
    controller.route_plan=[(current,None)]+[(planned,None)]*3
    controller.progress_m=lambda:0.
    controller._pid=EgoPIDController.__new__(EgoPIDController)
    controller.autonomous_lane_change_legal=lambda direction:legal and direction=='right'
    assert controller.route_lane_recovery_direction()==('right' if legal and not current_reaches else None)


@pytest.mark.parametrize('legal',[False,True])
def test_recovery_aligns_to_planned_lane_before_same_road_junction(legal):
    def wp(lane,y):
        return NS(road_id=1,lane_id=lane,is_junction=False,
            lane_type=__import__('carla').LaneType.Driving,
            transform=NS(location=NS(x=0.,y=y,z=0.),rotation=NS(yaw=0.)))
    current,adjacent=wp(-2,0.),wp(-1,-3.5)
    current.next=lambda d:[current]
    adjacent.next=lambda d:[adjacent]
    current.get_left_lane=lambda:adjacent
    current.get_right_lane=lambda:None
    controller=GenericRoutePID.__new__(GenericRoutePID)
    controller._completed_lane_navigation=True
    controller._corridor_road_map=NS(get_waypoint=lambda *a,**kw:current)
    controller.ego=NS(get_location=lambda:current.transform.location)
    controller.route_manager=None
    controller.route_context=NS(distances_m=[0.,15.,30.])
    controller.route_plan=[(current,None),(adjacent,None),(adjacent,None)]
    controller.progress_m=lambda:0.
    controller._pid=EgoPIDController.__new__(EgoPIDController)
    controller.autonomous_lane_change_legal=lambda d:legal and d=='left'
    assert controller.route_lane_recovery_direction()==('left' if legal else None)


@pytest.mark.parametrize('issued',[False,True])
def test_expired_navigation_window_only_cancels_an_unissued_merge(issued):
    from dataclasses import dataclass
    from universal_vla_controller import UniversalVLAController
    @dataclass
    class Parsed:
        parsed_intent:str='KEEP_LANE'
        source_text:str='continue'
        requested_lane_direction:str|None=None
    controller=UniversalVLAController.__new__(UniversalVLAController)
    controller.driving_plan=None
    controller._route_lane_recovery='left'
    controller._route_lane_recovery_key=('cruise','continue')
    controller._route_lane_recovery_command_id='cruise:nav:1'
    controller._lane_change_issued=issued
    controller._new_navigation_merge_window=lambda d:False
    controller.route_controller=NS(route_lane_recovery_direction=lambda:None)
    controller._lane_command_observation=lambda command,parsed:(parsed,['corridor'])
    command=dict(id='cruise');parsed=Parsed()
    result,output,corridor=controller._navigation_lane_recovery_observation(command,parsed,None)
    assert (output.requested_lane_direction=='left') is issued
    if not issued:
        assert result is command and output is parsed and corridor is None
        assert controller._route_lane_recovery is None


@pytest.mark.parametrize('intent',['STOP','EMERGENCY_BRAKE','WAIT','TURN_RIGHT','CHANGE_LANE_LEFT'])
def test_new_explicit_action_cancels_navigation_recovery(intent):
    from dataclasses import dataclass
    from universal_vla_controller import UniversalVLAController
    @dataclass
    class Parsed:
        parsed_intent:str
        source_text:str='new instruction'
        requested_lane_direction:str|None=None
    controller=UniversalVLAController.__new__(UniversalVLAController)
    controller.driving_plan=None
    controller._route_lane_recovery='right'
    controller._route_lane_recovery_key=('old','old instruction')
    controller.route_controller=NS(route_lane_recovery_direction=lambda:pytest.fail('explicit action must win'))
    command=dict(id='new');parsed=Parsed(intent)
    out=controller._navigation_lane_recovery_observation(command,parsed,[])
    assert out==(command,parsed,[])
    assert controller._route_lane_recovery is None


def test_new_cruise_command_does_not_inherit_old_navigation_recovery():
    from dataclasses import dataclass
    from universal_vla_controller import UniversalVLAController
    @dataclass
    class Parsed:
        parsed_intent:str='KEEP_LANE'
        source_text:str='new cruise'
        requested_lane_direction:str|None=None
    controller=UniversalVLAController.__new__(UniversalVLAController)
    controller.driving_plan=None
    controller._route_lane_recovery='right'
    controller._route_lane_recovery_key=('old','old cruise')
    controller.route_controller=NS(route_lane_recovery_direction=lambda:None)
    command=dict(id='new');parsed=Parsed()
    assert controller._navigation_lane_recovery_observation(command,parsed,None)==(command,parsed,None)
    assert controller._route_lane_recovery is None


def test_intermediate_shared_road_does_not_hide_a_later_turn_only_branch():
    def waypoint(road,lane,y=0.):
        return NS(road_id=road,lane_id=lane,is_junction=False,
            lane_type=__import__('carla').LaneType.Driving,
            transform=NS(location=NS(x=0.,y=y,z=0.),rotation=NS(yaw=0.)))
    current,adjacent,shared,planned,wrong=[waypoint(*pair) for pair in
        [(1,-1),(1,-2,3.5),(2,-1),(4,-1),(3,-1)]]
    current.next=lambda d:[shared if d<90 else wrong]
    adjacent.next=lambda d:[shared if d<90 else planned]
    current.get_left_lane=lambda:None
    current.get_right_lane=lambda:adjacent
    controller=GenericRoutePID.__new__(GenericRoutePID)
    controller._completed_lane_navigation=True
    controller._corridor_road_map=NS(get_waypoint=lambda *a,**kw:current)
    controller.ego=NS(get_location=lambda:current.transform.location)
    controller.route_manager=None
    controller.route_context=NS(distances_m=[0.,30.,45.,60.,90.,120.])
    controller.route_plan=[(w,None) for w in [current,shared,shared,shared,planned,planned]]
    controller.progress_m=lambda:0.
    controller._pid=EgoPIDController.__new__(EgoPIDController)
    controller.autonomous_lane_change_legal=lambda d:d=='right'
    assert controller.route_lane_recovery_direction()=='right'


def test_fork_choice_uses_future_route_not_nearest_local_geometry():
    def wp(road,x,y):
        return NS(road_id=road,transform=NS(location=NS(x=x,y=y,z=0.),rotation=NS(yaw=0.)))
    current,wrong,correct,future=wp(1,0.,0.),wp(2,8.,0.),wp(3,8.,3.5),wp(4,60.,3.5)
    current.next=lambda d:[wrong,correct]
    wrong.next=lambda d:[wp(5,d,0.)]
    correct.next=lambda d:[future]
    correct.previous=lambda d:[]
    controller=GenericRoutePID.__new__(GenericRoutePID)
    controller._corridor_road_map=NS(get_waypoint=lambda *a,**kw:current)
    controller.ego=NS(get_location=lambda:current.transform.location)
    controller.route_manager=None
    controller.route_context=NS(distances_m=[0.,8.,38.,68.,98.])
    controller.route_plan=[(w,None) for w in [current,wrong,future,future,future]]
    controller.progress_m=lambda:0.
    controller._pid=EgoPIDController.__new__(EgoPIDController)
    assert controller._current_lane_navigation_point(8.) is correct


def test_selected_branch_survives_nearest_map_projection_to_adjacent_fork():
    def wp(road,x,y):
        point=NS(road_id=road,lane_width=3.5,
            transform=NS(location=NS(x=x,y=y,z=0.),rotation=NS(yaw=0.)))
        point.next=lambda d:[wp(road,x+d,y)]
        point.previous=lambda d:[wp(road,x-d,y)]
        return point
    wrong,correct=wp(2,10.,0.),wp(3,10.,3.5)
    controller=GenericRoutePID.__new__(GenericRoutePID)
    controller._corridor_road_map=NS(get_waypoint=lambda *a,**kw:wrong)
    location=NS(x=10.,y=3.5,z=0.)
    controller.ego=NS(get_location=lambda:location,
        get_transform=lambda:NS(rotation=NS(yaw=0.)))
    controller._pid=EgoPIDController.__new__(EgoPIDController)
    controller._navigation_branch_samples=[correct]
    assert controller._current_lane_navigation_point(0.) is correct
    location.y=8.
    assert controller._current_lane_navigation_point(0.) is wrong
    assert controller._navigation_branch_samples is None


@pytest.mark.parametrize('action',['lane_change_left','lane_change_right','turn_left','turn_right'])
def test_new_explicit_lateral_action_clears_branch_latch(action):
    ego=NS(get_speed_limit=lambda:60.)
    world=NS(get_map=lambda:NS())
    controller=GenericRoutePID(world,ego,target_speed_kmh=30.,fixed_delta_seconds=.05)
    controller._navigation_branch_samples=[NS()]
    controller.set_high_level_decision(dict(action=action,target_speed_kmh=8.))
    assert controller._navigation_branch_samples is None


def test_non_lane_command_does_not_inherit_old_lane_completion_feedback():
    from universal_vla_controller import UniversalVLAController
    controller=UniversalVLAController.__new__(UniversalVLAController)
    controller._lane_goal=(1,0,-1);controller._lane_goal_complete=True
    controller._lane_feedback_key=('old','merge left')
    controller._lane_command_observation(dict(id='new'),NS(requested_lane_direction=None))
    feedback=controller._lane_task_feedback()
    assert feedback['active'] is False and feedback['completed'] is False
    assert feedback['command_id'] is None and feedback['target_lane'] is None


def test_completed_fact_keeps_its_command_and_not_another_maneuver_target():
    from universal_vla_controller import UniversalVLAController
    controller=UniversalVLAController.__new__(UniversalVLAController)
    controller._lane_goal=(2,0,-2);controller._lane_goal_complete=True
    controller._lane_feedback_key=('left','merge left')
    controller._completed_lane_commands={('left','merge left'):dict(command_id='left',
        target_lane=(1,0,-1),completed=True)}
    feedback=controller._lane_task_feedback()
    assert feedback['command_id']=='left' and feedback['target_lane']==(1,0,-1)


@pytest.mark.parametrize('direction',['left','right'])
@pytest.mark.parametrize('intent',['KEEP_LANE','RESUME','PROCEED'])
def test_cruise_prepass_preserves_inflight_navigation_target(direction,intent):
    from universal_vla_controller import UniversalVLAController
    controller=UniversalVLAController.__new__(UniversalVLAController)
    controller.driving_plan=None
    controller._route_lane_recovery=direction
    controller._route_lane_recovery_key=('cruise','continue safely')
    key=('cruise:navigation_lane_recovery','continue safely')
    controller._lane_command_key=key
    controller._lane_feedback_key=key
    controller._lane_goal=(1,0,-1)
    controller._lane_change_issued=True
    controller._lane_goal_stable=3
    parsed=NS(requested_lane_direction=None,parsed_intent=intent,source_text='continue safely')
    assert controller._lane_command_observation(dict(id='cruise'),parsed)==(parsed,None)
    assert controller._lane_command_key==key
    assert controller._lane_feedback_key==key
    assert controller._lane_goal==(1,0,-1)
    assert controller._lane_change_issued and controller._lane_goal_stable==3


def test_new_cruise_prepass_clears_previous_navigation_target_binding():
    from universal_vla_controller import UniversalVLAController
    controller=UniversalVLAController.__new__(UniversalVLAController)
    controller.driving_plan=None
    controller._route_lane_recovery='left'
    controller._route_lane_recovery_key=('old','old cruise')
    controller._lane_command_key=('old:navigation_lane_recovery','old cruise')
    controller._lane_feedback_key=controller._lane_command_key
    parsed=NS(requested_lane_direction=None,parsed_intent='KEEP_LANE',source_text='new cruise')
    controller._lane_command_observation(dict(id='new'),parsed)
    assert controller._lane_command_key is None and controller._lane_feedback_key is None


def test_each_navigation_merge_has_its_own_completion_identity():
    from dataclasses import dataclass,replace
    from universal_vla_controller import UniversalVLAController
    @dataclass
    class Parsed:
        parsed_intent:str='KEEP_LANE'
        source_text:str='continue safely'
        requested_lane_direction:str|None=None
    controller=UniversalVLAController.__new__(UniversalVLAController)
    controller.driving_plan=None
    controller.route_controller=NS(route_lane_recovery_direction=lambda:'left')
    controller._new_navigation_merge_window=lambda direction:True
    commands=[]
    def observe(command,parsed):
        commands.append(command['id'])
        if len(commands)==3:
            return replace(parsed,parsed_intent='KEEP_LANE',requested_lane_direction=None),None
        return parsed,[]
    controller._lane_command_observation=observe
    for _ in range(4):
        controller._navigation_lane_recovery_observation(dict(id='cruise'),Parsed(),None)
    assert commands[0]==commands[1]==commands[2]
    assert commands[3]!=commands[0]
    assert controller._route_lane_recovery_generation==2


@pytest.mark.parametrize('action,condition,status,allowed', [
    ('PROCEED','FINISH','ACTIVE',True),
    ('PROCEED','STRAIGHT_THROUGH_JUNCTION','ACTIVE',False),
    ('PROCEED',None,'ACTIVE',False),
    ('KEEP_LANE',None,'ACTIVE',False),
    ('STOP',None,'ACTIVE',False),
    ('PROCEED','FINISH','BLOCKED',False),
    ('PROCEED','FINISH','COMPLETED',False),
])
def test_active_route_destination_can_recover_lane_without_bypassing_other_plan_steps(action,condition,status,allowed):
    from dataclasses import dataclass
    from universal_vla_controller import UniversalVLAController
    @dataclass
    class Parsed:
        parsed_intent:str='PROCEED'
        source_text:str='follow the planned route to the destination'
        requested_lane_direction:str|None=None
    controller=UniversalVLAController.__new__(UniversalVLAController)
    step=dict(step_id='destination',action=action,parameters=dict(condition=condition))
    controller.driving_plan=NS(state=dict(plan_status=status,active_step_id='destination'),
        effective_document=None,document=dict(intent=dict(steps=[step])))
    controller.route_controller=NS(route_lane_recovery_direction=lambda:'left')
    controller._new_navigation_merge_window=lambda direction:True
    controller._lane_command_observation=lambda command,parsed:(parsed, ['measured_corridor'])
    command=dict(id='plan:destination')
    result,parsed,corridor=controller._navigation_lane_recovery_observation(command,Parsed(),None)
    assert (parsed.requested_lane_direction=='left') is allowed
    assert ('navigation_lane_recovery' in result) is allowed
    assert (corridor==['measured_corridor']) is allowed


def test_active_destination_prepass_preserves_inflight_navigation_merge_identity():
    from universal_vla_controller import UniversalVLAController
    controller=UniversalVLAController.__new__(UniversalVLAController)
    step=dict(step_id='destination',action='PROCEED',parameters=dict(condition='FINISH'))
    controller.driving_plan=NS(state=dict(plan_status='ACTIVE',active_step_id='destination'),
        effective_document=None,document=dict(intent=dict(steps=[step])))
    controller._route_lane_recovery='left'
    controller._route_lane_recovery_key=('plan:destination','follow route')
    locked=('plan:destination:navigation_lane_recovery:1','follow route')
    controller._lane_command_key=locked
    controller._lane_feedback_key=locked
    parsed=NS(parsed_intent='PROCEED',source_text='follow route',requested_lane_direction=None)
    assert controller._lane_command_observation(dict(id='plan:destination'),parsed)==(parsed,None)
    assert controller._lane_command_key==locked
    assert controller._lane_feedback_key==locked


@pytest.mark.parametrize('allowed',[False,True])
def test_navigation_merge_waits_for_legal_departure_window(allowed):
    from dataclasses import dataclass
    from universal_vla_controller import UniversalVLAController
    @dataclass
    class Parsed:
        parsed_intent:str='KEEP_LANE'
        source_text:str='continue safely'
        requested_lane_direction:str|None=None
    controller=UniversalVLAController.__new__(UniversalVLAController)
    controller.driving_plan=None
    controller.route_controller=NS(route_lane_recovery_direction=lambda:'left')
    controller._new_navigation_merge_window=lambda direction:allowed
    observed=[]
    def observe(command,parsed):
        observed.append(command)
        return parsed,['corridor']
    controller._lane_command_observation=observe
    command=dict(id='cruise')
    original=Parsed()
    result,parsed,corridor=controller._navigation_lane_recovery_observation(command,original,None)
    assert bool(observed) is allowed
    assert getattr(controller,'_route_lane_recovery_generation',0)==int(allowed)
    if not allowed:
        assert result is command and parsed is original and corridor is None


@pytest.mark.parametrize('forward,rear',[(False,True),(True,False),(True,True)])
def test_new_navigation_window_requires_forward_and_rear_proof(monkeypatch,forward,rear):
    import control.lane_departure_window as departure
    import lightweight_vla_adapter.src.navigation_merge as merge
    from universal_vla_controller import UniversalVLAController
    target=object()
    current=NS(get_left_lane=lambda:target)
    controller=UniversalVLAController.__new__(UniversalVLAController)
    controller.ego=NS(get_location=lambda:'location')
    controller._risk_road_map=NS(get_waypoint=lambda location:current)
    monkeypatch.setattr(departure,'lane_departure_window',lambda a,b:dict(allowed=forward))
    monkeypatch.setattr(merge,'rear_departure_window',lambda a,b:dict(allowed=rear))
    assert controller._new_navigation_merge_window('left') is (forward and rear)
