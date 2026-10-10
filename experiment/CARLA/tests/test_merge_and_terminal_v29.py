from types import SimpleNamespace as NS
import pytest
from benchmark.catalog import ConfigError
from benchmark.destination_fixture import SOURCE, terminal_lane_centers
from benchmark.task_oracle import TaskOracle


def profile(step):
    return dict(schema_version='task_oracle/1.0',task_id='case',activate_m=0,
                timeout_s=100,max_frame_gap_s=.15,steps=[step])


def observation(t,**changes):
    ego=dict(route_s_m=0.,speed_kmh=7.43,lane_key='r:0:1',road_key='r',
        in_junction=False,lateral_error_m=0.,heading_error_deg=0.,yaw_deg=0.,
        route_corridor_id='main',half_length_m=2.,front_route_s_m=2.,
        position_m=dict(x=0.,y=0.,z=0.))
    ego.update(changes)
    return dict(schema_version='task_truth/1.0',source='simulator_truth',frame=round(t*20),
        sim_time_s=t,scenario_valid=True,ego=ego,safety=dict(collisions=0,violations=0),
        actors={'lead':dict(actor_id=1,alive=True,speed_kmh=1.76,route_s_m=54.,
            rear_route_s_m=52.,lane_key='r:0:2',road_key='r',in_junction=False,
            route_corridor_id='main',half_length_m=2.)},
        fixture=dict(steps={'0':dict(lane_corridor=[dict(start_m=0.,end_m=100.,lane_keys=['r:0:1'])])}))


def cut_in_step():
    return dict(kind='yield_cut_in',target_role='lead',min_speed_drop_kmh=5.,
        min_gap_m=6.,min_ttc_s=2.5,hold_s=1.,max_lateral_error_m=.35,
        max_heading_error_deg=5.,stopped_kmh=.5,allow_already_safe_speed=True)


def run_merge(**changes):
    oracle=TaskOracle(profile(cut_in_step()))
    oracle.update(observation(0.))
    for i in range(1,23):
        row=observation(i*.05,speed_kmh=9.1)
        row['actors']['lead'].update(lane_key='r:0:1',speed_kmh=26.18)
        if 'target_speed' in changes:
            row['actors']['lead']['speed_kmh']=changes['target_speed']
        if 'gap' in changes:row['actors']['lead']['rear_route_s_m']=2.+changes['gap']
        if 'post_merge_speed' in changes and i>1:row['ego']['speed_kmh']=changes['post_merge_speed']
        oracle.update(row)
    return oracle.result()


def test_safe_merge_compares_speeds_when_actor_actually_enters_lane():
    result=run_merge()
    assert result['status']=='SUCCESS'
    evidence=next(e for e in result['evidence'] if e['event']=='ALREADY_SLOW_SAFE_YIELD')
    assert evidence['merge_ego_speed_kmh']==9.1
    assert evidence['merge_target_speed_kmh']==26.18


@pytest.mark.parametrize('changes',[dict(gap=5.9),dict(target_speed=8.),
                                  dict(post_merge_speed=10.)])
def test_safe_merge_does_not_waive_gap_closing_or_acceleration(changes):
    assert run_merge(**changes)['status']=='RUNNING'


def destination_step():
    return dict(kind='destination',keep_lane=False,allow_adjacent_destination_lanes=True,
        max_distance_m=3.,max_remaining_m=3.,max_lateral_error_m=.35,max_heading_error_deg=5.)


def terminal_observation(**changes):
    row=observation(0.,route_s_m=9.,position_m=dict(x=9.,y=3.5,z=0.),lane_key='r:0:2')
    row['ego'].update(changes)
    row['fixture']['steps']['0']=dict(route_end_s_m=10.,position_m=dict(x=10.,y=0.,z=0.),
        terminal_cross_section=dict(source=SOURCE,lane_points=[
            dict(lane_key='r:0:1',position_m=dict(x=10.,y=0.,z=0.)),
            dict(lane_key='r:0:2',position_m=dict(x=10.,y=3.5,z=0.))]))
    return row


def test_unspecified_final_lane_accepts_centered_legal_parallel_endpoint():
    assert TaskOracle(profile(destination_step())).update(terminal_observation())['status']=='SUCCESS'


@pytest.mark.parametrize('changes',[dict(lane_key='opposite'),dict(in_junction=True),
    dict(lateral_error_m=.351),dict(heading_error_deg=5.01),
    dict(position_m=dict(x=6.,y=3.5,z=0.),route_s_m=6.),
    dict(position_m=dict(x=9.,y=7.,z=0.))])
def test_terminal_still_requires_actual_position_and_lane_bounds(changes):
    assert TaskOracle(profile(destination_step())).update(terminal_observation(**changes))['status']=='RUNNING'


def test_unverified_terminal_fixture_is_rejected():
    row=terminal_observation();row['fixture']['steps']['0']['terminal_cross_section']['source']='model_guess'
    assert TaskOracle(profile(destination_step())).update(row)['status']=='SCENE_INVALID'


def test_fixed_lane_destination_cannot_enable_adjacent_endpoints():
    step=destination_step();step['keep_lane']=True
    with pytest.raises(ConfigError):TaskOracle(profile(step))


class Lane:
    def __init__(self,identity,y,yaw=0.,kind='Driving',road='r'):
        self.road_id=road;self.section_id=0;self.lane_id=identity;self.is_junction=False
        self.lane_type=kind;self.lane_width=3.5;self.left=self.right=None
        self.transform=NS(location=NS(x=10.,y=y,z=0.),rotation=NS(yaw=yaw))
    def get_left_lane(self):return self.left
    def get_right_lane(self):return self.right


@pytest.mark.parametrize('excluded',[dict(yaw=180.),dict(kind='Shoulder'),dict(road='other')])
def test_static_terminal_excludes_opposite_shoulder_or_other_road(excluded):
    base=Lane(1,0.);adjacent=Lane(2,3.5);other=Lane(3,7.,**excluded)
    base.left=adjacent;adjacent.left=other
    result=terminal_lane_centers(NS(get_waypoint=lambda location:base),
        dict(x=10.,y=0.,z=0.),lambda **kw:NS(**kw))
    assert [p['lane_key'] for p in result['lane_points']]==['r:0:1','r:0:2']
