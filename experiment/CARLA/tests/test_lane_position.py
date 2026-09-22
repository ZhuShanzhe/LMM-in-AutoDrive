from types import SimpleNamespace as NS

import pytest

from benchmark.catalog import ConfigError
from benchmark.lane_position import lane_position, matches
from benchmark.task_oracle import TaskOracle


def lane(identity, yaw=0, kind='Driving'):
    wp=NS(road_id=5,section_id=0,lane_id=identity,is_junction=False,lane_type=kind,
          transform=NS(rotation=NS(yaw=yaw)),left=None,right=None)
    wp.get_left_lane=lambda:wp.left
    wp.get_right_lane=lambda:wp.right
    return wp


def test_lane_number_not_lane_id_and_opposite_direction_excluded():
    left,middle,right,opposite=[lane(-7),lane(-10),lane(-15),lane(1,180)]
    middle.left,middle.right=left,right
    left.right,left.left=middle,opposite
    right.left=middle
    assert lane_position(middle)==dict(valid=True,reason=None,index_from_left=2,index_from_right=2,lane_count=3)
    assert lane_position(right)['index_from_right']==1


def test_shoulder_is_not_counted():
    wp=lane(10)
    wp.right=lane(11,kind='Shoulder')
    assert lane_position(wp)['lane_count']==1


def test_junction_and_cycles_are_not_numbered():
    wp=lane(1)
    wp.is_junction=True
    assert not lane_position(wp)['valid']
    wp.is_junction=False
    wp.left=wp
    assert not lane_position(wp)['valid']


def test_nonexistent_second_lane_is_invalid_not_driving_failure():
    with pytest.raises(ConfigError):
        matches(lane_position(lane(1)),'LEFT',2)


def test_ordinal_criterion_requires_centered_hold():
    step=dict(kind='lane_position',lane_side='LEFT',lane_ordinal=2,hold_s=1,
              max_lateral_error_m=.35,max_heading_error_deg=5)
    oracle=TaskOracle(dict(schema_version='task_oracle/1.0',task_id='test',activate_m=0,
                           timeout_s=10,max_frame_gap_s=1,steps=[step]))
    for t,lateral in [(0,1),(1,0),(2,0)]:
        row=dict(schema_version='task_truth/1.0',source='simulator_truth',frame=t,
                 sim_time_s=t,scenario_valid=True,safety=dict(collisions=0,violations=0),
                 ego=dict(speed_kmh=20,route_s_m=t,lateral_error_m=lateral,heading_error_deg=0,
                          lane_position=dict(valid=True,index_from_left=2,index_from_right=2,lane_count=3)))
        result=oracle.update(row)
        assert result['status']==('SUCCESS' if t==2 else 'RUNNING')
