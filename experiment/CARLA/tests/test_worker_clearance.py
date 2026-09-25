from types import SimpleNamespace as NS
from worker_clearance import worker_clear_of_lane
from worker_clearance import worker_trigger_ready
import pytest


def sample(y,yaw=0):
    box=NS(location=NS(x=0,y=0),rotation=NS(yaw=0),extent=NS(x=.3,y=.3))
    actor=NS(bounding_box=box,get_transform=lambda:NS(location=NS(x=0,y=y),rotation=NS(yaw=yaw)))
    lane=NS(lane_width=3.5,transform=NS(location=NS(x=0,y=0),rotation=NS(yaw=0)))
    return actor,lane


def test_destination_lane_center_is_not_clear():
    assert not worker_clear_of_lane(*sample(0))['clear']


def test_center_outside_but_body_still_overlapping_is_not_clear():
    assert not worker_clear_of_lane(*sample(1.9))['clear']


def test_body_and_margin_outside_is_clear():
    assert worker_clear_of_lane(*sample(2.5))['clear']


def test_planned_target_uses_destination_not_current_safe_position():
    worker,lane=sample(5)
    result=worker_clear_of_lane(worker,lane,target_location=NS(x=0,y=0))
    assert not result['clear']
    assert result['geometry']=='planned_heading_independent_radius'


def test_planned_heading_margin_is_conservative():
    worker,lane=sample(0)
    worker.bounding_box.extent.x=1
    assert not worker_clear_of_lane(worker,lane,target_location=NS(x=0,y=2.5))['clear']
    assert worker_clear_of_lane(worker,lane,target_location=NS(x=0,y=3.5))['clear']


def test_trigger_uses_configured_distance_and_rejects_missed_location():
    event={'safety':{'minimum_trigger_distance_m':50}}
    assert not worker_trigger_ready(100,49,event)
    assert worker_trigger_ready(100,50,event)
    with pytest.raises(RuntimeError,match='missed'):
        worker_trigger_ready(100,100,event)
