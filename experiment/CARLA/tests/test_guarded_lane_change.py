import json

from benchmark.catalog import CONFIG_ROOT, load_catalog
from benchmark.task_oracle import TaskOracle, load_profile


def oracle():
    spec=json.loads((CONFIG_ROOT/'benchmark/scene3_blocked_lane_change_left.json').read_text())
    return TaskOracle(spec)


def observation(t,clear=False,lane='entry',speed=0):
    def actor(identity,key,front,rear):
        return dict(actor_id=identity,alive=True,lane_key=key,route_corridor_id='route',
                    front_route_s_m=front,rear_route_s_m=rear,speed_kmh=0)
    return dict(schema_version='task_truth/1.0',source='simulator_truth',
        task_id='scene3_blocked_lane_change_left',source_sha256=oracle().spec['source_sha256'],
        frame=round(t*20),sim_time_s=t,scenario_valid=True,safety=dict(collisions=0,violations=0),
        ego=dict(route_s_m=4700,speed_kmh=speed,front_route_s_m=4702,rear_route_s_m=4698,
                 lane_key=lane,route_corridor_id='route',in_junction=False,lateral_error_m=0,heading_error_deg=0),
        actors={
            'scene3_gap_front_vehicle':actor(1,'target',4750,4746),
            'scene3_gap_rear_vehicle':actor(2,'target',4660 if clear else 4695,4656 if clear else 4691),
            'scene3_maintenance_vehicle':actor(3,'entry',4710,4706)},
        fixture=dict(steps={'0':dict(legal=True,direction='LEFT',entry_lane_key='entry',target_lane_key='target')}))


def test_unsafe_gap_requires_wait_then_safe_merge():
    evaluator=oracle()
    for i in range(36):
        t=i*.05
        result=evaluator.update(observation(t,clear=t>=.2,lane='target' if t>=.75 else 'entry'))
    assert result['status']=='SUCCESS'
    assert any(e['event']=='GAP_ACCEPTED' for e in result['evidence'])


def test_unsafe_merge_is_failure():
    evaluator=oracle()
    evaluator.update(observation(0))
    assert evaluator.update(observation(.05,lane='target'))['reason']=='entered_lane_before_verified_gap'


def test_moving_too_close_to_blockage_fails():
    assert oracle().update(observation(0,speed=10))['reason']=='failed_to_stop_before_blockage'


def test_waiting_alone_is_not_task_success():
    evaluator=oracle()
    for i in range(30):
        result=evaluator.update(observation(i*.05))
    assert result['status']=='RUNNING'


def test_profile_is_bound_to_scene3_source():
    catalog=load_catalog('scene_3')
    assert load_profile(catalog,catalog.select('scene3_blocked_lane_change_left')[0]) is not None
