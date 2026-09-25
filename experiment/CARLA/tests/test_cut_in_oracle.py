from benchmark.catalog import load_catalog
from benchmark.task_oracle import TaskOracle,load_profile


def oracle():
    catalog=load_catalog('scene_3')
    return TaskOracle(load_profile(catalog,catalog.select('scene3_cut_in_decelerate')[0]))


def row(t,merged=False,speed=40):
    return dict(schema_version='task_truth/1.0',source='simulator_truth',
        task_id='scene3_cut_in_decelerate',source_sha256=oracle().spec['source_sha256'],
        frame=round(t*20),sim_time_s=t,scenario_valid=True,safety=dict(collisions=0,violations=0),
        ego=dict(route_s_m=1250,speed_kmh=speed,lane_key='ego',road_key='r',in_junction=False,
                 front_route_s_m=1252,route_corridor_id='main',lateral_error_m=0,heading_error_deg=0),
        actors={'scene3_cut_in_vehicle':dict(actor_id=2,alive=True,lane_key='ego' if merged else 'side',
            rear_route_s_m=1280,speed_kmh=35,route_corridor_id='main')})


def test_adjacent_vehicle_alone_cannot_complete_cut_in_task():
    evaluator=oracle()
    for i in range(30):
        result=evaluator.update(row(i*.05,speed=30))
    assert result['status']=='RUNNING'


def test_merge_then_speed_response_and_hold():
    evaluator=oracle()
    for i in range(30):
        result=evaluator.update(row(i*.05,merged=i>=2,speed=30 if i>=4 else 40))
    assert result['status']=='SUCCESS'
    response=next(e for e in result['evidence'] if e['event']=='MEASURED_YIELD_RESPONSE')
    assert response['response_after_lane_entry_s']==.1


def test_already_merged_is_missing_event_evidence():
    assert oracle().update(row(0,merged=True))['status']=='SCENE_INVALID'
