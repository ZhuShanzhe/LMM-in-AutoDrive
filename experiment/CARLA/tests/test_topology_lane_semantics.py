import copy

from benchmark.task_oracle import TaskOracle


def row(frame, lane, *, junction=False, fixture=None):
    return dict(schema_version='task_truth/1.0',source='simulator_truth',frame=frame,
        sim_time_s=frame*.5,scenario_valid=True,safety=dict(collisions=0,violations=0),
        fixture=fixture or {},ego=dict(route_s_m=float(frame),speed_kmh=20,
            lane_key=lane,road_key=lane.split(':')[0],in_junction=junction,
            lateral_error_m=0,heading_error_deg=0))


def corridor():
    return dict(lane_corridor=[dict(start_m=0.,end_m=2.,lane_keys=['old:0:1']),
        dict(start_m=2.,end_m=3.,lane_keys=['junction:0:1']),
        dict(start_m=3.,end_m=10.,lane_keys=['new:0:1'])])


def evaluator(kind):
    return TaskOracle(dict(schema_version='task_oracle/1.0',task_id='topology',activate_m=0,
        timeout_s=20,max_frame_gap_s=1,steps=[dict(kind='progress',distance_m=1),
        dict(kind=kind,hold_s=1,max_lateral_error_m=.35,max_heading_error_deg=5)]))


def test_lane_hold_accepts_only_verified_physical_continuation_and_restarts_after_junction():
    oracle=evaluator('lane_hold');fixture=dict(steps={'1':corridor()})
    for frame,lane in enumerate(['old:0:1','old:0:1','old:0:1','junction:0:1',
                               'new:0:1','new:0:1','new:0:1']):
        result=oracle.update(row(frame,lane,junction=frame==3,fixture=fixture))
        assert result['status']==('SUCCESS' if frame==6 else 'RUNNING')


def test_parallel_lane_is_not_a_verified_hold_continuation():
    oracle=evaluator('lane_hold');fixture=dict(steps={'1':corridor()})
    for frame,lane in enumerate(['old:0:1','old:0:1','old:0:1',
                               'new:0:2','new:0:2','new:0:2']):
        result=oracle.update(row(frame,lane,fixture=fixture))
    assert result['status']=='RUNNING'


def test_return_original_lane_follows_frozen_physical_corridor_across_road_ids():
    oracle=evaluator('return_original_lane')
    fixture=dict(original_lane_corridor=corridor())
    for frame,lane in enumerate(['old:0:1','old:0:2','old:0:2',
                               'new:0:2','new:0:1','new:0:1','new:0:1']):
        result=oracle.update(row(frame,lane,fixture=fixture))
    assert result['status']=='SUCCESS'


def test_changing_original_lane_evidence_is_invalid_not_success():
    oracle=evaluator('return_original_lane')
    fixture=dict(original_lane_corridor=corridor())
    oracle.update(row(0,'old:0:1',fixture=fixture))
    changed=copy.deepcopy(fixture)
    changed['original_lane_corridor']['lane_corridor'][1]['lane_keys']=['other']
    assert oracle.update(row(1,'old:0:2',fixture=changed))['status']=='SCENE_INVALID'

