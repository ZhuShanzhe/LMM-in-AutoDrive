from benchmark.task_oracle import TaskOracle


def oracle():
    return TaskOracle(dict(schema_version='task_oracle/1.0',task_id='return',activate_m=0,
        timeout_s=10,max_frame_gap_s=1,steps=[dict(kind='progress',distance_m=1),
        dict(kind='return_original_lane',hold_s=1,max_lateral_error_m=.35,max_heading_error_deg=5)]))


def row(t,lane,road='r'):
    return dict(schema_version='task_truth/1.0',source='simulator_truth',frame=t,sim_time_s=t,
        scenario_valid=True,safety=dict(collisions=0,violations=0),
        ego=dict(route_s_m=t,speed_kmh=20,lane_key=lane,road_key=road,in_junction=False,
                 lateral_error_m=0,heading_error_deg=0))


def test_return_uses_original_identity_not_arbitrary_right_lane():
    evaluator=oracle()
    for t,lane in enumerate(['origin','left','left','other_right','origin','origin']):
        result=evaluator.update(row(t,lane))
        assert result['status']==('SUCCESS' if t==5 else 'RUNNING')


def test_return_before_its_step_is_not_accepted():
    evaluator=oracle()
    for t,lane in enumerate(['origin','left','origin']):
        result=evaluator.update(row(t,lane))
    assert result['reason']=='already_returned_before_return_step'


def test_road_transition_requires_explicit_continuity_not_lane_id_guess():
    evaluator=oracle()
    evaluator.update(row(0,'origin'))
    assert evaluator.update(row(1,'origin','other_road'))['status']=='SCENE_INVALID'
