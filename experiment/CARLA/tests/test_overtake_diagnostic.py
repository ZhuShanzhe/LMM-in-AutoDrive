from benchmark.episode_diagnostic import overtake_diagnostics


SPEC={'steps':[{'kind':'lane_change'},
               {'kind':'overtake','target_role':'lead','rear_clearance_m':8}]}


def row(frame,ego_s,target_s,corridor='road'):
    return dict(frame=frame,sim_time_s=frame*.05,scenario_valid=True,
                ego=dict(route_s_m=ego_s,rear_route_s_m=ego_s-2,route_corridor_id='road'),
                actors={'lead':dict(route_s_m=target_s,front_route_s_m=target_s+2,
                                    route_corridor_id=corridor)})


def result(frame):
    return {'evidence':[dict(event='STEP_SUCCESS',step=0,frame=frame,sim_time_s=frame*.05)]}


def test_pass_before_lane_acquisition_is_visible():
    report=overtake_diagnostics([row(1,0,10),row(2,25,10),row(3,30,10)],SPEC,result(2))[0]
    assert report['first_target_ahead']['frame']==1
    assert report['first_clearance_reached']['frame']==2
    assert not report['target_ahead_after_step_activation']
    assert report['rear_clearance_at_first_step_sample_m']==16


def test_ordered_pass_and_unrelated_corridor():
    report=overtake_diagnostics([row(1,100,0,'other'),row(2,0,10),row(3,25,10)],SPEC,result(1))[0]
    assert report['first_clearance_reached']['frame']==3
    assert report['target_ahead_after_step_activation']


def test_unactivated_step_has_no_activation_claim():
    report=overtake_diagnostics([row(1,0,10)],SPEC,{'evidence':[]})[0]
    assert report['previous_step_completed'] is None
    assert report['rear_clearance_at_first_step_sample_m'] is None
    assert not report['target_ahead_after_step_activation']
