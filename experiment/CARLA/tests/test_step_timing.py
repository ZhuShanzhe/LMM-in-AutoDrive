from benchmark.episode_diagnostic import step_timing_diagnostics


def test_timing_partitions_steps_without_guessing_stop_reason():
    rows=[dict(frame=i,sim_time_s=i*.05,scenario_valid=True,ego={'speed_kmh':v})
          for i,v in enumerate([0,0,10,10,10])]
    spec={'steps':[{'kind':'yield_pedestrian'},{'kind':'lane_change'}],'max_frame_gap_s':.15}
    result={'evidence':[{'event':'START','frame':0},
                        {'event':'STEP_SUCCESS','step':0,'frame':2}]}
    out=step_timing_diagnostics(rows,spec,result)
    assert out[0]['near_stopped_s']==.05
    assert out[0]['observed_duration_s']==.1
    assert out[1]['observed_duration_s']==.1
    assert not out[1]['completed']
    assert out[0]['stop_cause']=='not_recorded'


def test_timing_does_not_count_missing_intervals():
    rows=[dict(frame=i,sim_time_s=t,scenario_valid=True,ego={'speed_kmh':0})
          for i,t in enumerate([0,1])]
    spec={'steps':[{'kind':'speed'}],'max_frame_gap_s':.15}
    result={'evidence':[{'event':'START','frame':0}]}
    out=step_timing_diagnostics(rows,spec,result)
    assert out[0]['observed_duration_s']==1
    assert out[0]['covered_interval_s']==0
    assert out[0]['near_stopped_s']==0
