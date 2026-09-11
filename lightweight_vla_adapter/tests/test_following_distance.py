import pytest
from lightweight_vla_adapter.src.following_distance import audit_spacing


def rows(n=400,gap=lambda t:13.,speed=18.,lead=18.):
    return [dict(t_s=(i+1)*.05,gap_m=gap((i+1)*.05),speed_kmh=speed,lead_speed_kmh=lead) for i in range(n)]


def test_stable_spacing_passes_without_speed_fraction():
    assert audit_spacing(rows())['status']=='meets_provisional_spacing_checks'


def test_faster_catchup_not_automatically_failed():
    data=rows(600,gap=lambda t:max(13.,33.-2*t))
    for r in data:
        if r['t_s']<10:r['speed_kmh']=25.2
    result=audit_spacing(data)
    assert result['approach_faster_than_lead_frames']>0
    assert result['status']=='meets_provisional_spacing_checks'


def test_passing_through_band_does_not_count_as_settling():
    result=audit_spacing(rows(gap=lambda t:30.-8*t))
    assert result['first_confirmed_acquisition_s'] is None


def test_acquisition_then_falling_back_is_reported():
    result=audit_spacing(rows(gap=lambda t:13. if t<10 else 30.))
    assert result['status']=='acquired_then_lost_spacing'


def test_far_away_same_speed_cannot_pass():
    assert audit_spacing(rows(gap=lambda t:60.))['status']=='not_acquired_within_observation'


def test_late_acquisition_is_insufficient_not_pass():
    result=audit_spacing(rows(gap=lambda t:40. if t<15 else 13.))
    assert result['status']=='insufficient_post_acquisition_observation'


def test_collision_overrides_spacing():
    assert audit_spacing(rows(),collision_events=1)['status']=='observed_hazard'


def test_time_gaps_break_continuous_hold():
    data=rows(80);data=data[:30]+data[50:]
    assert audit_spacing(data)['first_confirmed_acquisition_s'] is None


def test_missing_lead_gap_rejected():
    data=rows();data[0]['gap_m']=None
    with pytest.raises(ValueError):audit_spacing(data)


def test_entering_band_at_end_is_insufficient_not_a_demonstrated_failure():
    data=rows(gap=lambda t:40. if t<18 else 13.)
    assert audit_spacing(data)['status']=='insufficient_acquisition_observation'


def test_existing_auditor_marks_speed_as_diagnostic(tmp_path,monkeypatch,capsys):
    import json
    import sys
    from lightweight_vla_adapter.scripts import audit_executed_following
    ep=dict(id='follow',mode='follow18',desired=30.,collision_events=0,progress_m=1.,frames=10)
    source=json.dumps(dict(status='completed',episodes=[ep],sequence_execution=dict(operation_interval_s=.5)))
    (tmp_path/'report.json').write_text(source)
    data=rows(10)
    for i,r in enumerate(data):
        r.update(t_s=15.+i*.05,frame=i,brake=0.,throttle=0.,decision_gap_injected=False)
    (tmp_path/'follow_truth.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in data))
    monkeypatch.setattr(sys,'argv',['audit','--run',str(tmp_path)])
    audit_executed_following.main();capsys.readouterr()
    result=json.loads((tmp_path/'following_audit.json').read_text())
    assert result['speed_fraction_used_for_acceptance'] is False
    assert result['episodes'][0]['speed_status']=='diagnostic_only'
    assert result['overall']['provisional_spacing_check_pass_episodes']==0
    assert (tmp_path/'report.json').read_text()==source
