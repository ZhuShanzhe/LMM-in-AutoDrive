import json
import pytest
from benchmark.report import summarize, main


def test_partial_success_not_full_instruction_success(tmp_path):
    path=tmp_path/'summary.json'
    path.write_text(json.dumps({'scene_id':'scene_1','tasks':{
        'a':{'status':'SUCCESS','instruction_status':'UNVERIFIED'},
        'b':{'status':'NOT_REACHED'},'c':{'status':'NOT_RUN'}},
        'episode_safety':{'status':'FAILURE'}}))
    result=summarize([path])['runs'][0]
    assert result['eligible_tasks']==2
    assert result['recorded_criteria_pass_rate']==.5
    assert result['verified_completion_lower_bound']==0
    assert result['episode_safety']['status']=='FAILURE'
    assert main([str(path),'--output',str(tmp_path/'report')])==0
    assert (tmp_path/'report/report.md').exists()


def test_missing_run_preserved_and_duplicates_rejected(tmp_path):
    path=tmp_path/'missing.json'
    result=summarize([path])
    assert result['requested_runs']==result['unreadable_runs']==1
    assert result['full_benchmark_acceptance'] is False
    with pytest.raises(ValueError,match='duplicate'):
        summarize([path,path])


def test_report_preserves_segment_traffic_evidence(tmp_path):
    from benchmark.report import markdown
    density=dict(status='RECORDED',route_segments=[dict(from_m=500,until_m=1000,
        mean={'front_cone':2.5},below_three_front_actor_fraction=.4,empty_ego_lane_fraction=.2)])
    path=tmp_path/'summary.json'
    path.write_text(json.dumps(dict(tasks={'a':{'status':'NOT_REACHED'}},traffic_density=density)))
    report=summarize([path])
    assert report['runs'][0]['traffic_density']==density
    output=markdown(report)
    assert '500-1000 m' in output
    assert 'below 3=40.0%' in output


def test_report_keeps_model_and_non_model_runs_identifiable(tmp_path):
    from benchmark.report import markdown
    paths = []
    for source in ('VLA_MODEL', 'NON_VLA_CONTROL', 'EXTERNAL'):
        path = tmp_path / (source + '.json')
        path.write_text(json.dumps({
            'scene_id': 'scene_2',
            'run_metadata': {'runner': {'policy_source': source}},
            'tasks': {'task': {'status': 'SUCCESS', 'instruction_status': 'SUCCESS'}},
        }))
        paths.append(path)
    report = summarize(paths)
    assert [row['policy_source'] for row in report['runs']] == [
        'VLA_MODEL', 'NON_VLA_CONTROL', 'EXTERNAL',
    ]
    table = markdown(report)
    assert '| Policy |' in table
    assert '| VLA_MODEL |' in table
    assert '| NON_VLA_CONTROL |' in table
