from types import SimpleNamespace
import pytest

from benchmark.route_sections import partition, validate_sections, event_extent
from tools.build_route_sections import coalesce_samples


def task(identity, start, end=None, dependencies=None):
    return SimpleNamespace(task_id=identity, instruction=identity, announce_m=start,
        activate_m=start, end_m=end, required_event_states=dependencies or {},
        linked_event_ids=(), source_command={})


def catalog():
    return SimpleNamespace(scene_id='scene_2', map_name='Town05_Opt', route_length_m=1000.,
        source_file='source.json', source_sha256='hash',
        tasks=[task('a', 0.), task('b', 600., dependencies={'pedestrian':'RESOLVED'})],
        events=[dict(id='pedestrian', activate_progress_m=450., anchor_progress_m=500.)])


def samples():
    return [dict(s_m=float(s), x=float(s), y=0., z=0., yaw_deg=0., road_id=1,
        lane_id=-1, lane_width_m=3.5, same_direction_lanes=2 if s < 750 else 1,
        is_junction=200 <= s <= 250) for s in range(0, 1001, 10)]


def test_full_coverage_and_semantic_boundaries():
    result = partition(catalog(), samples())
    assert validate_sections(result)
    assert any(s['start_m'] == 600. for s in result['sections'])
    assert set(t for s in result['sections'] for t in s['task_ids']) == {'a','b'}
    assert all(s['end_m'] - s['start_m'] <= 400.001 for s in result['sections'])
    assert any('junction' in s['geometry_kinds'] for s in result['sections'])
    assert any('lane_count_change' in s['geometry_kinds'] for s in result['sections'])
    assert len(result['phases']) == 2
    assert result['phases'][0]['start_m'] == 0. and result['phases'][-1]['end_m'] == 1000.


def test_prerequisite_replay_is_not_a_fake_completed_event():
    result = partition(catalog(), samples())
    for s in result['sections']:
        if 'b' in s['task_ids']:
            assert s['replay_start_m'] <= 300.
            assert s['prerequisites'] == {'pedestrian':'RESOLVED'}
            assert not s['cold_start_valid_for_task_completion']
    assert result['events'][0]['explicit_end_m'] is None
    assert all(not t['end_is_completion_criterion'] for t in result['tasks'])


def test_nested_event_positions_and_no_variant_mixing():
    event = dict(id='closure', activate_at_m=200., resolve_after_m=400.,
        closure=dict(taper_start_s_m=250., taper_end_s_m=350.),
        variants=[dict(distance_m=900.)])
    row = event_extent(event, 1000.)
    assert row['start_m'] == 200. and row['focus_end_m'] == 400.
    assert len(row['positions']) == 4


def test_historical_failures_are_not_asserted_same_route():
    result = partition(catalog(), samples(), failures=[dict(s_m=800., source='record.json')])
    linked = [f for s in result['sections'] for f in s['historical_failures']]
    assert len(linked) == 1
    assert next(b for b in result['breakpoints'] if b['kind']=='historical_failure')['route_equivalence']=='not_verified'


@pytest.mark.parametrize('fault', ['unordered', 'nonfinite', 'duplicate'])
def test_bad_geometry_rejected(fault):
    points = samples()
    if fault == 'unordered': points.reverse()
    if fault == 'nonfinite': points[1]['s_m'] = float('nan')
    if fault == 'duplicate': points[1]['s_m'] = points[0]['s_m']
    with pytest.raises(ValueError): partition(catalog(), points)


def test_nonfinite_options_rejected():
    with pytest.raises(ValueError): partition(catalog(), samples(), maximum_section_m=float('nan'))


def test_coincident_route_connector_samples_can_be_coalesced():
    points = samples()
    points.insert(5, dict(points[5], road_id=2))
    assert len(coalesce_samples(points)) == 101


def test_duplicate_progress_with_different_pose_is_rejected():
    points = samples()
    points.insert(5, dict(points[5], x=999.))
    with pytest.raises(ValueError): coalesce_samples(points)
