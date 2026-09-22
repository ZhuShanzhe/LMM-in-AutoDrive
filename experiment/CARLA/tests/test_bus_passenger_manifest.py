from benchmark.catalog import load_catalog
from benchmark.role_journal import configured_role_anchors
from benchmark.task_oracle import load_profile


def test_bus_passengers_explicit_and_retained():
    catalog=load_catalog('scene_2')
    event=next(e for e in catalog.events if e['kind']=='bus_stop')
    passengers=event['passengers']
    roles=[p['role_name'] for p in passengers]
    assert len(roles)==len(set(roles))==3
    assert set(roles)<=set(event['ground_truth']['actor_roles'])
    assert event['retain_after_completion'] is True
    anchors=configured_role_anchors(catalog.events)
    assert all(anchors[r]==event['anchor_progress_m'] for r in roles)


def test_all_existing_scene2_profiles_remain_bound_after_configuration_update():
    catalog=load_catalog('scene_2')
    expected={f's2_t05_cmd_{n:02d}' for n in range(1,16)}
    actual={t.task_id for t in catalog.tasks if load_profile(catalog,t) is not None}
    assert expected==actual
