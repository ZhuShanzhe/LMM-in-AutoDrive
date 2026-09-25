from benchmark.capabilities import isolated_capability
from benchmark.catalog import load_catalog
from benchmark.planning import build_plan


def test_plan_describes_existing_adapter_without_claiming_readiness():
    plan=build_plan(load_catalog('scene_2'),'s2_t05_cmd_07')
    adapter=plan['isolated_adapters']['s2_t05_cmd_07']
    assert adapter['adapter_available']
    assert adapter['kind']=='overtake'
    assert not plan['execution_supported']
    assert not adapter['model_inference']
    assert 'runtime_adapter_not_connected' not in plan['blockers']


def test_prior_task_cannot_be_skipped_by_single_action_adapter():
    profile=dict(task_id='test',steps=[dict(kind='speed')],requires_task_success=['prior'])
    assert not isolated_capability('scene_1',profile)['adapter_available']
    assert not isolated_capability('scene_1',None)['adapter_available']


def test_registered_full_plan_distinguishes_supported_and_unsupported_tasks():
    plan=build_plan(load_catalog('scene_2'),'all')
    assert plan['isolated_adapters']['s2_t05_cmd_03']['adapter_available']
    assert not plan['isolated_adapters']['s2_t05_cmd_10']['adapter_available']
    assert 'full_runner_invocation_required' in plan['blockers']
