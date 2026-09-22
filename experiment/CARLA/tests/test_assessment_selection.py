from types import SimpleNamespace
import pytest
from benchmark.catalog import load_catalog, ConfigError
from benchmark.selection import assessment_selection, validate_assessment_args


def test_selected_compound_task_includes_prior_evidence():
    selected, evidence = assessment_selection(load_catalog('scene_2'), 's2_t05_cmd_04')
    assert [task.task_id for task in selected] == ['s2_t05_cmd_04']
    assert {task.task_id for task in evidence} == {'s2_t05_cmd_03', 's2_t05_cmd_04'}


@pytest.mark.parametrize('scene,count',[('scene_1',15),('scene_2',15),('scene_3',8)])
def test_default_keeps_full_catalog(scene,count):
    selected,evidence=assessment_selection(load_catalog(scene))
    assert len(selected)==len(evidence)==count


def test_selection_without_assessment_rejected():
    with pytest.raises(ConfigError,match='requires'):
        validate_assessment_args(SimpleNamespace(benchmark_task='1',benchmark_assessment=False),'scene_1')
