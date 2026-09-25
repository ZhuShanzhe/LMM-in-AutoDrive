import io
import json
from types import SimpleNamespace as NS

import pytest

from benchmark.episode import assessment_outcome, finish_episode


def summary(statuses,safety='NO_RECORDED_VIOLATION'):
    return dict(tasks={str(i):{'status':s} for i,s in enumerate(statuses)},
                episode_safety={'status':safety})


@pytest.mark.parametrize('statuses,safety,expected',[
    (['SUCCESS'],'NO_RECORDED_VIOLATION',0),
    (['SUCCESS','NOT_RUN'],'NO_RECORDED_VIOLATION',0),
    (['SUCCESS'],'FAILURE',2),
    (['TIMEOUT','UNSUPPORTED'],'NO_RECORDED_VIOLATION',2),
    (['FAILURE'],'NO_RECORDED_VIOLATION',2),
    (['SUCCESS','UNSUPPORTED'],'NO_RECORDED_VIOLATION',3),
    (['SCENE_INVALID'],'NO_RECORDED_VIOLATION',3),
    (['NOT_REACHED'],'NO_RECORDED_VIOLATION',3),
    (['NOT_RUN'],'NO_RECORDED_VIOLATION',3),
    (['SUCCESS'],'UNKNOWN',3),
    ([],'NO_RECORDED_VIOLATION',3),
])
def test_exit_code_does_not_hide_missing_or_failed_evidence(statuses,safety,expected):
    outcome=assessment_outcome(summary(statuses,safety))
    assert outcome['exit_code']==expected
    assert outcome['full_benchmark_acceptance'] is False


def test_finish_failure_is_recorded_and_idempotent(tmp_path):
    calls=[]
    def fail():
        calls.append(True)
        raise RuntimeError('broken assessment')
    observer=NS(close=fail,stream=io.StringIO(),output=tmp_path)
    assert finish_episode(observer)==4
    assert finish_episode(observer)==4
    assert len(calls)==1 and observer.stream.closed
    assert json.loads((tmp_path/'summary.json').read_text())['status']=='ASSESSMENT_ERROR'
    assert json.loads((tmp_path/'run_outcome.json').read_text())['exit_code']==4


def test_finish_preserves_summary_and_reports_task_failure(tmp_path):
    existing='original evidence'
    (tmp_path/'summary.json').write_text(existing)
    observer=NS(close=lambda:summary(['TIMEOUT']),output=tmp_path)
    assert finish_episode(observer)==2
    assert (tmp_path/'summary.json').read_text()==existing


def test_result_write_failure_returns_error_without_interrupting_cleanup(tmp_path):
    observer=NS(close=lambda:summary(['SUCCESS']),output=tmp_path/'missing')
    assert finish_episode(observer)==4


def test_disabled_assessment_preserves_runner_behavior():
    assert finish_episode(None)==0
