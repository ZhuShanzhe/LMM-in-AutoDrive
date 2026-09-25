import pytest

from benchmark.catalog import ConfigError
from benchmark.safety_events import SafetyLedger


def test_future_events_are_not_assigned_to_earlier_frames():
    ledger=SafetyLedger()
    ledger.record('collision',12,other_actor_id=2)
    ledger.record('collision',10,other_actor_id=3)
    ledger.record('collision',10,other_actor_id=3)
    ledger.seal_after_quiet(.01,.2)
    assert ledger.packet(9)['collisions']==0
    assert ledger.packet(10)['collisions']==1
    assert ledger.packet(12)['collisions']==2


def test_late_event_invalidates_sealed_ledger():
    ledger=SafetyLedger()
    with pytest.raises(ConfigError):
        ledger.packet(0)
    ledger.seal_after_quiet(.01,.2)
    ledger.record('collision',0,other_actor_id=2)
    with pytest.raises(ConfigError):
        ledger.packet(1)
    assert len(ledger.export()['late_events'])==1


def test_legal_lane_marking_is_not_violation_and_scope_is_explicit():
    ledger=SafetyLedger()
    ledger.record('lane_invasion',1,illegal=False)
    ledger.record('lane_invasion',2,illegal=True)
    ledger.seal_after_quiet(.01,.2)
    assert ledger.packet(1)['violations']==0
    assert ledger.packet(2)['violations']==1
    assert not ledger.packet(2)['transport_acknowledged']


def test_runtime_rejects_unimplemented_task_before_connecting():
    from benchmark.runtime import run_speed_fixture
    with pytest.raises(ConfigError,match='single speed, lane-change or turn'):
        run_speed_fixture('scene_1','c15_keep_to_goal','missing','missing','missing')


def test_episode_safety_includes_events_after_task_success():
    ledger=SafetyLedger()
    assert ledger.episode_result()['status']=='UNKNOWN'
    ledger.record('collision',1000,other_actor_id=3)
    ledger.seal_after_quiet(.01,.2)
    assert ledger.packet(10)['collisions']==0
    assert ledger.episode_result()['status']=='FAILURE'


def test_late_event_makes_episode_safety_unknown():
    ledger=SafetyLedger()
    ledger.seal_after_quiet(.01,.2)
    assert ledger.episode_result()['status']=='NO_RECORDED_VIOLATION'
    ledger.record('lane_invasion',10,illegal=False)
    assert ledger.episode_result()['status']=='UNKNOWN'
