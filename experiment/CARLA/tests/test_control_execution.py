import json
from types import SimpleNamespace as NS
import pytest
from evaluation.control_execution import ControlExecutionJournal


def setup():
    control=NS(throttle=.2,brake=0.,steer=.1,gear=1,reverse=False,hand_brake=False,manual_gear_shift=False)
    actor=NS(get_velocity=lambda:NS(x=2,y=0,z=0),
             get_transform=lambda:NS(location=NS(x=1,y=2,z=3),rotation=NS(yaw=5)))
    snapshot=NS(frame=10,timestamp=NS(elapsed_seconds=.5),find=lambda identity:actor)
    ego=NS(id=1,apply_control=lambda value:None,get_control=lambda:control)
    return control,snapshot,NS(get_snapshot=lambda:snapshot),ego


def test_submission_not_mistaken_for_physics_confirmation(tmp_path):
    control,snapshot,world,ego=setup()
    path=tmp_path/'execution.jsonl'
    journal=ControlExecutionJournal(path)
    journal.apply(world,ego,control,8)
    journal.observe(world,ego)
    assert journal.pending is not None
    snapshot.frame=11
    journal.observe(world,ego)
    journal.close()
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    assert [r['status'] for r in rows]==['SUBMISSION_REQUESTED','SUBMITTED','PHYSICS_OBSERVED']
    assert rows[-1]['speed_kmh']==7.2
    assert rows[-1]['decision_frame']==8


def test_failed_submission_and_unobserved_end_are_explicit(tmp_path):
    control,_,world,ego=setup()
    path=tmp_path/'failed.jsonl'
    journal=ControlExecutionJournal(path)
    def fail(value):
        raise RuntimeError('rpc failed')
    ego.apply_control=fail
    with pytest.raises(RuntimeError):
        journal.apply(world,ego,control,8)
    journal.close()
    assert json.loads(path.read_text().splitlines()[-1])['status']=='SUBMISSION_FAILED'
    ego.apply_control=lambda value:None
    path=tmp_path/'unobserved.jsonl'
    journal=ControlExecutionJournal(path)
    journal.apply(world,ego,control,8)
    journal.close()
    assert json.loads(path.read_text().splitlines()[-1])['status']=='NO_LATER_PHYSICS_OBSERVATION'
