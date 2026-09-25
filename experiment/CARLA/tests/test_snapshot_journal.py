import json
from types import SimpleNamespace as NS
import pytest

from benchmark.catalog import ConfigError
from benchmark.snapshot_journal import SnapshotJournal
from benchmark.truth_capture import ActorBinding


def snapshot(frame):
    actor=NS(get_transform=lambda:NS(location=NS(x=frame,y=2,z=3),rotation=NS(yaw=5)),
             get_velocity=lambda:NS(x=4,y=0,z=0))
    return NS(frame=frame,timestamp=NS(elapsed_seconds=frame*.05),
              find=lambda identity:actor if identity==1 else None)


def test_incremental_snapshot_replay_and_missing_role(tmp_path):
    journal=SnapshotJournal(tmp_path/'capture',{'ego':ActorBinding(1,2,1,0),'target':ActorBinding(2,2,1,20)})
    journal.append(snapshot(1))
    row=json.loads((journal.output/'snapshots.jsonl').read_text())
    assert row['missing_roles']==['target']
    journal.append(snapshot(2))
    journal.close(completed=True)
    restored=list(journal.replay(NS))
    assert len(restored)==2
    assert restored[1].find(1).get_transform().location.x==2
    assert restored[1].find(2) is None
    assert journal.manifest['status']=='CAPTURED'


def test_interrupted_capture_is_preserved_but_not_accepted(tmp_path):
    journal=SnapshotJournal(tmp_path/'capture',{'ego':ActorBinding(1,2,1,0)})
    journal.append(snapshot(1))
    with pytest.raises(ConfigError,match='increase'):
        journal.append(snapshot(1))
    journal.close()
    assert journal.manifest['frames']==1
    assert journal.manifest['status']=='INTERRUPTED'
    with pytest.raises(ConfigError,match='completed'):
        list(journal.replay(NS))


def test_modified_capture_rejected(tmp_path):
    journal=SnapshotJournal(tmp_path/'capture',{'ego':ActorBinding(1,2,1,0)})
    journal.append(snapshot(1))
    journal.close(completed=True)
    (journal.output/'snapshots.jsonl').write_text('{}\n')
    with pytest.raises(ConfigError,match='changed'):
        list(journal.replay(NS))
