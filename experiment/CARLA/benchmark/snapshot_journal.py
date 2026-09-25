"""Incremental actor snapshots for bounded-memory, post-sensor-finalization assessment."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .catalog import ConfigError
from .episode import pack_actor, unpack_snapshot


class SnapshotJournal:
    def __init__(self,output,bindings):
        self.output=Path(output)
        self.output.mkdir(parents=True,exist_ok=False)
        self.bindings=dict(bindings)
        self.frames=0
        self.last_frame=None
        self.last_time=None
        self.closed=False
        self.manifest=dict(schema_version='snapshot_journal/1.0',status='RECORDING',
            policy_inputs=False,bindings={name:asdict(b) for name,b in self.bindings.items()},
            scope='ego and task actors only; safety events must be finalized separately')
        self._save_manifest()
        self.stream=(self.output/'snapshots.jsonl').open('x',encoding='utf-8')

    def _save_manifest(self):
        path=self.output/'manifest.json'
        temporary=self.output/'manifest.tmp'
        temporary.write_text(json.dumps(self.manifest,indent=2,allow_nan=False),encoding='utf-8')
        temporary.replace(path)

    def append(self,snapshot):
        if self.closed:
            raise RuntimeError('snapshot journal closed')
        now=snapshot.timestamp.elapsed_seconds
        if self.last_frame is not None and (snapshot.frame<=self.last_frame or now<=self.last_time):
            raise ConfigError('snapshot journal frames/time must increase')
        actors={}
        missing=[]
        for name,binding in self.bindings.items():
            actor=snapshot.find(binding.actor_id)
            if actor is None:
                missing.append(name)
            else:
                actors[str(binding.actor_id)]=pack_actor(actor)
        row=dict(frame=snapshot.frame,sim_time_s=now,actors=actors,missing_roles=missing)
        self.stream.write(json.dumps(row,allow_nan=False)+'\n')
        self.stream.flush()
        self.frames+=1
        self.last_frame=snapshot.frame
        self.last_time=now

    def close(self,completed=False):
        if self.closed:
            return
        self.stream.close()
        with (self.output/'snapshots.jsonl').open('rb') as stream:
            digest=hashlib.file_digest(stream,'sha256').hexdigest()
        self.manifest.update(status='CAPTURED' if completed else 'INTERRUPTED',
                             frames=self.frames,snapshots_sha256=digest)
        self._save_manifest()
        self.closed=True

    def replay(self,location_factory):
        if not self.closed or self.manifest['status']!='CAPTURED':
            raise ConfigError('completed snapshot capture required for assessment')
        with (self.output/'snapshots.jsonl').open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=self.manifest['snapshots_sha256']:
                raise ConfigError('snapshot journal changed after capture')
        with (self.output/'snapshots.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                yield unpack_snapshot(json.loads(line),location_factory)
