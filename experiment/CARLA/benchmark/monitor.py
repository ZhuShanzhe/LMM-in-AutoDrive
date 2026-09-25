"""Attach independent task evidence to an existing runner without taking tick ownership."""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

from .catalog import ConfigError
from .task_oracle import TaskOracle, TERMINAL


class TaskMonitor:
    def __init__(self, collector, output):
        self.collector = collector
        self.oracle = TaskOracle(collector.spec)
        self.output = Path(output)
        self.output.mkdir(parents=True,exist_ok=False)
        self.closed = False
        self.timings = []
        hashes = {}
        for name,value in dict(spec=collector.spec,route=collector.projector.route,fixture=collector.fixture).items():
            payload = (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')
            (self.output/f'{name}.json').write_bytes(payload)
            hashes[f'{name}_sha256'] = hashlib.sha256(payload).hexdigest()
        self.spec_sha256 = hashes['spec_sha256']
        manifest = dict(schema_version='task_capture/1.0', task_id=collector.spec['task_id'],
                        source_sha256=collector.spec.get('source_sha256'),
                        spec=collector.spec, fixture=collector.fixture,
                        bindings={k:asdict(v) for k,v in collector.bindings.items()},
                        **hashes,
                        scope='task_evaluation_only', sensor_capture_included=False)
        (self.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        self.stream = (self.output/'task_truth.jsonl').open('x',encoding='utf-8')

    def observe(self, snapshot, safety, validity):
        if self.closed:
            raise RuntimeError('monitor is closed')
        if self.oracle.status in TERMINAL:
            return self.feedback()
        start = time.perf_counter()
        try:
            observation = self.collector.collect(snapshot,safety,validity)
        except (ConfigError,KeyError,TypeError,ValueError) as error:
            raw_time = snapshot.timestamp.elapsed_seconds
            safe_time = raw_time if isinstance(raw_time,(float,int)) and math.isfinite(raw_time) else None
            observation = dict(schema_version='task_truth/1.0',source='simulator_truth',
                               frame=snapshot.frame if type(snapshot.frame) is int else None,sim_time_s=safe_time,
                               task_id=self.collector.spec['task_id'],
                               source_sha256=self.collector.spec.get('source_sha256'),
                               scenario_valid=False,invalid_reason=str(error))
        self.oracle.update(observation)
        elapsed = (time.perf_counter()-start)*1000
        self.timings.append(elapsed)
        observation['evaluation_wall_ms'] = elapsed
        self.stream.write(json.dumps(observation,ensure_ascii=False,allow_nan=False)+'\n')
        self.stream.flush()
        return self.feedback()

    def feedback(self):
        # No actor geometry or scripted target values cross the feedback boundary.
        result=self.oracle.result()
        return dict(task_id=self.oracle.spec['task_id'],status=self.oracle.status,
                    completed_steps=self.oracle.index,reason=self.oracle.reason,
                    instruction_status=result['instruction_status'],coverage=result['coverage'])

    def close(self):
        if not self.closed:
            self.oracle.end_of_stream()
            self.stream.close()
            result = self.oracle.result()
            with (self.output/'task_truth.jsonl').open('rb') as stream:
                result['observations_sha256'] = hashlib.file_digest(stream,'sha256').hexdigest()
            result['spec_sha256'] = self.spec_sha256
            result['evaluation_wall_mean_ms'] = sum(self.timings)/len(self.timings) if self.timings else None
            result['evaluation_frames'] = len(self.timings)
            result['scope'] = 'task_assessment_not_episode_safety_or_model_latency'
            (self.output/'task_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
            self.closed = True
        return self.feedback()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
