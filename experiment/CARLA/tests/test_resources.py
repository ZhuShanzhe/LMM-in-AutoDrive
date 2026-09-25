import json
import threading
from types import SimpleNamespace
from evaluation import resources


def test_gpu_parser_keeps_unavailable_values_distinct_from_zero(monkeypatch):
    monkeypatch.setattr(resources.shutil,'which',lambda name:'nvidia-smi')
    monkeypatch.setattr(resources.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=0,
        stdout='GPU-1, Test GPU, 123, 24000, N/A\n',stderr=''))
    row=resources.gpu_snapshot()
    assert row['scope']=='whole_device_all_processes'
    assert row['devices'][0]['used_mib']==123
    assert row['devices'][0]['utilization_pct'] is None


def test_missing_driver_not_zero_usage(monkeypatch):
    monkeypatch.setattr(resources.shutil,'which',lambda name:None)
    assert resources.gpu_snapshot()['status']=='UNAVAILABLE'


def test_sampler_writes_and_shuts_down(tmp_path,monkeypatch):
    sampler=resources.ResourceSampler(tmp_path/'resources.jsonl')
    sampled=threading.Event()
    def sample():
        sampled.set()
        return {'elapsed_wall_s':0,'test_fixture':True}
    monkeypatch.setattr(sampler,'_sample',sample)
    sampler.start()
    assert sampled.wait(2)
    summary=sampler.close()
    assert summary['sample_count']==1
    assert not sampler.thread.is_alive()
    assert json.loads((tmp_path/'resources.jsonl').read_text())['test_fixture']


def test_online_capture_failure_does_not_raise_or_overwrite(tmp_path, monkeypatch):
    journal = resources.OnlineResourceJournal(tmp_path/'decisions.jsonl')
    journal.path.write_text('previous run', encoding='utf-8')
    journal.start()
    journal.start()
    result = journal.close()
    assert result['status'] == 'UNAVAILABLE'
    assert result['error']
    assert journal.path.read_text() == 'previous run'
    assert journal.close() is result


def test_online_capture_lifecycle(tmp_path, monkeypatch):
    calls = []
    class Sampler:
        def __init__(self, path):
            pass
        def start(self):
            calls.append('start')
            return self
        def close(self):
            calls.append('close')
            return {'sample_count': 2, 'errors': []}
    monkeypatch.setattr(resources, 'ResourceSampler', Sampler)
    journal = resources.OnlineResourceJournal(tmp_path/'decisions.jsonl')
    journal.start()
    journal.start()
    result = journal.close()
    journal.close()
    assert calls == ['start', 'close']
    assert result['status'] == 'RECORDED'
    assert json.loads(journal.summary_path.read_text()) == result
