import json
from benchmark.cleanup import CleanupJournal


def test_failure_does_not_skip_later_restoration(tmp_path):
    cleanup=CleanupJournal()
    calls=[]
    def broken():
        raise RuntimeError('sensor disconnected')
    cleanup.attempt('destroy_sensor',broken)
    cleanup.attempt('restore_world',lambda:calls.append('restored'))
    result=cleanup.save(tmp_path/'cleanup.json')
    assert calls==['restored'] and not result['completed']
    assert result['steps'][0]['error_type']=='RuntimeError'
    assert json.loads((tmp_path/'cleanup.json').read_text())==result


def test_report_write_error_does_not_mask_original_failure(tmp_path):
    cleanup=CleanupJournal()
    result=cleanup.save(tmp_path/'absent'/'cleanup.json')
    assert not result['completed']
    assert result['steps'][-1]['step']=='write_cleanup_report'


def test_false_actor_destroy_result_is_failure():
    cleanup=CleanupJournal()
    cleanup.attempt('destroy_actor',lambda:False)
    assert not cleanup.result()['completed']
