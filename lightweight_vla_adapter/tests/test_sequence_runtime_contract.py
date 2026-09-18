from lightweight_vla_adapter.scripts.smoke_sequence_runtime import smoke


def test_sequence_runtime_without_training_data_or_weights():
    result = smoke()
    assert result['status'] == 'pass'
    assert result['weight_mode'] == 'random contract fixture'
