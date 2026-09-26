"""Integration contracts only: model and board inference are mocked."""
import importlib
from unittest import mock

import pytest


def test_lazy_pipeline_export():
    package = importlib.import_module('automatic_speech_recognition.src')
    module = importlib.import_module('automatic_speech_recognition.src.pipeline')
    assert package.ASRPipeline is module.ASRPipeline


def test_optional_backend_uses_public_package_import():
    module = importlib.import_module('automatic_speech_recognition.src.pipeline')
    deployment = importlib.import_module('automatic_speech_recognition.src.asr.deployment')
    with mock.patch.object(module, 'Qwen3ASRService'), mock.patch.object(deployment, 'ASRRuntime') as runtime:
        pipeline = module.ASRPipeline(asr_device='cpu', asr_num_gpus=0, asr_backend='onnx')
        assert pipeline.asr is runtime.return_value
        assert runtime.call_args.kwargs['backend'] == 'onnx'


@pytest.mark.parametrize('initial', [RuntimeError('frontend failure'), ({'language': 'Chinese'}, '', True)])
def test_optimized_failure_retries_raw_audio(initial, tmp_path):
    from automatic_speech_recognition.src.asr.qwen3_asr_service import Qwen3ASRService
    audio = tmp_path/'fixture.wav'
    audio.touch()
    with mock.patch.object(Qwen3ASRService, '_load_model'):
        service = Qwen3ASRService(device='cpu', num_gpus=0, frontend=mock.Mock())
    with mock.patch.object(service, '_transcribe_once', side_effect=[initial, ({'language': 'Chinese'}, '停车', False)]) as forward:
        result = service.transcribe_file(str(audio))
    assert result['text'] == '停车'
    assert result['optimization_fallback'] is True
    assert result['frontend_applied'] is False
    assert forward.call_count == 2


def test_disabled_fallback_does_not_retry(tmp_path):
    from automatic_speech_recognition.src.asr.qwen3_asr_service import Qwen3ASRService
    audio = tmp_path/'fixture.wav'
    audio.touch()
    with mock.patch.object(Qwen3ASRService, '_load_model'):
        service = Qwen3ASRService(device='cpu', num_gpus=0, frontend=mock.Mock(), fallback_on_failure=False)
    with mock.patch.object(service, '_transcribe_once', side_effect=RuntimeError('failure')) as forward:
        result = service.transcribe_file(str(audio))
    assert result['success'] is False
    assert 'error' in result
    assert forward.call_count == 1
