"""CPU-only package integration checks; no audio model weights are loaded."""
import importlib
from unittest import mock


def test_public_package_exports_pipeline():
    package = importlib.import_module("automatic_speech_recognition")
    pipeline = importlib.import_module("automatic_speech_recognition.src.pipeline")
    assert package.ASRPipeline is pipeline.ASRPipeline
    root = importlib.import_module("automatic_speech_recognition.src")
    assert root.Qwen3ASRService is pipeline.Qwen3ASRService


def test_pipeline_preserves_explicit_english_and_guard_metadata():
    module = importlib.import_module("automatic_speech_recognition.src.pipeline")
    with mock.patch.object(module, "Qwen3ASRService") as service:
        service.return_value.transcribe_file.return_value = {
            "text": "向左转", "language": "Chinese", "guard_safe": True,
            "guard_reasons": [], "processing_time_seconds": .01,
        }
        pipeline = module.ASRPipeline(
            asr_device="cpu", asr_num_gpus=0,
            translator_device="cpu", translator_num_gpus=0,
            enable_translation=True, output_language="english",
        )
        pipeline.translator = mock.Mock()
        pipeline.translator.translate_zh_to_en.return_value = "Turn left."
        result = pipeline.process("fixture.wav")
    assert result["output_text"] == "Turn left."
    assert result["translation_applied"] is True
    assert result["guard_safe"] is True
    assert result["text"] == "向左转"
