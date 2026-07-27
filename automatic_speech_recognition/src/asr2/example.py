import logging
from asr2 import Qwen3ASRConfig
from asr2 import Qwen3ASRService

logging.basicConfig(level=logging.INFO)


def example_custom():
    """Load model from Hugging Face (default)"""
    config = Qwen3ASRConfig(
        load_type="custom",
        model_name="Qwen/Qwen3-ASR-1.7B",
        language="Chinese"
    )
    service = Qwen3ASRService(config)
    result = service.transcribe("data/example.wav", output_json="data/result.json")
    print(result["text"])


def example_local():
    """Load model from local directory"""
    config = Qwen3ASRConfig(
        load_type="local",
        model_path="./models/Qwen3-ASR-1.7B",
        language="Chinese"
    )
    service = Qwen3ASRService(config)
    result = service.transcribe("data/example.wav", output_json="data/result.json")
    print(result["text"])


if __name__ == "__main__":
    # example_custom()
    example_local()
