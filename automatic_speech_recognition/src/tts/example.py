import logging
from tts import ChatTTSService

logging.basicConfig(level=logging.INFO)


def example_single_no_noise():
    """Single text, no noise (your original test)."""
    service = ChatTTSService(
        model_path="./models/rvcmd_linux_amd64",
        compile_model=True,
        output_dir="data/wav_files"
    )
    filepath = service.synthesize("前方路口先减速，后加速到40km/h")
    print(f"Generated: {filepath}")


def example_single_with_white_noise():
    """Single text with white noise."""
    service = ChatTTSService(
        model_path="./models/rvcmd_linux_amd64",
        compile_model=True,
        noise_enabled=True,
        noise_type="white",
        noise_level=0.01,
        output_dir="data/wav_files"
    )
    filepath = service.synthesize("今天天气真好啊！")
    print(f"Generated with white noise: {filepath}")


def example_batch():
    """Batch synthesis with custom batch_size."""
    service = ChatTTSService(
        model_path="./models/rvcmd_linux_amd64",
        compile_model=False,
        batch_size=2,
        output_dir="data/wav_files"
    )
    texts = [
        "你好，世界！",
        "欢迎使用语音合成。",
        "这是一个批量测试。",
        "感谢您的使用。"
    ]
    filepaths = service.synthesize(texts, batch_size=2)
    for fp in filepaths:
        print(fp)


def example_with_custom_noise():
    """Add noise from a file (if available)."""
    service = ChatTTSService(
        model_path="./models/rvcmd_linux_amd64",
        compile_model=True,
        output_dir="data/wav_files"
    )
    filepath = service.synthesize(
        "自定义噪声",
        add_noise=True,
        noise_type="from_file",
        noise_level=0.02,
        noise_file="data/wav_files/background_traffic.wav"
    )
    print(f"Generated with file noise: {filepath}")


if __name__ == "__main__":
    example_single_no_noise()
    # example_single_with_white_noise()
    # example_batch()
    # example_with_custom_noise()
