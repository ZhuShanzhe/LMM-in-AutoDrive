from dataclasses import dataclass


@dataclass
class RecorderConfig:
    """Configuration for audio recording."""

    sample_rate: int = 16000                # Sampling rate (Hz), 16kHz is recommended for ASR
    channels: int = 1                       # Mono audio
    dtype: str = "float32"                  # Data type for recording
    default_duration: float = 5.0           # Default recording duration (seconds)
    output_dir: str = "data/recordings"     # Directory to save WAV files
    countdown: bool = True                  # Enable countdown before recording
    countdown_seconds: int = 3              # Number of seconds for countdown
