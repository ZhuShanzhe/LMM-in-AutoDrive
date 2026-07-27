import numpy as np
import soundfile as sf
import librosa
from typing import Optional


def generate_white_noise(length: int, amplitude: float = 0.005) -> np.ndarray:
    """Generate white noise."""
    return amplitude * np.random.randn(length).astype(np.float32)


def generate_pink_noise(length: int, amplitude: float = 0.005) -> np.ndarray:
    """Generate pink noise (1/f spectrum)."""
    n = length
    white = np.random.randn(n)

    if n % 2 == 0:
        freq = np.fft.rfftfreq(n)
    else:
        freq = np.fft.rfftfreq(n)[:-1]

    with np.errstate(divide='ignore'):
        pink_filter = 1.0 / np.sqrt(freq[1:])
    pink_filter = np.concatenate(([1.0], pink_filter))
    fft = np.fft.rfft(white)
    fft[1:] *= pink_filter
    pink = np.fft.irfft(fft, n=n)

    max_val = np.max(np.abs(pink))
    if max_val > 0:
        pink = pink / max_val * amplitude

    return pink.astype(np.float32)


def generate_brown_noise(length: int, amplitude: float = 0.005) -> np.ndarray:
    """Generate Brownian noise (1/f^2 spectrum)."""
    white = np.random.randn(length)
    brown = np.cumsum(white)
    max_val = np.max(np.abs(brown))
    if max_val > 0:
        brown = brown / max_val * amplitude
    return brown.astype(np.float32)


def load_noise_from_file(file_path: str, target_length: int, target_sr: int) -> np.ndarray:
    """Load noise file, resample, trim/repeat to target length."""
    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    if len(audio) < target_length:
        repeats = (target_length // len(audio)) + 1
        audio = np.tile(audio, repeats)
    return audio[:target_length].astype(np.float32)
