import numpy as np
import librosa
import soundfile as sf
from typing import Optional


def white_noise(length: int, amplitude: float = 0.005) -> np.ndarray:
    """Generate white noise."""
    return amplitude * np.random.randn(length).astype(np.float32)


def pink_noise(length: int, amplitude: float = 0.005) -> np.ndarray:
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


def load_noise_file(filepath: str, target_length: int, target_sr: int) -> np.ndarray:
    """Load a noise file, resample, and trim/repeat to target length."""
    audio, sr = librosa.load(filepath, sr=target_sr, mono=True)
    if len(audio) < target_length:
        repeats = (target_length // len(audio)) + 1
        audio = np.tile(audio, repeats)
    return audio[:target_length].astype(np.float32)


def vehicle_noise(length: int, sr: int, noise_file: Optional[str] = None, amplitude: float = 0.005) -> np.ndarray:
    """
    Generate vehicle-like noise (engine, road, etc.).
    If noise_file is provided, use that; otherwise synthesize.
    """
    if noise_file is not None:
        noise = load_noise_file(noise_file, length, sr)
    else:
        pink = pink_noise(length, amplitude)
        t = np.arange(length) / sr
        low_freq = 0.5 * np.sin(2 * np.pi * 2 * t)
        engine = 0.3 * amplitude * low_freq
        combined = pink + engine
        max_val = np.max(np.abs(combined))
        if max_val > 0:
            combined = combined / max_val * amplitude
        noise = combined.astype(np.float32)

    if len(noise) != length:
        if len(noise) > length:
            noise = noise[:length]
        else:
            pad_len = length - len(noise)
            noise = np.pad(noise, (0, pad_len), mode='constant', constant_values=0)
    return noise
