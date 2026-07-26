import numpy as np
import librosa
from typing import Dict


def spectral_subtraction(audio: np.ndarray, sr: int, noise_floor: float = 0.01) -> np.ndarray:
    """Simple spectral subtraction for noise reduction."""
    n_fft = 2048
    hop_length = 512
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    # Estimate noise from the first 20% of frames (assuming silence at start)
    noise_frames = max(1, int(0.2 * magnitude.shape[1]))
    noise_est = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)

    # Apply spectral subtraction
    magnitude_enhanced = magnitude - noise_floor * noise_est
    magnitude_enhanced = np.maximum(magnitude_enhanced, 0.0)

    # Reconstruct the audio signal
    stft_enhanced = magnitude_enhanced * np.exp(1j * phase)
    audio_enhanced = librosa.istft(stft_enhanced, hop_length=hop_length)
    return audio_enhanced.astype(np.float32)


def wiener_filter(audio: np.ndarray, sr: int, noise_std: float = 0.01) -> np.ndarray:
    """Placeholder for Wiener filter (falls back to spectral subtraction)."""
    # In a real implementation, a more sophisticated Wiener filter would be used.
    # For simplicity, we use spectral subtraction.
    return spectral_subtraction(audio, sr, noise_floor=noise_std)


def apply_dialect_mapping(text: str, dialect_map: Dict[str, str]) -> str:
    """Replace dialect words with standard equivalents."""
    for dialect_word, std_word in dialect_map.items():
        text = text.replace(dialect_word, std_word)
    return text


DEFAULT_DIALECT_MAP = {
    "啥子": "什么",
    "咋子": "怎么",
    "整": "做",
    "晓得": "知道",
    "要得": "可以",
    "爪子": "干什么",
    "啷个": "怎么",
    "莫得": "没有",
}