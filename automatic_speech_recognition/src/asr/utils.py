import numpy as np
import librosa
from scipy import signal


def spectral_subtraction(audio: np.ndarray, sr: int, noise_floor: float = 0.01) -> np.ndarray:
    """
    Simple spectral subtraction for noise reduction.
    Args:
        audio: input audio array (float32)
        sr: sample rate
        noise_floor: fraction of the first few frames used as noise estimate
    Returns:
        Enhanced audio array
    """
    n_fft = 2048
    hop_length = 512
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    noise_frames = int(0.2 * magnitude.shape[1])
    noise_est = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)

    magnitude_enhanced = magnitude - noise_floor * noise_est
    magnitude_enhanced = np.maximum(magnitude_enhanced, 0.0)

    stft_enhanced = magnitude_enhanced * np.exp(1j * phase)
    audio_enhanced = librosa.istft(stft_enhanced, hop_length=hop_length)
    return audio_enhanced.astype(np.float32)


def wiener_filter(audio: np.ndarray, sr: int, noise_std: float = 0.01) -> np.ndarray:
    """
    Wiener filter for noise reduction (simplified version).
    """
    # This is a placeholder – in practice we'd use a more robust implementation.
    # For simplicity, we fall back to spectral subtraction.
    return spectral_subtraction(audio, sr, noise_floor=noise_std)


def apply_dialect_mapping(text: str, dialect_map: dict) -> str:
    """
    Replace dialect-specific words with standard Mandarin equivalents.
    Args:
        text: recognized Chinese text
        dialect_map: dictionary mapping dialect words to standard words
    Returns:
        normalized text
    """
    for dialect_word, std_word in dialect_map.items():
        text = text.replace(dialect_word, std_word)
    return text
