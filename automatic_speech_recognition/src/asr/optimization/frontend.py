import logging
from typing import Optional

import librosa
import numpy as np

logger = logging.getLogger(__name__)

def highpass(audio: np.ndarray, sr: int, cutoff: float = 80.0) -> np.ndarray:
    try:
        from scipy.signal import butter, sosfilt
        sos = butter(4, cutoff, btype="highpass", fs=sr, output="sos")
        return sosfilt(sos, audio).astype(np.float32)
    except Exception:
        if audio.size == 0:
            return audio
        spec = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(audio.size, 1.0 / sr)
        spec[freqs < cutoff] = 0.0
        return np.fft.irfft(spec, n=audio.size).astype(np.float32)

def pre_emphasis(audio: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    if audio.size == 0:
        return audio
    return np.append(audio[0], audio[1:] - coeff * audio[:-1]).astype(np.float32)

def voice_activity_trim(audio: np.ndarray, sr: int, frame_ms: int = 25, hop_ms: int = 10, energy_ratio: float = 0.12, pad_ms: int = 100) -> np.ndarray:
    if audio.size == 0:
        return audio
    frame = max(1, int(sr * frame_ms / 1000))
    hop = max(1, int(sr * hop_ms / 1000))
    energies = np.array([
        float(np.sum(audio[i:i + frame] ** 2))
        for i in range(0, max(1, audio.size - frame), hop)
    ])
    if energies.size == 0:
        return audio
    thresh = energy_ratio * float(np.max(energies))
    voiced = np.where(energies > thresh)[0]
    if voiced.size == 0:
        return audio
    pad = int(sr * pad_ms / 1000)
    start = max(0, voiced[0] * hop - pad)
    end = min(audio.size, (voiced[-1] + 1) * hop + pad)
    return audio[start:end].astype(np.float32)

def _noise_power(mag: np.ndarray, ratio: float = 0.1) -> np.ndarray:
    frame_energy = np.mean(mag ** 2, axis=0)
    n_noise = max(1, int(ratio * mag.shape[1]))
    idx = np.argsort(frame_energy)[:n_noise]
    return np.mean(mag[:, idx] ** 2, axis=1, keepdims=True)

def spectral_subtraction(audio: np.ndarray, sr: int, alpha: float = 2.0, n_fft: int = 512, hop_length: int = 128) -> np.ndarray:
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    mag, phase = np.abs(stft), np.angle(stft)
    noise_pow = _noise_power(mag)
    power = mag ** 2
    clean = np.maximum(power - alpha * noise_pow, 0.0)
    floor = 0.05 * power  # spectral floor suppresses musical noise
    clean = np.maximum(clean, floor)
    out = librosa.istft(np.sqrt(clean) * np.exp(1j * phase), hop_length=hop_length)
    return out.astype(np.float32)

def wiener_filter(audio: np.ndarray, sr: int, n_fft: int = 512, hop_length: int = 128, smooth: float = 0.98) -> np.ndarray:
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    mag, phase = np.abs(stft), np.angle(stft)
    noise_pow = _noise_power(mag)
    power = mag ** 2
    snr_post = power / (noise_pow + 1e-12)
    # Decision-directed a-priori SNR smoothing.
    gain = snr_post / (1.0 + snr_post)
    gain = smooth * gain + (1.0 - smooth) * snr_post / (1.0 + snr_post)
    out = librosa.istft((mag * np.clip(gain, 0.0, 1.0)) * np.exp(1j * phase), hop_length=hop_length)
    return out.astype(np.float32)

def deepfilternet_denoise(audio: np.ndarray, sr: int) -> np.ndarray:
    try:
        import torch
        from df.enhance import enhance, init_df
        model, state, _ = init_df()
        target_sr = state.sr()
        sig = librosa.resample(audio, orig_sr=sr, target_sr=target_sr) if sr != target_sr else audio
        tensor = torch.from_numpy(sig).unsqueeze(0)
        out = enhance(model, state, tensor).squeeze(0).detach().cpu().numpy()
        if target_sr != sr:
            out = librosa.resample(out, orig_sr=target_sr, target_sr=sr)
        return out.astype(np.float32)
    except Exception as exc:  # noqa: BLE001
        logger.warning("DeepFilterNet unavailable (%s); using Wiener filter.", exc)
        return wiener_filter(audio, sr)

def rms_normalize(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    rms = float(np.sqrt(np.mean(audio ** 2)) + 1e-12)
    target = 10 ** (target_db / 20.0)
    return (audio * (target / rms)).astype(np.float32)

def peak_limit(audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
    m = float(np.max(np.abs(audio))) if audio.size else 0.0
    if m > peak:
        audio = audio * (peak / m)
    return audio.astype(np.float32)

class AudioFrontend:
    _METHODS = {"spectral", "wiener", "deepfilternet", "none"}

    def __init__(
        self, 
        method: str = "wiener", 
        vad: bool = True,
        highpass_filter: bool = True, 
        highpass_cutoff: float = 80.0,
        preemphasis: bool = True, 
        rms_norm: bool = True,
        target_db: float = -20.0
    ):
        if method not in self._METHODS:
            raise ValueError("unsupported enhancement method: " + str(method))
        self.method = method
        self.vad = vad
        self.highpass_filter = highpass_filter
        self.highpass_cutoff = highpass_cutoff
        self.preemphasis = preemphasis
        self.rms_norm = rms_norm
        self.target_db = target_db

    def _denoise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if self.method == "spectral":
            return spectral_subtraction(audio, sr)
        if self.method == "wiener":
            return wiener_filter(audio, sr)
        if self.method == "deepfilternet":
            return deepfilternet_denoise(audio, sr)
        return audio

    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        out = audio
        if self.vad:
            out = voice_activity_trim(out, sr)
        if self.highpass_filter:
            out = highpass(out, sr, self.highpass_cutoff)
        out = self._denoise(out, sr)
        if self.preemphasis:
            out = pre_emphasis(out)
        if self.rms_norm:
            out = rms_normalize(out, self.target_db)
        return peak_limit(out)