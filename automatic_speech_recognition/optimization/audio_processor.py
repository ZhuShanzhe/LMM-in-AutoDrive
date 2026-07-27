import os
import random
import numpy as np
import soundfile as sf
import librosa
from .config import Config


class AudioProcessor:
    """
    Audio post‑processing utilities, including background noise addition.
    """

    def __init__(self, config: Config):
        self.config = config

    def add_background_noise(
        self,
        audio_path: str,
        output_path: str = None,
        noise_type: str = None,
        snr_db: float = None,
    ) -> str:
        """
        Add background noise to an audio file.

        Args:
            audio_path: Path to the clean audio file.
            output_path: Path to save the noisy audio.
            noise_type: Type of noise (white, traffic, cafe, rain).
            snr_db: Signal-to-noise ratio in dB.

        Returns:
            Path to the noisy audio file.
        """
        if output_path is None:
            base, ext = os.path.splitext(audio_path)
            output_path = f"{base}_noisy{ext}"

        noise_type = noise_type or self.config.noise_type
        snr_db = snr_db if snr_db is not None else self.config.noise_snr_db

        audio, sr = librosa.load(audio_path, sr=self.config.sample_rate)

        if noise_type == "white":
            noise = self._generate_white_noise(len(audio))
        else:
            noise = self._load_noise_sample(noise_type, len(audio))

        audio_power = np.mean(audio ** 2)
        noise_power = np.mean(noise ** 2)
        if noise_power > 0:
            target_noise_power = audio_power / (10 ** (snr_db / 10))
            scale = np.sqrt(target_noise_power / noise_power)
            noise = noise * scale

        noisy_audio = audio + noise
        noisy_audio = np.clip(noisy_audio, -1.0, 1.0)

        sf.write(output_path, noisy_audio, self.config.sample_rate)
        return output_path

    def _generate_white_noise(self, length: int) -> np.ndarray:
        return np.random.normal(0, 1, length)

    def _load_noise_sample(self, noise_type: str, target_length: int) -> np.ndarray:
        noise_dir = self.config.noise_dir
        noise_files = [f for f in os.listdir(noise_dir) if noise_type in f]

        if not noise_files:
            print(f"Warning: No noise sample for '{noise_type}'. Falling back to white noise.")
            return self._generate_white_noise(target_length)

        noise_file = os.path.join(noise_dir, random.choice(noise_files))
        noise, sr = librosa.load(noise_file, sr=self.config.sample_rate)

        if len(noise) < target_length:
            repeats = (target_length // len(noise)) + 1
            noise = np.tile(noise, repeats)
        return noise[:target_length]

    def add_reverb(self, audio_path: str, output_path: str = None) -> str:
        raise NotImplementedError("Reverb addition is not yet implemented.")
