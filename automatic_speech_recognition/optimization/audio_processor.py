import os
import time
import logging
import numpy as np
import librosa
import soundfile as sf
from typing import Optional, Dict, Any, List, Tuple

from .config import AugmentConfig
from .noise_generator import white_noise, pink_noise, vehicle_noise, load_noise_file
from .utils import ensure_dir, save_json, get_file_name

logger = logging.getLogger(__name__)


class AudioAugmenter:
    """Main class for adding noise to audio files."""

    def __init__(self, config: AugmentConfig):
        self.config = config
        ensure_dir(config.output_dir)

    def _compute_noise_scale(self, audio_power: float, noise_power: float, snr_db: float) -> float:
        """Compute scaling factor for noise to achieve target SNR."""
        if noise_power == 0:
            return 0.0
        target_noise_power = audio_power / (10 ** (snr_db / 10))
        return np.sqrt(target_noise_power / noise_power)

    def _generate_noise(self, length: int, sr: int, noise_type: Optional[str] = None, noise_file: Optional[str] = None) -> np.ndarray:
        noise_type = noise_type or self.config.noise_type
        noise_file = noise_file or self.config.noise_file

        if noise_type == "white":
            noise = white_noise(length)
        elif noise_type == "pink":
            noise = pink_noise(length)
        elif noise_type == "vehicle":
            noise = vehicle_noise(length, sr, noise_file)
        elif noise_type == "from_file":
            if noise_file is None:
                raise ValueError("noise_file must be specified for 'from_file' type.")
            noise = load_noise_file(noise_file, length, sr)
        else:
            raise ValueError(f"Unsupported noise_type: {noise_type}")
        return noise

    def add_noise_to_audio(self, audio: np.ndarray, sr: int, snr_db: Optional[float] = None, noise_type: Optional[str] = None, noise_file: Optional[str] = None) -> np.ndarray:
        """
        Add noise to an audio array.

        Args:
            audio: Input audio (float32, range [-1, 1]).
            sr: Sample rate.
            snr_db: Target SNR (overrides config).
            noise_type: Override noise type.
            noise_file: Override noise file.

        Returns:
            Noisy audio array (same shape as input).
        """
        snr = snr_db if snr_db is not None else self.config.snr_db
        length = len(audio)
        noise = self._generate_noise(length, sr, noise_type, noise_file)

        if len(noise) != length:
            if len(noise) > length:
                noise = noise[:length]
            else:
                pad_len = length - len(noise)
                noise = np.pad(noise, (0, pad_len), mode='constant', constant_values=0)

        audio_power = np.mean(audio ** 2)
        noise_power = np.mean(noise ** 2)
        scale = self._compute_noise_scale(audio_power, noise_power, snr)
        noisy = audio + noise * scale
        noisy = np.clip(noisy, -1.0, 1.0)
        return noisy.astype(np.float32)

    def process_file(self, input_path: str, output_path: Optional[str] = None,
                     snr_db: Optional[float] = None,
                     noise_type: Optional[str] = None,
                     noise_file: Optional[str] = None) -> str:
        """
        Load an audio file, add noise, and save the result.

        Returns:
            Path to the saved file.
        """
        audio, sr = librosa.load(input_path, sr=self.config.sample_rate, mono=True)

        noisy = self.add_noise_to_audio(audio, sr, snr_db, noise_type, noise_file)

        if output_path is None:
            base = get_file_name(input_path)
            out_dir = self.config.output_dir
            output_path = os.path.join(out_dir, f"{base}_noisy.wav")

        ensure_dir(os.path.dirname(output_path))
        sf.write(output_path, noisy, self.config.sample_rate)
        logger.info(f"Saved noisy audio to {output_path}")
        return output_path

    def process_dataset(self, dataset_json: str,
                        audio_key: str = "audio_file",
                        output_json_clean: Optional[str] = None,
                        output_json_noisy: Optional[str] = None,
                        snr_db: Optional[float] = None,
                        noise_type: Optional[str] = None,
                        noise_file: Optional[str] = None) -> Tuple[List[Dict], List[Dict]]:
        """
        Process a dataset JSON file.

        Expected format:
        [
            {"index": 1, "audio_file": "path/to/audio.wav", ...},
            ...
        ]

        Returns:
            (clean_mapping, noisy_mapping) – lists of dicts with updated audio_file paths.
        """
        from .utils import load_json, save_json

        data = load_json(dataset_json)
        clean_mapping = []
        noisy_mapping = []

        for item in data:
            audio_path = item.get(audio_key)
            if not audio_path or not os.path.exists(audio_path):
                logger.warning(f"Skipping {audio_path}: file not found.")
                continue

            noisy_path = self.process_file(
                audio_path,
                snr_db=snr_db,
                noise_type=noise_type,
                noise_file=noise_file
            )

            clean_item = item.copy()
            noisy_item = item.copy()
            clean_item[audio_key] = audio_path
            noisy_item[audio_key] = noisy_path

            clean_mapping.append(clean_item)
            noisy_mapping.append(noisy_item)

        # Save mapping if requested
        if output_json_clean:
            save_json(clean_mapping, output_json_clean)
        if output_json_noisy:
            save_json(noisy_mapping, output_json_noisy)

        return clean_mapping, noisy_mapping

    def process_directory(self, input_dir: str, output_dir: Optional[str] = None,
                          snr_db: Optional[float] = None,
                          noise_type: Optional[str] = None,
                          noise_file: Optional[str] = None) -> List[str]:
        """Process all WAV files in a directory."""
        from .utils import list_wav_files
        wav_files = list_wav_files(input_dir)
        results = []
        for wav in wav_files:
            out = self.process_file(wav, snr_db=snr_db, noise_type=noise_type, noise_file=noise_file)
            results.append(out)
        return results
