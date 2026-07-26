import os
import time
import logging
from typing import List, Optional, Union
from math import ceil
import torch
import torchaudio
import numpy as np
from tts import ChatTTSConfig
from tts import ChatTTSModel
from .noise_utils import (
    generate_white_noise, generate_pink_noise,
    generate_brown_noise, load_noise_from_file
)

logger = logging.getLogger(__name__)


class ChatTTSService:
    """
    TTS service with batch support and optional noise addition.
    Outputs WAV files named 'command_{id}.wav'.
    """

    def __init__(self, config: Optional[ChatTTSConfig] = None, **kwargs):
        if config is None:
            config = ChatTTSConfig.from_dict(kwargs)
        elif kwargs:
            config = config.update(**kwargs)
        self.config = config
        self.model = ChatTTSModel(config)
        self.sample_rate = config.sample_rate
        self.last_synthesis_time = 0.0
        self.last_batch_time = 0.0
        self._counter = 0
        os.makedirs(self.config.output_dir, exist_ok=True)

    def apply_noise(
        self,
        audio: np.ndarray,
        noise_type: Optional[str] = None,
        noise_level: Optional[float] = None,
        noise_file: Optional[str] = None,
    ) -> np.ndarray:
        """Apply noise to a single audio array."""
        n_type = noise_type or self.config.noise_type
        n_level = noise_level if noise_level is not None else self.config.noise_level
        n_file = noise_file or self.config.noise_file

        length = len(audio)
        sr = self.sample_rate

        if n_type == 'white':
            noise = generate_white_noise(length, n_level)
        elif n_type == 'pink':
            noise = generate_pink_noise(length, n_level)
        elif n_type == 'brown':
            noise = generate_brown_noise(length, n_level)
        elif n_type == 'from_file':
            if n_file is None:
                raise ValueError("noise_file must be provided for type 'from_file'")
            noise = load_noise_from_file(n_file, length, sr)
            max_val = np.max(np.abs(noise))
            if max_val > 0:
                noise = noise / max_val * n_level
        else:
            raise ValueError(f"Unsupported noise_type: {n_type}")

        noisy = audio + noise
        noisy = np.clip(noisy, -1.0, 1.0)
        return noisy.astype(np.float32)

    def synthesize(
        self,
        text: Union[str, List[str]],
        output_dir: Optional[str] = None,
        batch_size: Optional[int] = None,
        add_noise: Optional[bool] = None,
        noise_type: Optional[str] = None,
        noise_level: Optional[float] = None,
        noise_file: Optional[str] = None,
        **infer_kwargs
    ) -> Union[str, List[str]]:
        """
        Synthesize speech from text(s) and optionally add noise.

        Args:
            text: Single string or list of strings.
            output_dir: Output directory (overrides config).
            batch_size: Override default batch size.
            add_noise: Enable noise addition (if None, use config.noise_enabled).
            noise_type: Override noise type.
            noise_level: Override noise level.
            noise_file: Override noise file path.
            **infer_kwargs: Additional arguments passed to chat.infer().

        Returns:
            Single file path or list of file paths.
        """
        if isinstance(text, str):
            return self._synthesize_single(
                text, output_dir, add_noise, noise_type, noise_level, noise_file, **infer_kwargs
            )
        elif isinstance(text, list):
            return self._synthesize_batch(
                text, output_dir, batch_size, add_noise, noise_type, noise_level, noise_file, **infer_kwargs
            )
        else:
            raise TypeError(f"Expected str or list, got {type(text)}")

    def _synthesize_single(
        self,
        text: str,
        output_dir: Optional[str] = None,
        add_noise: Optional[bool] = None,
        noise_type: Optional[str] = None,
        noise_level: Optional[float] = None,
        noise_file: Optional[str] = None,
        **infer_kwargs
    ) -> str:
        if not text or not text.strip():
            raise ValueError("Text cannot be empty.")
        output_dir = output_dir or self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        self._counter += 1
        index = self._counter

        start_time = time.perf_counter()
        wavs = self.model.synthesize([text], **infer_kwargs)
        elapsed = time.perf_counter() - start_time
        self.last_synthesis_time = elapsed

        if not wavs:
            raise RuntimeError("No output from synthesis.")

        audio = wavs[0]

        # Apply noise if enabled
        use_noise = add_noise if add_noise is not None else self.config.noise_enabled
        if use_noise:
            audio = self.apply_noise(audio, noise_type, noise_level, noise_file)

        filename = f"command_{index:04d}.wav"
        filepath = os.path.join(output_dir, filename)
        self._save_audio(audio, filepath)
        logger.info(f"Synthesized in {elapsed:.3f}s: '{text[:30]}...' -> {filepath}")
        return filepath

    def _synthesize_batch(
        self,
        texts: List[str],
        output_dir: Optional[str] = None,
        batch_size: Optional[int] = None,
        add_noise: Optional[bool] = None,
        noise_type: Optional[str] = None,
        noise_level: Optional[float] = None,
        noise_file: Optional[str] = None,
        **infer_kwargs
    ) -> List[str]:
        if not texts:
            return []
        output_dir = output_dir or self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        batch_size = batch_size or self.config.batch_size
        total = len(texts)
        num_batches = ceil(total / batch_size)

        results = []
        total_start = time.perf_counter()

        # Determine noise settings once for all batches
        use_noise = add_noise if add_noise is not None else self.config.noise_enabled
        n_type = noise_type or self.config.noise_type
        n_level = noise_level if noise_level is not None else self.config.noise_level
        n_file = noise_file or self.config.noise_file

        for batch_idx in range(num_batches):
            print(f"\n[{batch_idx}/{num_batches}] Processing...")
            start = batch_idx * batch_size
            end = min(start + batch_size, total)
            batch_texts = texts[start:end]

            logger.info(f"Processing batch {batch_idx+1}/{num_batches}: {start}~{end-1}")
            batch_start = time.perf_counter()
            wavs = self.model.synthesize(batch_texts, **infer_kwargs)
            batch_elapsed = time.perf_counter() - batch_start

            for i, audio in enumerate(wavs):
                if use_noise:
                    audio = self.apply_noise(audio, n_type, n_level, n_file)
                global_idx = start + i + 1
                filename = f"command_{global_idx:04d}.wav"
                filepath = os.path.join(output_dir, filename)
                self._save_audio(audio, filepath)
                results.append(filepath)

            logger.info(f"Batch {batch_idx+1} completed in {batch_elapsed:.3f}s")

        total_elapsed = time.perf_counter() - total_start
        self.last_batch_time = total_elapsed
        logger.info(f"Batch synthesis finished in {total_elapsed:.3f}s, {total} files generated")
        return results

    def _save_audio(self, audio: np.ndarray, filepath: str) -> None:
        """Save audio with torchaudio, handling different versions."""
        try:
            tensor = torch.from_numpy(audio)
            try:
                torchaudio.save(filepath, tensor.unsqueeze(0), self.sample_rate)
            except Exception:
                torchaudio.save(filepath, tensor, self.sample_rate)
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            raise

    def reset_counter(self, start: int = 0) -> None:
        self._counter = start

    def get_counter(self) -> int:
        return self._counter
