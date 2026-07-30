import os
import logging
import tempfile
from typing import Optional, Tuple, Union

import numpy as np
import soundfile as sf
import librosa
import torch

from df import enhance, init_df
from .config import DenoiserConfig

logger = logging.getLogger(__name__)


class DeepFilterNetDenoiser:
    """
    DeepFilterNet speech enhancer using official `df` module.
    """

    def __init__(self, config: Optional[DenoiserConfig] = None):
        self.config = config or DenoiserConfig()
        self._model = None
        self._df_state = None
        self._model_loaded = False

    def _load_model(self):
        """Lazy-load the DeepFilterNet model."""
        if not self._model_loaded:
            model_name = self.config.model_name
            logger.info(f"Loading DeepFilterNet model: {model_name}")

            self._model, self._df_state, _ = init_df(model_base_dir=model_name)
            self._model_loaded = True
            logger.info("Model loaded successfully.")
        return self._model, self._df_state

    def denoise_audio(
        self,
        audio: np.ndarray,
        sr: int,
        output_sr: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Denoise an audio array.

        Args:
            audio: Input audio (float32, range [-1, 1]).
            sr: Sample rate of input.
            output_sr: Desired output sample rate. Defaults to config.output_sr.

        Returns:
            (denoised_audio, output_sample_rate)
        """
        if sr != 48000:
            logger.debug(f"Resampling from {sr}Hz to 48000Hz for DeepFilterNet")
            audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
            sr = 48000

        model, df_state = self._load_model()

        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)

        logger.debug("Enhancing audio...")
        enhanced_tensor = enhance(model, df_state, audio_tensor)
        logger.debug("Enhancement completed.")

        enhanced = enhanced_tensor.squeeze(0).cpu().numpy()

        out_sr = output_sr if output_sr is not None else self.config.output_sr
        if out_sr != 48000:
            enhanced = librosa.resample(enhanced, orig_sr=48000, target_sr=out_sr)
            final_sr = out_sr
        else:
            final_sr = 48000

        return enhanced.astype(np.float32), final_sr

    def denoise_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        output_sr: Optional[int] = None,
    ) -> str:
        """
        Denoise a WAV file and save the result.

        Args:
            input_path: Path to input audio file.
            output_path: Optional output path.
            output_sr: Output sample rate.

        Returns:
            Path to denoised file.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        audio, sr = sf.read(input_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        enhanced, final_sr = self.denoise_audio(audio, sr, output_sr)

        if output_path is None:
            base, ext = os.path.splitext(input_path)
            output_path = f"{base}_denoised{ext}"

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        sf.write(output_path, enhanced, final_sr)
        logger.info(f"Denoised file saved to: {output_path}")
        return output_path

    def denoise_batch(
        self,
        input_files: list,
        output_dir: str,
        output_sr: Optional[int] = None,
        suffix: str = "_denoised",
    ) -> list:
        """
        Batch denoise multiple files.

        Args:
            input_files: List of input file paths.
            output_dir: Directory to save denoised files.
            output_sr: Output sample rate.
            suffix: Suffix to append to filename before extension.

        Returns:
            List of output file paths.
        """
        os.makedirs(output_dir, exist_ok=True)
        outputs = []
        for f in input_files:
            base = os.path.splitext(os.path.basename(f))[0]
            out_path = os.path.join(output_dir, f"{base}{suffix}.wav")
            out_path = self.denoise_file(f, out_path, output_sr)
            outputs.append(out_path)
        return outputs
