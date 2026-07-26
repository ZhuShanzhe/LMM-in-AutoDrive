import os
import time
import json
import logging
import tempfile
import soundfile as sf
import numpy as np
import librosa
from typing import List, Optional, Union, Dict, Any

from .config import Qwen3ASRConfig
from .asr_model import Qwen3ASRModel
from .utils import spectral_subtraction, wiener_filter, apply_dialect_mapping, DEFAULT_DIALECT_MAP

logger = logging.getLogger(__name__)


class Qwen3ASRService:
    def __init__(self, config: Optional[Qwen3ASRConfig] = None, **kwargs):
        if config is None:
            config = Qwen3ASRConfig(**kwargs)
        elif kwargs:
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        self.config = config
        self.model = Qwen3ASRModel(config)
        self.last_transcription_time = 0.0
        self.last_batch_time = 0.0

        if self.config.enable_dialect_mapping and not self.config.dialect_map:
            self.config.dialect_map = DEFAULT_DIALECT_MAP

    def transcribe(
        self,
        audio: Union[str, List[str]],
        output_json: Optional[str] = None,
        language: Optional[str] = None,
        task: Optional[str] = None,
        enable_enhancement: Optional[bool] = None,
        enable_dialect_mapping: Optional[bool] = None,
        **kwargs
    ) -> Union[str, List[str], Dict[str, Any]]:
        if isinstance(audio, str):
            return self._transcribe_single(
                audio, output_json, language, task,
                enable_enhancement, enable_dialect_mapping, **kwargs
            )
        elif isinstance(audio, list):
            return self._transcribe_batch(
                audio, language, task,
                enable_enhancement, enable_dialect_mapping, **kwargs
            )
        else:
            raise TypeError(f"Expected str or list, got {type(audio)}")

    def _apply_enhancement(self, audio: np.ndarray, sr: int) -> np.ndarray:
        method = self.config.enhancement_method
        floor = self.config.noise_floor
        if method == "spectral":
            return spectral_subtraction(audio, sr, noise_floor=floor)
        elif method == "wiener":
            return wiener_filter(audio, sr, noise_std=floor)
        else:
            logger.warning(f"Unknown enhancement method: {method}, skipping.")
            return audio

    def _transcribe_single(
        self,
        audio_path: str,
        output_json: Optional[str] = None,
        language: Optional[str] = None,
        task: Optional[str] = None,
        enable_enhancement: Optional[bool] = None,
        enable_dialect_mapping: Optional[bool] = None,
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        use_enhancement = self.config.enable_enhancement if enable_enhancement is None else enable_enhancement
        use_mapping = self.config.enable_dialect_mapping if enable_dialect_mapping is None else enable_dialect_mapping

        start_time = time.perf_counter()

        audio, sr = librosa.load(audio_path, sr=16000, mono=True)

        temp_file_path = None
        if use_enhancement:
            audio = self._apply_enhancement(audio, sr)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                temp_file_path = tmp_file.name
            sf.write(temp_file_path, audio, sr)
            audio_input = temp_file_path
        else:
            audio_input = audio_path

        try:
            results = self.model.transcribe(audio_input, language, **kwargs)
            if not results:
                raise RuntimeError("ASR returned no results.")
            first = results[0]
            if hasattr(first, 'text'):
                text = first.text
                detected_lang = getattr(first, 'language', None)
            else:
                text = first.get("text", "")
                detected_lang = first.get("language", None)
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

        elapsed = time.perf_counter() - start_time
        self.last_transcription_time = elapsed

        if use_mapping and text:
            text = apply_dialect_mapping(text, self.config.dialect_map)

        result = {
            "audio_file": audio_path,
            "text": text,
            "language": detected_lang or language or self.config.language,
            "processing_time_seconds": elapsed,
            "enhancement_applied": use_enhancement,
            "dialect_mapping_applied": use_mapping,
        }

        if output_json:
            self._save_json(result, output_json)
        return result if output_json else text

    def _transcribe_batch(
        self,
        audio_paths: List[str],
        language: Optional[str] = None,
        task: Optional[str] = None,
        enable_enhancement: Optional[bool] = None,
        enable_dialect_mapping: Optional[bool] = None,
        **kwargs
    ) -> List[str]:
        if not audio_paths:
            return []

        results = []
        total_start = time.perf_counter()
        for i, path in enumerate(audio_paths, 1):
            if not os.path.exists(path):
                logger.warning(f"File not found: {path}, skipping.")
                results.append("")
                continue
            try:
                text = self._transcribe_single(
                    path,
                    output_json=None,
                    language=language,
                    task=task,
                    enable_enhancement=enable_enhancement,
                    enable_dialect_mapping=enable_dialect_mapping,
                    **kwargs
                )
                results.append(text)
            except Exception as e:
                logger.error(f"Failed to transcribe {path}: {e}")
                results.append("")

        total_elapsed = time.perf_counter() - total_start
        self.last_batch_time = total_elapsed
        logger.info(f"Batch transcription finished in {total_elapsed:.3f}s")
        return results

    @staticmethod
    def _save_json(data: Dict[str, Any], filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Result saved to: {filepath}")
