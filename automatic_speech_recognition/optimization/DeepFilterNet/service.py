import os
import time
import json
import logging
from typing import Optional, Dict, Any, List, Union

from .config import DenoiserConfig
from .denoiser import DeepFilterNetDenoiser

logger = logging.getLogger(__name__)


class DenoiseService:
    """
    High-level service for DeepFilterNet noise suppression.
    Supports single file, batch processing, and JSON output with timing.
    """

    def __init__(self, config: Optional[DenoiserConfig] = None, **kwargs):
        if config is None:
            config = DenoiserConfig(**kwargs)
        elif kwargs:
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        self.config = config
        self.denoiser = DeepFilterNetDenoiser(config)
        self.last_processing_time = 0.0
        self.last_batch_time = 0.0

        os.makedirs(self.config.output_dir, exist_ok=True)

    def denoise(
        self,
        audio: Union[str, List[str]],
        output_path: Optional[str] = None,
        output_json: Optional[str] = None,
        output_sr: Optional[int] = None,
    ) -> Union[str, List[str], Dict[str, Any]]:
        """
        Unified entry point for denoising.

        Args:
            audio: Single audio path or list of audio paths.
            output_path: Output path (for single file only).
            output_json: Optional JSON output file.
            output_sr: Target sample rate.

        Returns:
            Single file path or list of file paths, or result dict.
        """
        if isinstance(audio, str):
            return self._denoise_single(audio, output_path, output_json, output_sr)
        elif isinstance(audio, list):
            return self._denoise_batch(audio, output_json, output_sr)
        else:
            raise TypeError(f"Expected str or list, got {type(audio)}")

    def _denoise_single(
        self,
        audio_path: str,
        output_path: Optional[str] = None,
        output_json: Optional[str] = None,
        output_sr: Optional[int] = None,
    ) -> Union[str, Dict[str, Any]]:
        """Denoise a single audio file."""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        start_time = time.perf_counter()
        out_path = self.denoiser.denoise_file(audio_path, output_path, output_sr)
        elapsed = time.perf_counter() - start_time
        self.last_processing_time = elapsed

        result = {
            "input_file": audio_path,
            "output_file": out_path,
            "processing_time_seconds": elapsed,
            "sample_rate": output_sr or self.config.output_sr,
        }

        if output_json:
            self._save_json(result, output_json)

        return result if output_json else out_path

    def _denoise_batch(
        self,
        audio_paths: List[str],
        output_json: Optional[str] = None,
        output_sr: Optional[int] = None,
    ) -> List[str]:
        """Denoise multiple audio files."""
        if not audio_paths:
            return []

        results = []
        total_start = time.perf_counter()
        total = len(audio_paths)

        for i, path in enumerate(audio_paths, 1):
            if not os.path.exists(path):
                logger.warning(f"File not found: {path}, skipping.")
                results.append("")
                continue

            try:
                start = time.perf_counter()
                out_path = self.denoiser.denoise_file(path, output_sr=output_sr)
                elapsed = time.perf_counter() - start
                results.append(out_path)
                logger.info(f"Denoised {i}/{total} in {elapsed:.3f}s: {path}")
            except Exception as e:
                logger.error(f"Failed to denoise {path}: {e}")
                results.append("")

        total_elapsed = time.perf_counter() - total_start
        self.last_batch_time = total_elapsed
        logger.info(f"Batch denoising finished in {total_elapsed:.3f}s")

        if output_json:
            summary = {
                "total_files": total,
                "processed_files": len([r for r in results if r]),
                "total_time_seconds": total_elapsed,
                "results": results,
            }
            self._save_json(summary, output_json)

        return results

    @staticmethod
    def _save_json(data: Dict[str, Any], filepath: str) -> None:
        """Save data to JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Result saved to: {filepath}")
