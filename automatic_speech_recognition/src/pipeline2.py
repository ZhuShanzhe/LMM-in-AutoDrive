import os
import time
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from asr2.service import Qwen3ASRService
from asr2.config import Qwen3ASRConfig
from translation.service import Translation
from translation.config import ModelConfig
from .config import Qwen3PipelineConfig

logger = logging.getLogger(__name__)


class Qwen3ASRPipeline:
    """
    Pipeline using Qwen3-ASR for speech recognition.
    Supports optional noise reduction and dialect mapping.
    """

    def __init__(self, config: Optional[Qwen3PipelineConfig] = None, **kwargs):
        if config is None:
            config = Qwen3PipelineConfig(**kwargs)
        elif kwargs:
            for k, v in kwargs.items():
                if hasattr(config, k):
                    setattr(config, k, v)
        self.config = config

        # Initialize ASR service (Qwen3-ASR)
        asr_cfg = Qwen3ASRConfig(
            load_type=self.config.load_type,
            model_name=self.config.model_name,
            model_path=self.config.model_path,
            device=self.config.device,
            dtype=self.config.dtype,
            attn_implementation=self.config.attn_implementation,
            max_inference_batch_size=self.config.max_inference_batch_size,
            max_new_tokens=self.config.max_new_tokens,
            language=self.config.language,
            task=self.config.task,
            enable_enhancement=self.config.enable_enhancement,
            enhancement_method=self.config.enhancement_method,
            noise_floor=self.config.noise_floor,
            enable_dialect_mapping=self.config.enable_dialect_mapping,
            dialect_map=self.config.dialect_map,
        )
        self.asr_service = Qwen3ASRService(config=asr_cfg)

        os.makedirs(self.config.output_dir, exist_ok=True)

    def process(
        self,
        audio_path: str,
        output_json: Optional[str] = None,
        language: Optional[str] = None,
        enable_enhancement: Optional[bool] = None,
        enable_dialect_mapping: Optional[bool] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Transcribe a single audio file.

        Args:
            audio_path: Path to the input WAV file.
            output_json: Optional path to save the JSON result.
            language: Override language hint (e.g., "Chinese").
            enable_enhancement: Override noise reduction.
            enable_dialect_mapping: Override dialect mapping.
            **kwargs: Additional arguments passed to ASR service.

        Returns:
            Dictionary with transcription result and metadata.
        """
        start_time = time.perf_counter()
        result = self.asr_service.transcribe(
            audio=audio_path,
            output_json=output_json,
            language=language,
            enable_enhancement=enable_enhancement,
            enable_dialect_mapping=enable_dialect_mapping,
            **kwargs
        )
        total_time = time.perf_counter() - start_time

        # If result is a dict (with output_json) or string, ensure dict format
        if isinstance(result, str):
            result = {
                "audio_file": audio_path,
                "text": result,
                "processing_time_seconds": total_time,
            }
        else:
            result["total_time_seconds"] = total_time

        if output_json:
            self._save_json(result, output_json)

        return result

    @staticmethod
    def _save_json(data: Dict[str, Any], filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Result saved to: {filepath}")


class ASR2:
    """
    Pipeline that takes an audio file, performs ASR using Qwen3-ASR (Chinese),
    and optionally translates the result to English.

    Translation model is loaded lazily only when needed.
    """

    def __init__(
        self,
        config: Optional[Qwen3PipelineConfig] = None,

        asr_load_type: str = "local",
        asr_model_path: str = "./models/Qwen3-ASR-1.7B",
        asr_device: str = "cuda:0",
        asr_dtype: str = "bfloat16",
        asr_language: str = "Chinese",
        asr_task: str = "transcribe",
        asr_enable_enhancement: bool = False,
        asr_enhancement_method: str = "spectral",
        asr_noise_floor: float = 0.01,
        asr_enable_dialect_mapping: bool = False,
        asr_dialect_map: Optional[Dict[str, str]] = None,

        trans_model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        trans_load_type: str = "custom",
        trans_model_path: Optional[str] = None,
        trans_src_lang: str = "zho_Hans",
        trans_tgt_lang: str = "eng_Latn",
        trans_max_length: int = 512,
        trans_generation_max_length: Optional[int] = None,
        trans_num_beams: int = 4,
        trans_temperature: float = 0.0,
        trans_device: str = "cuda:0",

        output_dir: str = "data",
        raise_on_error: bool = False,
    ):
        """
        Initialize the ASR2 pipeline.

        Args:
            config: Optional full Qwen3PipelineConfig instance. If provided,
                    all other arguments are ignored.
            asr_*: Parameters for Qwen3-ASR recognition.
            trans_*: Parameters for translation (Chinese -> English) — stored for lazy init.
            output_dir: Directory to save JSON results.
            raise_on_error: If True, exceptions are raised; otherwise logged.
        """
        if config is not None:
            self.config = config
            self.output_dir = config.output_dir
            self.raise_on_error = config.raise_on_error

            asr_cfg = Qwen3ASRConfig(
                load_type=config.load_type,
                model_path=config.model_path,
                device=config.device,
                dtype=config.dtype,
                language=config.language,
                task=config.task,
                enable_enhancement=config.enable_enhancement,
                enhancement_method=config.enhancement_method,
                noise_floor=config.noise_floor,
                enable_dialect_mapping=config.enable_dialect_mapping,
                dialect_map=config.dialect_map or {},
            )
            self.asr = Qwen3ASRService(config=asr_cfg)

            self._trans_model_name = config.trans_model_name
            self._trans_load_type = config.trans_load_type
            self._trans_model_path = config.trans_model_path
            self._trans_src_lang = config.trans_src_lang
            self._trans_tgt_lang = config.trans_tgt_lang
            self._trans_max_length = config.trans_max_length
            self._trans_generation_max_length = config.trans_generation_max_length
            self._trans_num_beams = config.trans_num_beams
            self._trans_temperature = config.trans_temperature
            self._trans_device = config.trans_device
            self._trans_raise_on_error = config.raise_on_error

        else:
            asr_cfg = Qwen3ASRConfig(
                load_type=asr_load_type,
                model_path=asr_model_path,
                device=asr_device,
                dtype=asr_dtype,
                language=asr_language,
                task=asr_task,
                enable_enhancement=asr_enable_enhancement,
                enhancement_method=asr_enhancement_method,
                noise_floor=asr_noise_floor,
                enable_dialect_mapping=asr_enable_dialect_mapping,
                dialect_map=asr_dialect_map or {},
            )
            self.asr = Qwen3ASRService(config=asr_cfg)

            self._trans_model_name = trans_model_name
            self._trans_load_type = trans_load_type
            self._trans_model_path = trans_model_path
            self._trans_src_lang = trans_src_lang
            self._trans_tgt_lang = trans_tgt_lang
            self._trans_max_length = trans_max_length
            self._trans_generation_max_length = trans_generation_max_length
            self._trans_num_beams = trans_num_beams
            self._trans_temperature = trans_temperature
            self._trans_device = trans_device
            self._trans_raise_on_error = raise_on_error

            self.output_dir = output_dir
            self.raise_on_error = raise_on_error

        self.translator = None
        os.makedirs(self.output_dir, exist_ok=True)

    def _ensure_translator(self):
        """Lazy initialization of the translation service."""
        if self.translator is None:
            logger.info("Loading translation model (lazy initialization)...")
            trans_cfg = ModelConfig(
                model_name=self._trans_model_name,
                load_type=self._trans_load_type,
                model_path=self._trans_model_path,
                device=self._trans_device,
                max_length=self._trans_max_length,
            )
            self.translator = Translation(
                model_name=self._trans_model_name,
                load_type=self._trans_load_type,
                model_path=self._trans_model_path,
                src_lang=self._trans_src_lang,
                tgt_lang=self._trans_tgt_lang,
                max_length=self._trans_max_length,
                generation_max_length=self._trans_generation_max_length,
                num_beams=self._trans_num_beams,
                temperature=self._trans_temperature,
                raise_on_error=self._trans_raise_on_error,
            )
            logger.info("Translation model loaded successfully.")
        return self.translator

    def process(
        self,
        audio_path: str,
        output_json: Optional[str] = None,
        translate: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a single audio file: ASR (Chinese) and optional translation to English.

        Args:
            audio_path: Path to the input WAV file.
            output_json: Optional path to save the JSON result.
            translate: If True, translate the Chinese transcription to English.
            **kwargs: Additional arguments passed to the ASR service (e.g., language, enable_enhancement).

        Returns:
            Dictionary containing:
                - audio_file: input path
                - chinese_text: transcribed Chinese text
                - english_translation: translated English text (if translate=True)
                - asr_processing_time: time for ASR (seconds)
                - translation_time: time for translation (seconds)
                - total_time: total processing time
        """
        start_total = time.perf_counter()

        asr_start = time.perf_counter()
        asr_result = self.asr.transcribe(
            audio=audio_path,
            output_json=None,
            **kwargs
        )
        asr_time = time.perf_counter() - asr_start

        if isinstance(asr_result, dict):
            chinese_text = asr_result.get("text", "")
        else:
            chinese_text = asr_result

        if translate and chinese_text:
            translator = self._ensure_translator()
            trans_start = time.perf_counter()
            english_text = translator.translate_text(chinese_text)
            trans_time = time.perf_counter() - trans_start
        else:
            english_text = ""
            trans_time = 0.0

        total_time = time.perf_counter() - start_total

        result = {
            "audio_file": audio_path,
            "chinese_text": chinese_text,
            "english_translation": english_text,
            "asr_processing_time_seconds": asr_time,
            "translation_time_seconds": trans_time,
            "total_time_seconds": total_time,
        }

        if output_json is None:
            stem = Path(audio_path).stem
            output_json = os.path.join(self.output_dir, f"{stem}_result.json")
        self._save_json(result, output_json)

        return result

    @staticmethod
    def _save_json(data: Dict[str, Any], filepath: str) -> None:
        """Save result dictionary to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Result saved to: {filepath}")
