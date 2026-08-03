import os
import time
import json
import logging
from typing import Optional, Dict, Any
from pathlib import Path

from asr.service import FunASRService
from asr.config import FunASRConfig
from .config import ASRConfig
from translation.service import Translation
from translation.config import ModelConfig

logger = logging.getLogger(__name__)


class ASR:
    """
    Pipeline that takes an audio file, performs ASR (Chinese), and optionally translates it to English.
    Translation model is loaded lazily only when needed.
    """

    def __init__(
        self,
        config: Optional[ASRConfig] = None,
        asr_mode: str = "single",
        asr_device: str = "cuda:0",
        asr_model: Optional[str] = None,
        asr_vad_model: Optional[str] = None,
        asr_punc_model: Optional[str] = None,
        asr_spk_model: Optional[str] = None,
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
        output_dir: str = "outputs",
        raise_on_error: bool = False,
    ):
        if config is not None:
            self.config = config
        else:
            self.config = ASRConfig(
                asr_mode=asr_mode,
                asr_device=asr_device,
                asr_model=asr_model,
                asr_vad_model=asr_vad_model,
                asr_punc_model=asr_punc_model,
                asr_spk_model=asr_spk_model,
                trans_model_name=trans_model_name,
                trans_load_type=trans_load_type,
                trans_model_path=trans_model_path,
                trans_src_lang=trans_src_lang,
                trans_tgt_lang=trans_tgt_lang,
                trans_max_length=trans_max_length,
                trans_generation_max_length=trans_generation_max_length,
                trans_num_beams=trans_num_beams,
                trans_temperature=trans_temperature,
                trans_device=trans_device,
                output_dir=output_dir,
                raise_on_error=raise_on_error,
            )

        asr_cfg = FunASRConfig(
            mode=self.config.asr_mode,
            device=self.config.asr_device,
            model=self.config.asr_model,
            vad_model=self.config.asr_vad_model,
            punc_model=self.config.asr_punc_model,
            spk_model=self.config.asr_spk_model,
        )
        self.asr = FunASRService(config=asr_cfg)

        self.translator = None

        os.makedirs(self.config.output_dir, exist_ok=True)

    def _ensure_translator(self):
        """
        Lazy initialization of the translation service.
        """
        if self.translator is None:
            logger.info("Loading translation model (lazy initialization)...")
            trans_cfg = ModelConfig(
                model_name=self.config.trans_model_name,
                load_type=self.config.trans_load_type,
                model_path=self.config.trans_model_path,
                device=self.config.trans_device,
                max_length=self.config.trans_max_length,
            )
            self.translator = Translation(
                model_name=self.config.trans_model_name,
                load_type=self.config.trans_load_type,
                model_path=self.config.trans_model_path,
                src_lang=self.config.trans_src_lang,
                tgt_lang=self.config.trans_tgt_lang,
                max_length=self.config.trans_max_length,
                generation_max_length=self.config.trans_generation_max_length,
                num_beams=self.config.trans_num_beams,
                temperature=self.config.trans_temperature,
                raise_on_error=self.config.raise_on_error,
            )
            logger.info("Translation model loaded successfully.")
        return self.translator

    def process(
        self,
        audio_path: str,
        output_json: Optional[str] = None,
        translate: bool = True
    ) -> Dict[str, Any]:
        """
        Process a single audio file: ASR and optional translation.

        Args:
            audio_path: Path to the input WAV file.
            output_json: Optional path to save the JSON result.
            translate: If True, translate Chinese text to English.

        Returns:
            Dictionary containing transcription and optional translation.
        """
        start_total = time.perf_counter()

        asr_start = time.perf_counter()
        asr_result = self.asr.transcribe(audio_path, print_result=False)
        asr_text = asr_result.get("text", "")
        asr_time = time.perf_counter() - asr_start

        if translate and asr_text:
            translator = self._ensure_translator()
            trans_start = time.perf_counter()
            english_text = translator.translate_text(asr_text)
            trans_time = time.perf_counter() - trans_start
        else:
            english_text = ""
            trans_time = 0.0

        total_time = time.perf_counter() - start_total

        result = {
            "audio_file": audio_path,
            "chinese_text": asr_text,
            "english_translation": english_text,
            "asr_processing_time_seconds": asr_time,
            "translation_time_seconds": trans_time,
            "total_time_seconds": total_time,
        }

        if output_json is None:
            stem = Path(audio_path).stem
            output_json = os.path.join(self.config.output_dir, f"{stem}_result.json")
        self._save_json(result, output_json)

        return result

    @staticmethod
    def _save_json(data: Dict[str, Any], filepath: str) -> None:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Result saved to: {filepath}")
