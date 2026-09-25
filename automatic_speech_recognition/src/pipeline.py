import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.asr import Qwen3ASRService
from src.asr.optimization import build_optimizer
from src.asr.utils import load_yaml, save_json, to_rel_path
from src.utils import DEFAULT_NUM_GPUS, resolve_device

logger = logging.getLogger("pipeline")

OUTPUT_LANGUAGES = ("chinese", "english", "both")
_ALIASES = {"zh": "chinese", "cn": "chinese", "en": "english", "eng": "english"}

def _load_optimization_config(config: Optional[Union[str, Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """Accept a yaml path, a dict, or None; return a config dict or None."""
    if config is None:
        return None
    if isinstance(config, dict):
        return config
    path = config if os.path.isabs(config) else os.path.join(str(ROOT), config)
    if not os.path.exists(path):
        logger.warning("optimization config not found: %s", path)
        return None
    return load_yaml(path)

class ASRPipeline:
    """External-facing facade combining ASR, optimization and translation."""

    def __init__(
        self,
        asr_model_path: str = "models/Qwen3-ASR-1.7B",
        asr_device: Optional[str] = None,
        asr_num_gpus: int = DEFAULT_NUM_GPUS,
        asr_dtype: str = "bfloat16",
        asr_attn_implementation: Optional[str] = None,
        language: Optional[str] = None,
        # --- optimization ---
        enable_optimization: bool = False,
        optimization_config: Optional[Union[str, Dict[str, Any]]] = None,
        # --- translation ---
        enable_translation: bool = False,
        translator_model_path: str = "models/Qwen3-1.7B",
        translator_device: Optional[str] = None,
        translator_num_gpus: int = DEFAULT_NUM_GPUS,
        translator_dtype: str = "bfloat16",
        # --- output ---
        output_language: str = "chinese",
        output_dir: str = "outputs",
        enable_guard: bool = False,
        raise_on_error: bool = False,
    ) -> None:
        self.output_language = self._normalize_language(output_language)
        self.output_dir = output_dir
        self.raise_on_error = raise_on_error
        self.enable_optimization = enable_optimization
        self.enable_translation = enable_translation

        frontend: Optional[Callable] = None
        dialect_normalizer: Optional[Callable] = None
        if enable_optimization:
            opt_cfg = _load_optimization_config(
                optimization_config or "configs/asr/optimization.yaml")
            if opt_cfg:
                frontend, dialect_normalizer = build_optimizer(opt_cfg).as_hooks()
            else:
                logger.warning("optimization enabled but no valid config; skipping.")

        guard = None
        if enable_guard:
            from src.asr.guard import ASROutputGuard
            guard = ASROutputGuard()

        self.asr = Qwen3ASRService(
            model_id_or_path=asr_model_path,
            device=asr_device or resolve_device(asr_num_gpus),
            dtype=asr_dtype,
            attn_implementation=asr_attn_implementation,
            language=language,
            frontend=frontend,
            dialect_normalizer=dialect_normalizer,
            guard=guard,
            output_dir=output_dir,
            raise_on_error=raise_on_error,
        )

        self.translator = None
        self._translator_args = dict(
            model_id_or_path=translator_model_path,
            device_map=translator_device or resolve_device(translator_num_gpus),
            dtype=translator_dtype,
            source_lang="Chinese",
            target_lang="English",
            output_dir=output_dir,
            raise_on_error=raise_on_error,
        )

    @staticmethod
    def _normalize_language(value: str) -> str:
        key = (value or "chinese").strip().lower()
        key = _ALIASES.get(key, key)
        if key not in OUTPUT_LANGUAGES:
            raise ValueError(
                f"unsupported output_language: {value}; "
                f"expected one of {OUTPUT_LANGUAGES} (aliases: zh, cn, en, eng)"
            )
        return key

    def _ensure_translator(self):
        if self.translator is None:
            from src.translator import Qwen3TranslatorService
            self.translator = Qwen3TranslatorService(**self._translator_args)
        return self.translator

    def process(
        self,
        audio: str,
        output_json: Optional[str] = None,
        output_language: Optional[str] = None,
        use_frontend: Optional[bool] = None,
        use_dialect: Optional[bool] = None,
        save_enhanced: Optional[str] = None,
        translate: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Run the full pipeline on one audio file.

        Args:
            audio: input wav path.
            output_json: optional path to write the JSON result.
            output_language: override default; "chinese" / "english" / "both".
            use_frontend / use_dialect: override optimization switches per call.
            save_enhanced: optional path to save the denoised audio.
            translate: override the translation switch per call.

        Returns:
            dict with keys: audio_file, text (Chinese), translation, output_text,
            language, timing and applied-flag fields.
        """
        asr_rec = self.asr.transcribe_file(
            audio,
            output_json=None,
            use_frontend=use_frontend,
            use_dialect=use_dialect,
            save_enhanced=save_enhanced,
        )
        chinese = asr_rec.get("text", "")

        want_translation = self.enable_translation if translate is None else translate
        english = ""
        if want_translation and chinese:
            english = self._ensure_translator().translate_zh_to_en(chinese)

        lang = self._normalize_language(output_language or self.output_language)
        output_text = self._select_output(chinese, english, lang)

        record: Dict[str, Any] = {
            "audio_file": to_rel_path(audio),
            "text": chinese,
            "translation": english,
            "output_text": output_text,
            "output_language": lang,
            "language": asr_rec.get("language"),
            "success": bool(output_text),
            "frontend_applied": asr_rec.get("frontend_applied", False),
            "dialect_normalized": asr_rec.get("dialect_normalized", False),
            "translation_applied": bool(want_translation and english),
            "processing_time_seconds": asr_rec.get("processing_time_seconds"),
        }
        for key in ("guard_safe", "guard_reasons", "slots"):
            if key in asr_rec:
                record[key] = asr_rec[key]
        if output_json:
            save_json(record, self._resolve_output(output_json))
        return record

    def process_batch(
        self,
        audio_paths: List[str],
        output_json: Optional[str] = None,
        output_language: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        records = [self.process(path, output_language=output_language, **kwargs) for path in audio_paths]
        if output_json:
            save_json({"count": len(records), "records": records}, self._resolve_output(output_json))
        return records

    def process_dir(
        self,
        input_dir: str,
        output_json: Optional[str] = None,
        output_language: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        exts = (".wav", ".flac", ".mp3")
        files = sorted(os.path.join(input_dir, n) for n in os.listdir(input_dir) if n.lower().endswith(exts))
        return self.process_batch(files, output_json=output_json, output_language=output_language, **kwargs)

    @classmethod
    def from_yaml(cls, path: str) -> "ASRPipeline":
        with open(path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        asr_cfg = cfg.get("asr", {}) or {}
        trans_cfg = cfg.get("translation", {}) or {}
        out_cfg = cfg.get("output", {}) or {}
        opt_cfg = cfg.get("optimization", {}) or {}
        guard_cfg = cfg.get("guard", {}) or {}

        return cls(
            asr_model_path=asr_cfg.get("model_id_or_path", "models/Qwen3-ASR-1.7B"),
            asr_device=asr_cfg.get("device"),
            asr_num_gpus=int(asr_cfg.get("num_gpus", 1)),
            asr_dtype=asr_cfg.get("dtype", "bfloat16"),
            asr_attn_implementation=asr_cfg.get("attn_implementation"),
            language=asr_cfg.get("language"),
            enable_optimization=bool(opt_cfg.get("enabled", False)),
            optimization_config=opt_cfg.get("config", "configs/asr/optimization.yaml"),
            enable_translation=bool(trans_cfg.get("enabled", False)),
            translator_model_path=trans_cfg.get("model_id_or_path", "models/Qwen3-1.7B"),
            translator_device=trans_cfg.get("device_map"),
            translator_num_gpus=int(trans_cfg.get("num_gpus", 1)),
            translator_dtype=trans_cfg.get("dtype", "bfloat16"),
            output_language=out_cfg.get("language", "chinese"),
            output_dir=out_cfg.get("dir", "outputs"),
            enable_guard=bool(guard_cfg.get("enabled", False)),
            raise_on_error=bool(cfg.get("raise_on_error", False)),
        )

    @staticmethod
    def _select_output(chinese: str, english: str, lang: str) -> str:
        if lang == "chinese":
            return chinese
        if lang == "english":
            return english or chinese
        if chinese and english:
            return f"{chinese}\t{english}"
        return chinese or english

    @staticmethod
    def _resolve_output(path: str) -> str:
        return path if os.path.isabs(path) else os.path.join(str(ROOT), path)
