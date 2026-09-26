import logging
import os
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from src.utils import DEFAULT_NUM_GPUS, resolve_device, resolve_num_gpus

from .utils import load_audio, load_yaml, normalize_text, save_audio, save_json, to_rel_path

logger = logging.getLogger(__name__)

_LOADED: Dict[Tuple[str, str], Any] = {}

class Qwen3ASRService:
    def __init__(
        self,
        model_id_or_path: str = "models/Qwen3-ASR-1.7B",
        device: Optional[str] = None,
        num_gpus: int = DEFAULT_NUM_GPUS,
        rank: int = 0,
        dtype: str = "bfloat16",
        attn_implementation: Optional[str] = None,
        max_inference_batch_size: int = 32,
        max_new_tokens: int = 256,
        language: Optional[str] = None,
        target_sample_rate: int = 16000,
        frontend: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
        dialect_normalizer: Optional[Callable[[str], str]] = None,
        guard: Optional[Any] = None,
        output_dir: str = "outputs",
        raise_on_error: bool = False,
        fallback_on_failure: bool = True,
    ) -> None:
        self.model_id_or_path = model_id_or_path
        self.num_gpus = resolve_num_gpus(num_gpus)
        self.rank = int(rank)
        self.device = device or resolve_device(self.num_gpus, self.rank)
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.max_inference_batch_size = max_inference_batch_size
        self.max_new_tokens = max_new_tokens
        self.language = language
        self.target_sample_rate = target_sample_rate
        self.frontend = frontend
        self.dialect_normalizer = dialect_normalizer
        self.guard = guard
        self.output_dir = output_dir
        self.raise_on_error = raise_on_error
        self.fallback_on_failure = fallback_on_failure
        self.model = self._load_model()

    @classmethod
    def from_yaml(
        cls, 
        path: str, 
        frontend: Optional[Callable] = None,
        dialect_normalizer: Optional[Callable] = None,
        guard: Optional[Any] = None
    ) -> "Qwen3ASRService":
        cfg = load_yaml(path)
        params = {
            "model_id_or_path", "device", "num_gpus", "rank", "dtype", "attn_implementation",
            "max_inference_batch_size", "max_new_tokens", "language",
            "target_sample_rate", "output_dir", "raise_on_error", "fallback_on_failure",
        }
        known = {k: v for k, v in cfg.items() if k in params}
        unknown = [k for k in cfg if k not in params]
        if unknown:
            logger.warning("ignoring unknown ASR config keys: %s", unknown)
        return cls(frontend=frontend, dialect_normalizer=dialect_normalizer, guard=guard, **known)

    def _load_model(self):
        cache_key = (self.model_id_or_path, self.device)
        if cache_key in _LOADED:
            return _LOADED[cache_key]

        import torch
        from qwen_asr import Qwen3ASRModel

        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(self.dtype, torch.bfloat16)
        kwargs: Dict[str, Any] = {
            "dtype": torch_dtype,
            "device_map": self.device,
            "max_inference_batch_size": self.max_inference_batch_size,
            "max_new_tokens": self.max_new_tokens,
            "trust_remote_code": True,
        }
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation
        if os.path.isdir(self.model_id_or_path):
            kwargs["local_files_only"] = True

        logger.info("Loading Qwen3-ASR from %s on %s ...", self.model_id_or_path, self.device)
        model = Qwen3ASRModel.from_pretrained(self.model_id_or_path, **kwargs)
        _LOADED[cache_key] = model
        logger.info("Qwen3-ASR ready.")
        return model

    def _forward(self, audio: Union[str, Tuple[np.ndarray, int]], language: Optional[str]) -> Dict[str, Any]:
        results = self.model.transcribe(audio=audio, language=language or self.language)
        if not results:
            raise RuntimeError("ASR returned no results.")
        first = results[0]
        if hasattr(first, "text"):
            return {"text": first.text, "language": getattr(first, "language", None)}
        return {"text": first.get("text", ""), "language": first.get("language", None)}

    def _transcribe_once(
        self,
        audio_path: str,
        language: Optional[str],
        do_dialect: bool,
        use_frontend: bool,
        save_enhanced: Optional[str] = None
    ) -> Tuple[Dict[str, Any], str, bool]:
        tmp_path: Optional[str] = None
        try:
            if use_frontend and self.frontend is not None:
                audio = load_audio(audio_path, self.target_sample_rate)
                audio = self.frontend(audio, self.target_sample_rate)
                if save_enhanced:
                    save_audio(audio, self.target_sample_rate, save_enhanced)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = tmp.name
                save_audio(audio, self.target_sample_rate, tmp_path)
                audio_input: Union[str, Tuple[np.ndarray, int]] = tmp_path
                applied = True
            else:
                audio_input = audio_path
                applied = False
            raw = self._forward(audio_input, language)
            text = normalize_text(raw["text"])
            if do_dialect and self.dialect_normalizer is not None:
                text = self.dialect_normalizer(text)
            return raw, text, applied
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def transcribe_file(
        self, 
        audio_path: str,
        output_json: Optional[str] = None,
        language: Optional[str] = None,
        use_frontend: Optional[bool] = None,
        use_dialect: Optional[bool] = None,
        save_enhanced: Optional[str] = None,
        fallback_on_failure: Optional[bool] = None
    ) -> Dict[str, Any]:
        do_frontend = (self.frontend is not None) if use_frontend is None else use_frontend
        do_dialect = (self.dialect_normalizer is not None) if use_dialect is None else use_dialect

        do_fallback = self.fallback_on_failure if fallback_on_failure is None else fallback_on_failure

        start = time.perf_counter()
        raw: Dict[str, Any] = {"language": None}
        text = ""
        frontend_applied = False
        fallback = False
        error: Optional[str] = None

        if not os.path.exists(audio_path):
            message = "audio file not found: " + audio_path
            if self.raise_on_error:
                raise FileNotFoundError(message)
            logger.error(message)
            error = message

        if not error and do_frontend and self.frontend is not None:
            try:
                raw, text, frontend_applied = self._transcribe_once(audio_path, language, do_dialect, True, save_enhanced)
            except Exception as exc:  # noqa: BLE001
                if self.raise_on_error:
                    raise
                logger.error("optimized transcription failed for %s: %s", audio_path, exc)
                error = "frontend: " + str(exc)
                frontend_applied = False
                fallback = do_fallback
            else:
                if not text and do_fallback:
                    logger.warning(
                        "optimized transcription is empty for %s; retrying without optimization",
                        audio_path,
                    )
                    fallback = True

            if fallback:
                try:
                    raw, text, frontend_applied = self._transcribe_once(audio_path, language, do_dialect, False)
                    error = None
                except Exception as exc:  # noqa: BLE001
                    if self.raise_on_error:
                        raise
                    logger.error("transcription failed for %s: %s", audio_path, exc)
                    raw, text, frontend_applied = {"language": None}, "", False
                    error = str(exc)
        elif not error:
            try:
                raw, text, frontend_applied = self._transcribe_once(audio_path, language, do_dialect, False)
            except Exception as exc:  # noqa: BLE001
                if self.raise_on_error:
                    raise
                logger.error("transcription failed for %s: %s", audio_path, exc)
                raw, text, frontend_applied = {"language": None}, "", False
                error = str(exc)

        record = {
            "audio_file": to_rel_path(audio_path),
            "text": text,
            "language": raw.get("language") or language or self.language,
            "success": bool(text),
            "processing_time_seconds": round(time.perf_counter() - start, 4),
            "frontend_applied": frontend_applied,
            "dialect_normalized": bool(do_dialect and self.dialect_normalizer is not None),
            "optimization_fallback": fallback,
        }
        if error:
            record["error"] = error
        if self.guard is not None:
            verdict = self.guard.check(text, record["language"])
            record["guard_safe"] = verdict["safe"]
            record["guard_reasons"] = verdict["reasons"]
            record["slots"] = verdict["slots"]
        if output_json:
            save_json(record, output_json)
        return record

    def transcribe(self, audio: Union[str, List[str]], output_json: Optional[str] = None, **kwargs):
        if isinstance(audio, str):
            return self.transcribe_file(audio, output_json=output_json, **kwargs)
        if isinstance(audio, list):
            return self.transcribe_batch(audio, output_json=output_json, **kwargs)
        raise TypeError("audio must be str or List[str]")

    def transcribe_batch(self, audio_paths: List[str], output_json: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for path in audio_paths:
            if not os.path.exists(path):
                logger.warning("file not found, skipping: %s", path)
                records.append({"audio_file": to_rel_path(path), "text": "", "success": False})
                continue
            records.append(self.transcribe_file(path, output_json=None, **kwargs))
        if output_json:
            save_json({"count": len(records), "records": records}, output_json)
        return records

    def transcribe_dir(self, input_dir: str, output_json: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        files = sorted(os.path.join(input_dir, n) for n in os.listdir(input_dir) if n.lower().endswith((".wav", ".flac", ".mp3")))
        return self.transcribe_batch(files, output_json=output_json, **kwargs)
