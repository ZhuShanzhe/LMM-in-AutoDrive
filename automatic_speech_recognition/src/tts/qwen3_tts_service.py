import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from qwen_tts import Qwen3TTSModel

from src.utils import DEFAULT_NUM_GPUS, resolve_device

from .utils import add_noise, read_wav, save_wav_16k

logger = logging.getLogger(__name__)

_LOADED: Dict[Tuple[str, str], "Qwen3TTSModel"] = {}

_DEFAULT_SPEAKERS = ["Vivian", "Serena", "Uncle_Fu"]

class Qwen3TTSService:
    def __init__(
        self,
        model_id_or_path: str,
        device_map: Optional[str] = None,
        num_gpus: int = DEFAULT_NUM_GPUS,
        dtype: str = "bfloat16",
        attn_implementation: str = "",
        gen_kwargs: Optional[Dict[str, Any]] = None,
        output_dir: str = "outputs",
        target_sample_rate: int = 16000,
    ) -> None:
        self.model_id_or_path = model_id_or_path
        self.output_dir = output_dir
        self.target_sample_rate = target_sample_rate
        self.gen_kwargs = dict(gen_kwargs or {})
        device_map = device_map or resolve_device(num_gpus)
        cache_key = (model_id_or_path, str(device_map))
        if cache_key not in _LOADED:
            kwargs: Dict[str, Any] = {"device_map": device_map, "dtype": getattr(torch, dtype)}
            if attn_implementation:
                kwargs["attn_implementation"] = attn_implementation
            logger.info("Loading Qwen3-TTS from %s on %s ...", model_id_or_path, device_map)
            _LOADED[cache_key] = Qwen3TTSModel.from_pretrained(model_id_or_path, **kwargs)
            logger.info("Qwen3-TTS model ready.")
        self.model = _LOADED[cache_key]

    def synthesize(
        self,
        texts: List[str],
        speakers: Optional[List[str]] = None,
        instructs: Optional[List[str]] = None,
        language: str = "Chinese",
    ) -> Tuple[List[np.ndarray], int]:
        n = len(texts)
        if n == 0:
            return [], 0
        if speakers is None:
            speakers = [_DEFAULT_SPEAKERS[i % len(_DEFAULT_SPEAKERS)] for i in range(n)]
        if instructs is None:
            instructs = [None] * n
        return self.model.generate_custom_voice(
            text=list(texts),
            language=[language] * n,
            speaker=list(speakers),
            instruct=list(instructs),
            **self.gen_kwargs,
        )

    DIALECT_SPEAKERS: Dict[str, Dict[str, str]] = {
        "sichuan": {"speaker": "Eric", "instruct": "请用四川口音自然朗读，尾音上扬、略带椒盐味。"},
        "beijing": {"speaker": "Dylan", "instruct": "请用北京口音自然朗读，儿化音明显、语气轻松。"},
        "dongbei": {"speaker": "Vivian", "instruct": "请用东北口音朗读，语调豪爽、略带大碴子味。"},
        "shaanxi": {"speaker": "Uncle_Fu", "instruct": "请用陕西方言口音朗读，厚重朴实、拖腔明显。"},
        "shandong": {"speaker": "Uncle_Fu", "instruct": "请用山东口音朗读，语调直朴、鼻音略重。"},
        "henan": {"speaker": "Serena", "instruct": "请用河南口音朗读，语调圆润平实。"},
        "shanghai": {"speaker": "Vivian", "instruct": "请用带上海腔调的普通话朗读，语调软糯。"},
        "cantonese": {"speaker": "Uncle_Fu", "instruct": "请用带广东腔调的普通话朗读，尾音短促。"},
        "hunan": {"speaker": "Serena", "instruct": "请用带湖南腔调的普通话朗读，语速略快。"},
        "tianjin": {"speaker": "Dylan", "instruct": "请用天津口音朗读，语调幽默俏皮、抑扬夸张。"},
    }

    @classmethod
    def supported_dialects(cls) -> List[str]:
        return list(cls.DIALECT_SPEAKERS.keys())

    @staticmethod
    def _resolve_voice(dialect, speaker, instruct):
        if isinstance(dialect, str) and dialect in Qwen3TTSService.DIALECT_SPEAKERS:
            p = Qwen3TTSService.DIALECT_SPEAKERS[dialect]
            return p["speaker"], p["instruct"]
        if isinstance(dialect, dict):
            return dialect.get("speaker", speaker), dialect.get("instruct", instruct)
        return speaker, instruct

    def generate(
        self,
        text: str,
        output_file: Optional[str] = None,
        speaker: Optional[str] = None,
        instruct: Optional[str] = None,
        dialect = None,  # str preset name or dict(speaker=..., instruct=...)
        language: str = "Chinese",
        add_noise: bool = False,
        noise_type: str = "white",
        snr_db: float = 20.0,
        real_noise_file: Optional[str] = None,
        save: bool = True,
    ) -> Dict[str, Any]:
        speaker, instruct = self._resolve_voice(dialect, speaker, instruct)
        wavs, sr = self.synthesize([text], [speaker], [instruct], language)
        audio = wavs[0]
        if add_noise:
            audio = add_noise(audio, noise_type, snr_db, real_noise_file, sr)
        if output_file is None:
            output_file = os.path.join(self.output_dir, "utterance.wav")
        if save:
            save_wav_16k(audio, sr, output_file, self.target_sample_rate)
        return {
            "text": text, 
            "audio_file": output_file,
            "sample_rate": self.target_sample_rate,
            "speaker": speaker, 
            "instruct": instruct,
            "dialect": dialect, 
            "noise_type": noise_type if add_noise else None,
            "snr_db": snr_db if add_noise else None,
            "real_noise_file": real_noise_file if add_noise else None,
        }

    def generate_batch(
        self,
        texts: List[str],
        output_dir: Optional[str] = None,
        file_names: Optional[List[str]] = None,
        speakers: Optional[List[str]] = None,
        instructs: Optional[List[str]] = None,
        language: str = "Chinese",
        add_noise: bool = False,
        noise_type: str = "white",
        snr_db: float = 20.0,
        real_noise_file: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        out_dir = output_dir or self.output_dir
        os.makedirs(out_dir, exist_ok=True)
        wavs, sr = self.synthesize(texts, speakers, instructs, language)
        if file_names is None:
            file_names = [f"utt_{i:05d}" for i in range(len(texts))]
        records = []
        for i, (text, audio) in enumerate(zip(texts, wavs)):
            if add_noise:
                audio = add_noise(audio, noise_type, snr_db, real_noise_file, sr)
            name = file_names[i]
            if not name.lower().endswith(".wav"):
                name += ".wav"
            path = os.path.join(out_dir, name)
            save_wav_16k(audio, sr, path, self.target_sample_rate)
            records.append({
                "text": text, 
                "audio_file": path,
                "sample_rate": self.target_sample_rate,
                "noise_type": noise_type if add_noise else None,
                "snr_db": snr_db if add_noise else None,
            })
        return records

    def add_noise_to_file(
        self,
        audio_path: str,
        output_path: str,
        noise_type: str = "white",
        snr_db: float = 20.0,
        real_noise_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load a wav, mix noise, save the processed wav, return its record."""
        audio, _ = read_wav(audio_path, self.target_sample_rate)
        noisy = add_noise(audio, noise_type, snr_db, real_noise_file, self.target_sample_rate)
        save_wav_16k(noisy, self.target_sample_rate, output_path, self.target_sample_rate)
        return {
            "audio_file": output_path, 
            "noise_type": noise_type,
            "snr_db": snr_db, 
            "real_noise_file": real_noise_file,
        }
