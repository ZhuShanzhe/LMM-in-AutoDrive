import torch
import numpy as np
from typing import Optional, List, Union, Dict, Any
from qwen_asr import Qwen3ASRModel as _Qwen3ASRModel

from .config import Qwen3ASRConfig


class Qwen3ASRModel:
    def __init__(self, config: Optional[Qwen3ASRConfig] = None):
        self.config = config or Qwen3ASRConfig()
        self._model = None

    def _load_model(self):
        if self._model is None:
            model_id = self.config.get_model_identifier()
            print(f"Loading Qwen3-ASR model from: {model_id} (load_type={self.config.load_type})")
            kwargs = {
                "dtype": self.config.get_torch_dtype(),
                "device_map": self.config.device,
                "attn_implementation": self.config.attn_implementation,
                "max_inference_batch_size": self.config.max_inference_batch_size,
                "max_new_tokens": self.config.max_new_tokens,
                "trust_remote_code": True,
            }
            if self.config.load_type == "local":
                kwargs["local_files_only"] = True
            self._model = _Qwen3ASRModel.from_pretrained(
                model_id,
                **kwargs
            )
            print("Qwen3-ASR model loaded successfully.")
        return self._model

    def transcribe(
        self,
        audio_input: Union[str, np.ndarray],
        language: Optional[str] = None,
        **generate_kwargs
    ) -> List[Dict[str, Any]]:
        model = self._load_model()
        results = model.transcribe(
            audio=audio_input,
            language=language or self.config.language,
            **generate_kwargs
        )
        return results
