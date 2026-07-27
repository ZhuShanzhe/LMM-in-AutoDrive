from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import torch


class Qwen3ASRConfig:
    """
    Configuration for Qwen3-ASR model with __init__ style.
    Supports two load types:
        - 'custom': load from Hugging Face / ModelScope by model name
        - 'local': load from a local directory path
    """

    def __init__(
        self,
        load_type: str = "custom",                     # "custom" or "local"
        model_name: str = "Qwen/Qwen3-ASR-1.7B",       # used when load_type == "custom"
        model_path: Optional[str] = None,              # required when load_type == "local"
        device: str = "cuda:0",
        dtype: str = "bfloat16",                       # "bfloat16", "float16", "float32"
        attn_implementation: Optional[str] = None,     # "flash_attention_2" or None
        max_inference_batch_size: int = 32,
        max_new_tokens: int = 256,
        language: Optional[str] = None,
        task: str = "transcribe",                      # "transcribe" or "translate"
        enable_enhancement: bool = False,
        enhancement_method: str = "spectral",
        noise_floor: float = 0.01,
        enable_dialect_mapping: bool = False,
        dialect_map: Optional[Dict[str, str]] = None,
        **extra_kwargs,
    ):
        if load_type == "local" and model_path is None:
            raise ValueError("model_path must be provided when load_type='local'.")

        self.load_type = load_type
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.max_inference_batch_size = max_inference_batch_size
        self.max_new_tokens = max_new_tokens
        self.language = language
        self.task = task
        self.enable_enhancement = enable_enhancement
        self.enhancement_method = enhancement_method
        self.noise_floor = noise_floor
        self.enable_dialect_mapping = enable_dialect_mapping
        self.dialect_map = dialect_map or {}
        self.extra_kwargs = extra_kwargs

    def get_model_identifier(self) -> str:
        """Return the model name or local path based on load_type."""
        if self.load_type == "local":
            return self.model_path
        return self.model_name

    def get_torch_dtype(self) -> torch.dtype:
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.dtype, torch.bfloat16)
