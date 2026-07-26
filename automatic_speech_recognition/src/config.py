from dataclasses import dataclass, field
from typing import Optional, Dict


@dataclass
class ASRConfig:
    asr_mode: str = "single"
    asr_device: str = "cuda:0"
    asr_model: Optional[str] = None
    asr_vad_model: Optional[str] = None
    asr_punc_model: Optional[str] = None
    asr_spk_model: Optional[str] = None

    trans_model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    trans_load_type: str = "custom"
    trans_model_path: Optional[str] = None
    trans_src_lang: str = "zho_Hans"
    trans_tgt_lang: str = "eng_Latn"
    trans_max_length: int = 512
    trans_generation_max_length: Optional[int] = None
    trans_num_beams: int = 4
    trans_temperature: float = 0.0
    trans_device: str = "cuda:0"

    output_dir: str = "data"
    raise_on_error: bool = False


@dataclass
class Qwen3PipelineConfig:
    load_type: str = "custom"
    model_name: str = "Qwen/Qwen3-ASR-1.7B-hf"
    model_path: Optional[str] = None
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    attn_implementation: Optional[str] = None
    max_inference_batch_size: int = 32
    max_new_tokens: int = 256
    language: Optional[str] = None
    task: str = "transcribe"

    enable_enhancement: bool = False
    enhancement_method: str = "spectral"
    noise_floor: float = 0.01
    enable_dialect_mapping: bool = False
    dialect_map: Dict[str, str] = field(default_factory=dict)

    trans_model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    trans_load_type: str = "custom"
    trans_model_path: Optional[str] = None
    trans_src_lang: str = "zho_Hans"
    trans_tgt_lang: str = "eng_Latn"
    trans_max_length: int = 512
    trans_generation_max_length: Optional[int] = None
    trans_num_beams: int = 4
    trans_temperature: float = 0.0
    trans_device: str = "cuda:0"

    output_dir: str = "data"
    raise_on_error: bool = False
