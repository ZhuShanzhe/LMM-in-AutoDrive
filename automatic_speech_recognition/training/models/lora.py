import logging
from typing import Any, Dict, Tuple

import torch

logger = logging.getLogger("training")

def load_asr_model(model_id_or_path: str, dtype: str = "bfloat16", device_map: str = "cuda:0", attn_implementation: str = "") -> torch.nn.Module:
    from transformers import AutoModel

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(dtype, torch.bfloat16)
    kwargs: Dict[str, Any] = {"trust_remote_code": True, "torch_dtype": torch_dtype}
    if device_map:
        kwargs["device_map"] = device_map
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    logger.info("Loading ASR backbone from %s ...", model_id_or_path)
    try:
        return AutoModel.from_pretrained(model_id_or_path, **kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("AutoModel load failed (%s); trying qwen_asr backend.", exc)
        from qwen_asr import Qwen3ASRModel
        wrapper = Qwen3ASRModel.from_pretrained(model_id_or_path, **kwargs)
        return getattr(wrapper, "model", wrapper)

def build_lora_model(model_id_or_path: str, lora_cfg: Dict[str, Any], dtype: str = "bfloat16", device_map: str = "cuda:0", attn_implementation: str = "") -> torch.nn.Module:
    from peft import LoraConfig, get_peft_model

    model = load_asr_model(model_id_or_path, dtype, device_map, attn_implementation)
    config = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=list(lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model