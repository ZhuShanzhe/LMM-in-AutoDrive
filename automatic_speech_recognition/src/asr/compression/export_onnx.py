import logging
import os
from typing import Any, Dict, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

def _file_size_mb(path: str) -> float:
    return round(os.path.getsize(path) / (1024 * 1024), 3) if os.path.exists(path) else 0.0

def export_module_to_onnx(
    module, 
    dummy_inputs, 
    output_path: str,
    input_names: Sequence[str], 
    output_names: Sequence[str],
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    opset: int = 17,
) -> Dict[str, Any]:
    import torch

    folder = os.path.dirname(output_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    module.eval()
    with torch.no_grad():
        torch.onnx.export(
            module, 
            dummy_inputs, 
            output_path,
            input_names=list(input_names), 
            output_names=list(output_names),
            dynamic_axes=dynamic_axes or {}, 
            opset_version=opset,
            do_constant_folding=True,
        )
    report = {"onnx_path": output_path, "opset": opset, "size_mb": _file_size_mb(output_path)}
    logger.info("exported ONNX: %s (%s MB)", output_path, report["size_mb"])
    return report

def _load_asr_model(model_id_or_path: str, torch_dtype):
    try:
        from qwen_asr import Qwen3ASRModel
    except ImportError as exc:
        raise ImportError("qwen_asr is required to export Qwen3-ASR; install the qwen-asr package") from exc

    kwargs: Dict[str, Any] = {
        "dtype": torch_dtype,
        "device_map": "cpu",
        "trust_remote_code": True,
    }
    if os.path.isdir(model_id_or_path):
        kwargs["local_files_only"] = True
    return Qwen3ASRModel.from_pretrained(model_id_or_path, **kwargs)

def _feat_len_after_cnn(mel_len: int) -> int:
    leave = mel_len % 100
    feat = (leave - 1) // 2 + 1
    return ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (mel_len // 100) * 13

def _build_cu_seqlens(total_len: int, tokens_per_chunk: int, n_window: int, n_window_infer: int):
    import torch

    window = tokens_per_chunk * (n_window_infer // (n_window * 2))
    parts = [0]
    parts.extend([window] * (total_len // window))
    rem = total_len % window
    if rem:
        parts.append(rem)
    return torch.tensor(parts, dtype=torch.int32).cumsum(0)

def _set_attn_eager(module) -> None:
    if hasattr(module, "config") and hasattr(module.config, "_attn_implementation"):
        module.config._attn_implementation = "eager"
    for child in module.modules():
        if hasattr(child, "config") and hasattr(child.config, "_attn_implementation"):
            child.config._attn_implementation = "eager"

def _build_encoder_wrapper(encoder, mel_frames: int):
    import torch
    from torch.nn import functional as F

    cfg = encoder.config
    _set_attn_eager(encoder)
    chunk_mel = int(cfg.n_window) * 2
    if mel_frames <= 0 or mel_frames % chunk_mel != 0:
        raise ValueError(f"mel_frames must be a positive multiple of {chunk_mel}, got {mel_frames}")
    num_chunks = mel_frames // chunk_mel
    if num_chunks > int(cfg.conv_chunksize):
        raise ValueError(f"mel_frames={mel_frames} yields {num_chunks} chunks, exceeding conv_chunksize={cfg.conv_chunksize}")
    tokens_per_chunk = _feat_len_after_cnn(chunk_mel)
    total_tokens = _feat_len_after_cnn(mel_frames)
    logger.info("encoder wrapper: %d chunks x %d mel -> %d audio tokens", num_chunks, chunk_mel, total_tokens)

    class _StaticAudioEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder
            self.num_chunks = num_chunks
            self.chunk_mel = chunk_mel
            self.tokens_per_chunk = tokens_per_chunk
            self.total_tokens = total_tokens
            cu = _build_cu_seqlens(total_tokens, tokens_per_chunk, int(cfg.n_window), int(cfg.n_window_infer))
            self.register_buffer("cu_seqlens", cu, persistent=False)

        def forward(self, input_features):
            enc = self.encoder
            x = input_features.reshape(enc.num_mel_bins, self.num_chunks, self.chunk_mel)
            x = x.permute(1, 0, 2).contiguous().unsqueeze(1)
            x = F.gelu(enc.conv2d1(x))
            x = F.gelu(enc.conv2d2(x))
            x = F.gelu(enc.conv2d3(x))
            b, c, f, t = x.size()
            x = enc.conv_out(x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))
            pos = enc.positional_embedding.positional_embedding[: self.tokens_per_chunk, :].unsqueeze(0).to(x.dtype)
            x = x + pos
            hidden_states = x.reshape(self.total_tokens, -1)
            for layer in enc.layers:
                hidden_states = layer(hidden_states, self.cu_seqlens)[0]
            hidden_states = enc.ln_post(hidden_states)
            hidden_states = enc.act(enc.proj1(hidden_states))
            return enc.proj2(hidden_states)

    return _StaticAudioEncoder()

def export_asr_onnx(
    model_id_or_path: str, 
    output_dir: str = "outputs/onnx",
    opset: int = 17, dtype: str = "float32",
    mel_frames: int = 3000,
) -> Dict[str, Any]:
    import torch
    from transformers import WhisperFeatureExtractor

    from ..utils import module_tree_hint, resolve_submodule, unwrap_module

    torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}.get(dtype, torch.float32)
    wrapper = _load_asr_model(model_id_or_path, torch_dtype)

    base = unwrap_module(wrapper) or wrapper
    if hasattr(base, "eval"):
        base.eval()

    found = resolve_submodule(wrapper)
    if found is None:
        raise RuntimeError("could not locate the audio encoder; " + module_tree_hint(wrapper))
    owner, attr_name, dotted = found
    encoder = getattr(owner, attr_name)
    logger.info("audio encoder located at %s", dotted)

    local_only = os.path.isdir(model_id_or_path)
    fe = WhisperFeatureExtractor.from_pretrained(model_id_or_path, local_files_only=local_only)
    hop = int(getattr(fe, "hop_length", 160))
    audio = np.zeros(mel_frames * hop, dtype=np.float32)
    features = fe(audio, sampling_rate=fe.sampling_rate, return_tensors="pt").input_features[0]
    actual_frames = int(features.shape[-1])
    logger.info("mel features %s (mel_frames=%d)", list(features.shape), actual_frames)

    module = _build_encoder_wrapper(encoder, actual_frames)
    with torch.no_grad():
        reference = encoder(features, feature_lens=torch.tensor([actual_frames], dtype=torch.long)).last_hidden_state
        produced = module(features)
    max_diff = float((reference - produced).abs().max().item())
    logger.info("wrapper vs original encoder max_abs_diff = %s", max_diff)

    out_path = os.path.join(output_dir, "asr_encoder.onnx")
    report = export_module_to_onnx(
        module, 
        (features,), 
        out_path,
        input_names=["input_features"], output_names=["encoder_output"],
        dynamic_axes={}, 
        opset=opset,
    )
    report["encoder_path"] = dotted
    report["mel_frames"] = actual_frames
    report["input_shape"] = list(features.shape)
    report["output_shape"] = list(produced.shape)
    report["wrapper_vs_encoder_max_abs_diff"] = max_diff
    logger.info("ASR encoder exported: %s", report)
    return {"encoder": report}
