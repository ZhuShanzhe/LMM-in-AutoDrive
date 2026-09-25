import logging
import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger("training")

class BottleneckAdapter(nn.Module):
    def __init__(self, dim: int, bottleneck: int = 64, dropout: float = 0.0, init_scale: float = 1e-3):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * self.up(self.act(self.down(self.norm(x))))

class SoftPrefix(nn.Module):
    def __init__(self, dim: int, num_tokens: int = 8):
        super().__init__()
        self.num_tokens = num_tokens
        self.embedding = nn.Parameter(torch.randn(1, num_tokens, dim) * 0.02)

    def expand(self, batch_size: int) -> torch.Tensor:
        return self.embedding.expand(batch_size, -1, -1)

def _infer_dim(module: nn.Module) -> Optional[int]:
    for m in module.modules():
        if isinstance(m, nn.Linear) and m.in_features == m.out_features:
            return m.in_features
        if isinstance(m, nn.LayerNorm):
            return int(m.normalized_shape[-1])
    return None

def _find_layers(model: nn.Module, name_contains: str) -> Tuple[Optional[str], Optional[nn.ModuleList], Optional[int]]:
    for name, module in model.named_modules():
        if name_contains and name_contains not in name:
            continue
        layers = getattr(module, "layers", None)
        if isinstance(layers, nn.ModuleList) and len(layers) > 0:
            dim = (getattr(module, "hidden_size", None) or getattr(module, "d_model", None) or _infer_dim(module))
            return name, layers, int(dim) if dim else None
    return None, None, None

def _add_delta(output: Any, delta: torch.Tensor) -> Any:
    if isinstance(output, torch.Tensor):
        return output + delta
    if isinstance(output, (tuple, list)):
        head = output[0]
        if isinstance(head, torch.Tensor):
            rebuilt = (head + delta,) + tuple(output[1:])
            return list(rebuilt) if isinstance(output, list) else type(output)(rebuilt)
    return output

def inject_adapters(model: nn.Module, target: str = "audio", bottleneck: int = 64, dropout: float = 0.0, last_n: Optional[int] = None) -> Tuple[nn.ModuleDict, Dict[str, Any]]:
    name, layers, dim = _find_layers(model, target)
    if layers is None:
        logger.warning("no layer stack matched '%s'; retrying with empty filter.", target)
        name, layers, dim = _find_layers(model, "")
    if layers is None or dim is None:
        raise RuntimeError("could not locate transformer layers for adapter injection")

    ref = next(layers.parameters())
    adapters = nn.ModuleDict()
    state: Dict[str, Any] = {"layers": [], "target": name, "dim": dim}
    indices = range(len(layers)) if last_n is None else range(len(layers) - last_n, len(layers))

    for idx in indices:
        adapter = BottleneckAdapter(dim, bottleneck, dropout)
        adapter.to(device=ref.device, dtype=ref.dtype)
        adapters[f"{name}_{idx}".replace(".", "_")] = adapter
        original_forward = layers[idx].forward

        def make_forward(orig, ad):
            def forward(*args, **kwargs):
                out = orig(*args, **kwargs)
                hidden = out[0] if isinstance(out, (tuple, list)) and out else out
                if isinstance(hidden, torch.Tensor):
                    return _add_delta(out, ad(hidden))
                return out
            return forward

        layers[idx].forward = make_forward(original_forward, adapter)
        state["layers"].append(layers[idx])

    logger.info("injected %d adapters into '%s' (dim=%d)", len(adapters), name, dim)
    return adapters, state

def freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False

def parameter_summary(model: nn.Module) -> Dict[str, Any]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "trainable_ratio": round(trainable / max(1, total), 6)}

def save_adapters(adapters: nn.ModuleDict, path: str) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    torch.save(adapters.state_dict(), path)
    logger.info("adapters saved -> %s", path)

def load_adapters(adapters: nn.ModuleDict, path: str, strict: bool = True) -> None:
    adapters.load_state_dict(torch.load(path, map_location="cpu"), strict=strict)
    logger.info("adapters loaded <- %s", path)