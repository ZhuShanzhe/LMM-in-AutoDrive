import json
import os
from typing import Any, Dict, List

import librosa
import numpy as np
import soundfile as sf

def load_yaml(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}

def save_json(obj: Any, path: str) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)

def load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    audio, _ = librosa.load(path, sr=target_sr, mono=True)
    return audio.astype(np.float32)

def save_audio(audio: np.ndarray, sr: int, path: str) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    sf.write(path, audio, sr)

def peak_normalize(audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
    max_val = float(np.max(np.abs(audio))) if audio.size else 0.0
    if max_val > peak:
        audio = audio * (peak / max_val)
    return audio.astype(np.float32)

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def to_rel_path(path: str) -> str:
    if not path:
        return path
    absolute = path if os.path.isabs(path) else os.path.join(_PROJECT_ROOT, path)
    absolute = os.path.abspath(absolute)
    try:
        rel = os.path.relpath(absolute, _PROJECT_ROOT)
    except ValueError:
        return absolute
    if rel.startswith(".."):
        return absolute
    return rel.replace(os.sep, "/")

def normalize_text(text: str) -> str:
    if not text:
        return ""
    table = {ord(c): ord(d) for c, d in zip("０１２３４５６７８９", "0123456789")}
    return " ".join(text.translate(table).split()).strip()

_ENCODER_CANDIDATES = (
    "thinker.audio_tower",
    "audio_tower",
    "model.thinker.audio_tower",
    "model.audio_tower",
    "encoder",
)

_UNWRAP_ATTRS = ("model", "thinker", "module", "_model", "engine", "asr_model")

def unwrap_module(root, max_depth: int = 2):
    import torch

    if isinstance(root, torch.nn.Module):
        return root

    frontier = [(root, 0)]
    seen = {id(root)}
    while frontier:
        current, depth = frontier.pop(0)
        if depth >= max_depth:
            continue
        names = list(vars(current).keys()) if hasattr(current, "__dict__") else []
        names += [n for n in _UNWRAP_ATTRS if n not in names]
        for name in names:
            if name.startswith("__"):
                continue
            try:
                value = getattr(current, name)
            except Exception:
                continue
            if id(value) in seen:
                continue
            seen.add(id(value))
            if isinstance(value, torch.nn.Module):
                return value
            if hasattr(value, "__dict__"):
                frontier.append((value, depth + 1))
    return None

def _locate(base, dotted: str):
    owner = base
    parts = dotted.split(".")
    for name in parts[:-1]:
        owner = getattr(owner, name, None)
        if owner is None:
            return None, None
    return (owner, parts[-1]) if getattr(owner, parts[-1], None) is not None else (None, None)

def resolve_submodule(root, candidates=_ENCODER_CANDIDATES):
    import torch

    base = unwrap_module(root) or root

    for dotted in candidates:
        owner, attr_name = _locate(base, dotted)
        if owner is not None:
            return owner, attr_name, dotted

    if isinstance(base, torch.nn.Module):
        for name, _ in base.named_modules():
            if name == "audio_tower" or name.endswith(".audio_tower"):
                owner, attr_name = _locate(base, name)
                if owner is not None:
                    return owner, attr_name, name
    return None

def module_tree_hint(root, limit: int = 25) -> str:
    base = unwrap_module(root) or root
    if not hasattr(base, "named_modules"):
        attrs = list(vars(base).keys())[:limit] if hasattr(base, "__dict__") else []
        return f"no nn.Module reached; wrapper attributes: {attrs}"
    names = [n for n, _ in base.named_modules() if n]
    return f"reachable submodules ({len(names)}): {names[:limit]}"

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def write_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")