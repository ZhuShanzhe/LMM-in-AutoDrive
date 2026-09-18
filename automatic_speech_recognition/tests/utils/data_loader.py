import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]

_META_KEYS = ("dialect", "speaker", "noise_type", "snr_db", "dataset", "condition", "sample_rate")

def _resolve_audio(path: str) -> str:
    if not path or os.path.isabs(path):
        return path
    return os.path.join(str(ROOT), path)

def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)

def load_mapping(path: str) -> List[Dict[str, Any]]:
    data = _read_json(path)
    if isinstance(data, dict):
        data = data.get("records", [])
    records: List[Dict[str, Any]] = []
    for i, item in enumerate(data, 1):
        audio = _resolve_audio(item.get("audio_file") or item.get("audio"))
        text = item.get("text") or item.get("translation") or item.get("original") or ""
        if audio and str(text).strip():
            rec = {"index": item.get("index", i), "audio_file": audio, "text": str(text).strip()}
            for key in _META_KEYS:
                if key in item:
                    rec[key] = item[key]
            records.append(rec)
    return records

def load_commands(path: str) -> List[Dict[str, Any]]:
    data = _read_json(path)
    if not isinstance(data, list):
        raise ValueError("commands file must be a list")
    out: List[Dict[str, Any]] = []
    for i, item in enumerate(data, 1):
        if isinstance(item, str):
            text = item.strip()
            idx = i
        else:
            text = str(item.get("text") or item.get("translation") or "").strip()
            idx = item.get("index", i)
        if text:
            out.append({"index": idx, "text": text})
    return out

def load_dir_with_commands(audio_dir: str, commands_file: str, pattern: str = "*.wav", name_format: Optional[str] = None) -> List[Dict[str, Any]]:
    import glob

    commands = load_commands(commands_file)
    files = sorted(glob.glob(os.path.join(audio_dir, pattern)))
    if name_format:
        by_name = {os.path.basename(f): f for f in files}
        records = []
        for cmd in commands:
            name = name_format.format(index=cmd["index"], text=cmd["text"])
            path = by_name.get(name)
            if path:
                records.append({"index": cmd["index"], "audio_file": path, "text": cmd["text"]})
        return records
    return [{"index": cmd["index"], "audio_file": f, "text": cmd["text"]} for cmd, f in zip(commands, files)]

def load_dataset(source: str, commands_file: Optional[str] = None, limit: Optional[int] = None, exists_only: bool = True) -> List[Dict[str, Any]]:
    if os.path.isdir(source):
        if not commands_file:
            raise ValueError("commands_file is required when source is a directory")
        records = load_dir_with_commands(source, commands_file)
    else:
        records = load_mapping(source)

    if exists_only:
        records = [r for r in records if os.path.exists(r["audio_file"])]
    if limit:
        records = records[:limit]
    return records

def subset_by_condition(records: List[Dict[str, Any]], condition: Optional[str], keys: tuple = ("condition", "dataset")) -> List[Dict[str, Any]]:
    if not condition:
        return records
    return [r for r in records if any(str(r.get(k)) == condition for k in keys)]