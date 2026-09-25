import json
import os
from typing import Any, Dict, Optional

import yaml

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}

def ensure_dir(path: str) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)

def save_json(obj: Any, path: str) -> None:
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)

def resolve_path(path: str, repo_root: Optional[str] = None) -> str:
    if os.path.isabs(path):
        return path
    base = repo_root or os.getcwd()
    return os.path.join(base, path)