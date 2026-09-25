import json
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DEFAULT_LEXICON_DIR = os.path.join(_PROJECT_ROOT, "resources", "dialect")
SHARED_ALIASES_FILE = "_aliases.json"

def _resolve_dir(directory: Optional[str]) -> str:
    return directory or DEFAULT_LEXICON_DIR

def available_dialects(directory: Optional[str] = None) -> List[str]:
    d = _resolve_dir(directory)
    if not os.path.isdir(d):
        return []
    return sorted(
        os.path.splitext(n)[0]
        for n in os.listdir(d)
        if n.endswith(".json") and not n.startswith("_")
    )

def load_lexicon(name: str, directory: Optional[str] = None) -> Dict[str, str]:
    d = _resolve_dir(directory)
    path = os.path.join(d, f"{name}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"dialect lexicon not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    words = data.get("words", data) if isinstance(data, dict) else {}
    return {str(k): str(v) for k, v in words.items()}

def load_shared_aliases(directory: Optional[str] = None) -> Dict[str, str]:
    d = _resolve_dir(directory)
    path = os.path.join(d, SHARED_ALIASES_FILE)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    words = data.get("words", data) if isinstance(data, dict) else {}
    return {str(k): str(v) for k, v in words.items()}

def build_lexicon(
    dialect: Optional[str] = None,
    extra: Optional[Dict[str, str]] = None,
    use_shared_aliases: bool = True,
    directory: Optional[str] = None
) -> Dict[str, str]:
    merged: Dict[str, str] = {}
    if use_shared_aliases:
        merged.update(load_shared_aliases(directory))
    if dialect:
        if dialect.lower() not in available_dialects(directory):
            logger.warning("unknown dialect '%s'; using shared aliases only.", dialect)
        else:
            merged.update(load_lexicon(dialect.lower(), directory))
    if extra:
        merged.update({str(k): str(v) for k, v in extra.items()})
    return merged
