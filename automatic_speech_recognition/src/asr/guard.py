import logging
import re
from typing import Any, Dict, List, Optional

from .slots import extract_slots

logger = logging.getLogger(__name__)

_REPEAT_RE = re.compile(r"(.{2,6}?)\1{3,}")

class ASROutputGuard:
    def __init__(
        self, 
        min_chars: int = 2, 
        max_chars: int = 120,
        min_diversity: float = 0.35, 
        require_slot: bool = False,
        expected_language: Optional[str] = None,
        allowed_directions: Optional[List[str]] = None
    ):
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.min_diversity = min_diversity
        self.require_slot = require_slot
        self.expected_language = expected_language
        self.allowed_directions = allowed_directions

    @classmethod
    def from_yaml(cls, path: str) -> "ASROutputGuard":
        from .utils import load_yaml
        return cls(**(load_yaml(path).get("guard", {}) or {}))

    def check(self, text: str, language: Optional[str] = None) -> Dict[str, Any]:
        reasons: List[str] = []
        text = (text or "").strip()
        if len(text) < self.min_chars:
            reasons.append("too_short")
        if len(text) > self.max_chars:
            reasons.append("too_long")
        if text and _REPEAT_RE.search(text):
            reasons.append("repetition_hallucination")
        if text and len(set(text)) / len(text) < self.min_diversity:
            reasons.append("low_diversity")
        if self.expected_language and language and language != self.expected_language:
            reasons.append("unexpected_language")

        slots = extract_slots(text)
        if self.require_slot and not (slots["direction"] or slots["action"]):
            reasons.append("missing_critical_slot")
        if self.allowed_directions is not None:
            if [d for d in slots["direction"] if d not in self.allowed_directions]:
                reasons.append("unsupported_direction")

        safe = not reasons
        if not safe:
            logger.warning("guard rejected ASR text %s: %r", reasons, text[:60])
        return {"safe": safe, "reasons": reasons, "slots": slots, "text": text}
