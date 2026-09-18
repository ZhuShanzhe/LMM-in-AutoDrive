import logging
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

from .lexicons import build_lexicon

logger = logging.getLogger(__name__)

try:
    from pypinyin import lazy_pinyin
    _HAS_PINYIN = True
except Exception:  # noqa: BLE001
    _HAS_PINYIN = False

class DialectNormalizer:
    def __init__(
        self, 
        dialect: Optional[str] = None,
        custom_map: Optional[Dict[str, str]] = None,
        lexicon_dir: Optional[str] = None,
        use_shared_aliases: bool = True,
        fuzzy_threshold: float = 0.85,
        hotwords: Optional[List[str]] = None,
        enable_pinyin: bool = True
    ):
        self.dialect = dialect
        self.lexicon = build_lexicon(dialect, custom_map, use_shared_aliases, lexicon_dir)
        self._ordered: List[Tuple[str, str]] = sorted(
            self.lexicon.items(), key=lambda kv: len(kv[0]), reverse=True)
        self.fuzzy_threshold = fuzzy_threshold
        self.hotwords = [w for w in (hotwords or []) if w]
        self.enable_pinyin = enable_pinyin and _HAS_PINYIN
        self._pinyin_index: Dict[str, str] = {}
        if self.enable_pinyin:
            for src, tgt in self.lexicon.items():
                if len(src) >= 2 and len(src) == len(tgt):
                    try:
                        self._pinyin_index["".join(lazy_pinyin(src))] = tgt
                    except Exception:
                        continue

    def _apply_lexicon(self, text: str) -> str:
        for src, tgt in self._ordered:
            if src in text:
                text = text.replace(src, tgt)
        return text

    def _apply_pinyin(self, text: str) -> str:
        if not self._pinyin_index or not text:
            return text
        chars = list(text)
        n = len(chars)
        max_len = max((len(k) for k in self._pinyin_index), default=0) // 2
        for start in range(n):
            for length in range(min(max_len, n - start), 1, -1):
                window = "".join(chars[start:start + length])
                try:
                    key = "".join(lazy_pinyin(window))
                except Exception:  # noqa: BLE001
                    continue
                tgt = self._pinyin_index.get(key)
                if tgt and tgt != window:
                    chars[start:start + length] = list(tgt)
                    break
        return "".join(chars)

    def _apply_hotwords(self, text: str) -> str:
        if not self.hotwords or not text:
            return text
        out = text
        for hw in self.hotwords:
            L = len(hw)
            if L == 0:
                continue
            best_ratio, best_seg, best_pos = self.fuzzy_threshold, None, -1
            for pos in range(0, len(out) - L + 1):
                seg = out[pos:pos + L]
                if seg == hw:
                    break
                ratio = SequenceMatcher(None, seg, hw).ratio()
                if ratio > best_ratio:
                    best_ratio, best_seg, best_pos = ratio, seg, pos
            if best_seg is not None:
                out = out[:best_pos] + hw + out[best_pos + L:]
        return out

    def normalize(self, text: str) -> str:
        if not text or not text.strip():
            return text
        out = self._apply_lexicon(text)
        if self.enable_pinyin:
            out = self._apply_pinyin(out)
        out = self._apply_hotwords(out)
        return " ".join(out.split()).strip()