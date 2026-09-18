import re
import unicodedata
from typing import List

_WORD_RE = re.compile(r"[A-Za-z0-9.]+")

def strip_punctuation(text: str) -> str:
    _DECIMAL_MARKS = { "." , "．" , "。" , "｡" , "､" }
    if not text:
        return ""
    chars: List[str] = []
    for i, ch in enumerate(text):
        if ch in _DECIMAL_MARKS and 0 < i < len(text) - 1 and text[i - 1].isdigit() and text[i + 1].isdigit():
            chars.append(".")       # normalise to an ASCII dot
        elif unicodedata.category(ch).startswith("P"):
            continue
        else:
            chars.append(ch)
    return " ".join("".join(chars).split())

def tokenize(text: str) -> List[str]:
    stripped = strip_punctuation(text)
    units: List[str] = []
    i = 0
    while i < len(stripped):
        ch = stripped[i]
        if ch.isspace():
            i += 1
            continue
        match = _WORD_RE.match(stripped, i)
        if match:
            units.append(match.group(0).lower())
            i = match.end()
        else:
            units.append(ch.lower())
            i += 1
    return units

def _unit_distance(ref_units: List[str], hyp_units: List[str]) -> int:
    if not ref_units:
        return len(hyp_units)
    if not hyp_units:
        return len(ref_units)
    prev = list(range(len(hyp_units) + 1))
    for i, ru in enumerate(ref_units, 1):
        cur = [i]
        for j, hu in enumerate(hyp_units, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ru != hu)))
        prev = cur
    return prev[-1]

def edit_distance(ref: str, hyp: str) -> int:
    return _unit_distance(tokenize(ref), tokenize(hyp))

def corpus_cer(refs: List[str], hyps: List[str]) -> float:
    ref_units = [tokenize(r) for r in refs]
    hyp_units = [tokenize(h) for h in hyps]
    total_ed = sum(_unit_distance(r, h) for r, h in zip(ref_units, hyp_units))
    total_len = sum(len(u) for u in ref_units) or 1
    return total_ed / total_len

def sentence_error_rate(refs: List[str], hyps: List[str]) -> float:
    if not refs:
        return 0.0
    wrong = sum(1 for r, h in zip(refs, hyps) if tokenize(r) != tokenize(h))
    return wrong / len(refs)

def is_same(ref: str, hyp: str) -> bool:
    return tokenize(ref) == tokenize(hyp)

def is_empty(text: str) -> bool:
    return not tokenize(text)

__all__ = [
    "strip_punctuation",
    "tokenize",
    "edit_distance",
    "corpus_cer",
    "sentence_error_rate",
    "is_same",
    "is_empty",
]
