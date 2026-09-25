from typing import Any, Dict, List

from src.asr.slots import (ACTION_PATTERNS, DIRECTION_PATTERNS, NEGATION_PATTERNS, extract_quantities, extract_slots, slot_preservation)
from src.asr.text_metrics import corpus_cer, edit_distance, is_empty, is_same, sentence_error_rate, strip_punctuation, tokenize


def slot_scores(ref: str, hyp: str) -> Dict[str, float]:
    return slot_preservation(extract_slots(ref), extract_slots(hyp))

def aggregate_slots(refs: List[str], hyps: List[str]) -> Dict[str, float]:
    keys = ["direction_match", "action_match", "negation_match", "quantity_recall", "critical_score"]
    totals = {k: 0.0 for k in keys}
    n = len(refs) or 1
    for r, h in zip(refs, hyps):
        scores = slot_scores(r, h)
        for k in keys:
            totals[k] += scores[k]
    return {k: round(v / n, 4) for k, v in totals.items()}

def summarize(refs: List[str], hyps: List[str], include_slots: bool = False) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "samples": len(refs),
        "cer": round(corpus_cer(refs, hyps), 4),
        "ser": round(sentence_error_rate(refs, hyps), 4),
    }
    if include_slots:
        summary.update(aggregate_slots(refs, hyps))
    return summary