import re
from typing import List, Tuple, Optional, Callable


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute Levenshtein distance between two strings (character-level).
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def levenshtein_distance_list(a: List[str], b: List[str]) -> int:
    """
    Compute Levenshtein distance between two lists of strings (word-level).
    """
    if len(a) < len(b):
        return levenshtein_distance_list(b, a)
    if len(b) == 0:
        return len(a)

    previous_row = list(range(len(b) + 1))
    for i, word_a in enumerate(a):
        current_row = [i + 1]
        for j, word_b in enumerate(b):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (word_a != word_b)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def word_tokenize(text: str, tokenizer: Optional[Callable[[str], List[str]]] = None) -> List[str]:
    """
    Tokenize text into words. If tokenizer is None, split by whitespace.
    For Chinese, you can pass a custom tokenizer like jieba.lcut.
    """
    if tokenizer:
        return tokenizer(text)
    return text.split()


def wer(reference: str, hypothesis: str, tokenizer: Optional[Callable[[str], List[str]]] = None) -> float:
    """
    Compute Word Error Rate (WER) between reference and hypothesis.
    Uses word-level Levenshtein distance.
    """
    ref_tokens = word_tokenize(reference, tokenizer)
    hyp_tokens = word_tokenize(hypothesis, tokenizer)
    dist = levenshtein_distance_list(ref_tokens, hyp_tokens)
    if len(ref_tokens) == 0:
        return 0.0 if len(hyp_tokens) == 0 else 1.0
    return dist / len(ref_tokens)


def cer(reference: str, hypothesis: str) -> float:
    """
    Compute Character Error Rate (CER) between reference and hypothesis.
    For Chinese, this is more appropriate.
    """
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0
    dist = levenshtein_distance(reference, hypothesis)
    return dist / len(reference)


def sentence_accuracy(reference: str, hypothesis: str) -> bool:
    """
    Returns True if reference and hypothesis are exactly equal (after stripping).
    """
    return reference.strip() == hypothesis.strip()


def evaluate_pair(reference: str, hypothesis: str, tokenizer: Optional[Callable[[str], List[str]]] = None) -> dict:
    """
    Evaluate a single pair and return dict of metrics.
    """
    return {
        "cer": cer(reference, hypothesis),
        "wer": wer(reference, hypothesis, tokenizer),
        "sentence_accuracy": sentence_accuracy(reference, hypothesis),
    }
