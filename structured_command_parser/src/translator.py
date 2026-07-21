from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from time import perf_counter
from typing import Any

from .normalizer import normalize_text
from .qwen_runtime import QwenRuntime


MODULE_ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = MODULE_ROOT / "configs" / "translator_prompt.txt"
GLOSSARY_PATH = MODULE_ROOT / "configs" / "translation_glossary.json"


@dataclass(frozen=True)
class GlossaryMatch:
    source: str
    target: str
    aliases: tuple[str, ...] = ()


@dataclass
class TranslationResult:
    source_text: str
    normalized_source_text: str
    translated_text: str
    source_language: str
    target_language: str
    model: str
    glossary_version: str
    matched_terms: list[GlossaryMatch]
    term_constraints_passed: bool
    warnings: list[str]
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["latency_ms"] = round(self.latency_ms, 3)
        return payload


class ConstrainedQwenTranslator:
    def __init__(
        self,
        model_path: str,
        *,
        glossary_path: Path = GLOSSARY_PATH,
        max_new_tokens: int = 160,
    ) -> None:
        self.model_path = model_path
        self.runtime = QwenRuntime(model_path)
        self.max_new_tokens = max_new_tokens
        glossary = json.loads(glossary_path.read_text(encoding="utf-8"))
        self.glossary_version = str(glossary["version"])
        self.terms = sorted(
            glossary["terms"], key=lambda item: len(item["source"]), reverse=True
        )

    def translate(self, source_text: str) -> TranslationResult:
        if not source_text or not source_text.strip():
            raise ValueError("Source text cannot be empty")
        normalized = normalize_text(source_text)
        _, matches = self._protect_terms(normalized)
        prompt = PROMPT_PATH.read_text(encoding="utf-8")
        if matches:
            prompt += "\nRequired term mappings for this command:\n" + "\n".join(
                f"- {match.source} => {match.target}" for match in matches
            )
        started = perf_counter()
        generated = self._clean(
            self.runtime.generate(prompt, normalized, max_new_tokens=self.max_new_tokens)
        )
        translated = self._preserve_source_semantics(
            normalized, self._enforce_terms(generated, matches)
        )
        missing = self._missing_targets(translated, matches)
        empty_translation = not translated.strip()
        has_untranslated_chinese = bool(re.search(r"[\u4e00-\u9fff]", translated))
        if missing or has_untranslated_chinese or empty_translation:
            retry_prompt = (
                prompt
                + "\nThe first attempt omitted required canonical terms. Translate again and "
                + "use every right-hand English term exactly."
            )
            generated = self._clean(
                self.runtime.generate(
                    retry_prompt, normalized, max_new_tokens=self.max_new_tokens
                )
            )
            translated = self._preserve_source_semantics(
                normalized, self._enforce_terms(generated, matches)
            )
            missing = self._missing_targets(translated, matches)
            empty_translation = not translated.strip()
            has_untranslated_chinese = bool(
                re.search(r"[\u4e00-\u9fff]", translated)
            )

        warnings = []
        if missing:
            translated = ", ".join(match.target for match in missing) + ", " + translated
            warnings.append(
                "Canonical terms inserted after two constrained translation attempts: "
                + ", ".join(match.target for match in missing)
            )
            missing = []
        if has_untranslated_chinese:
            warnings.append("Translation still contains Chinese text.")
        if empty_translation:
            translated = "ambiguous driving command"
            warnings.append(
                "Translation model returned empty text twice; clarification fallback used."
            )
        return TranslationResult(
            source_text=source_text,
            normalized_source_text=normalized,
            translated_text=translated,
            source_language="zh-CN",
            target_language="en-US",
            model=Path(self.model_path).name,
            glossary_version=self.glossary_version,
            matched_terms=matches,
            term_constraints_passed=not missing and not has_untranslated_chinese,
            warnings=warnings,
            latency_ms=(perf_counter() - started) * 1000,
        )

    def _protect_terms(self, text: str) -> tuple[str, list[GlossaryMatch]]:
        occupied = [False] * len(text)
        matches: list[tuple[int, int, GlossaryMatch]] = []
        for term in self.terms:
            for found in re.finditer(re.escape(term["source"]), text):
                if any(occupied[found.start() : found.end()]):
                    continue
                for index in range(found.start(), found.end()):
                    occupied[index] = True
                matches.append(
                    (
                        found.start(),
                        found.end(),
                        GlossaryMatch(
                            term["source"],
                            term["target"],
                            tuple(term.get("aliases", [])),
                        ),
                    )
                )
        protected = text
        ordered = [match for _, _, match in sorted(matches)]
        index_by_match = {match: index for index, match in enumerate(ordered)}
        for start, end, match in sorted(matches, reverse=True):
            placeholder = f"__DRIVE_TERM_{index_by_match[match]:03d}__"
            protected = protected[:start] + f" {placeholder} " + protected[end:]
        return protected, ordered

    @staticmethod
    def _restore_placeholders(
        translated: str, matches: list[GlossaryMatch]
    ) -> str:
        restored = translated
        for index, match in enumerate(matches):
            restored = restored.replace(f"__DRIVE_TERM_{index:03d}__", match.target)
        return " ".join(restored.split())

    @staticmethod
    def _enforce_terms(
        translated: str, matches: list[GlossaryMatch]
    ) -> str:
        enforced = translated
        for match in matches:
            if match.target.casefold() in enforced.casefold():
                continue
            for alias in match.aliases:
                pattern = re.compile(re.escape(alias), re.IGNORECASE)
                if pattern.search(enforced):
                    enforced = pattern.sub(match.target, enforced, count=1)
                    break
        return " ".join(enforced.split())

    @staticmethod
    def _preserve_source_semantics(source: str, translated: str) -> str:
        result = translated
        ambiguous_lane = bool(
            re.search(r"并线|变道|换(?:到)?.{0,3}车道|换道|换个道|并过去", source)
            and not re.search(r"左|右", source)
        )
        if ambiguous_lane:
            result = re.sub(
                r"change lane to (?:the )?(?:left|right)",
                "change lane",
                result,
                flags=re.IGNORECASE,
            )
            if not re.search(r"change lane", result, re.IGNORECASE):
                result = "change lane"
        ambiguous_turn = bool(
            re.search(r"转|拐", source) and not re.search(r"左|右|直行|直走", source)
        )
        if ambiguous_turn:
            result = re.sub(
                r"turn (?:to the )?(?:left|right)|go straight",
                "turn",
                result,
                flags=re.IGNORECASE,
            )
        if "避让" not in source and "让行" not in source:
            result = re.sub(
                r"(?:,?\s*)yield to (?:the )?(?:pedestrian|passengers?)(?:,?\s*)",
                ", ",
                result,
                flags=re.IGNORECASE,
            )
        return " ".join(result.strip(" ,").split())

    @staticmethod
    def _missing_targets(
        translated: str, matches: list[GlossaryMatch]
    ) -> list[GlossaryMatch]:
        lowered = translated.casefold()
        return [match for match in matches if match.target.casefold() not in lowered]

    @staticmethod
    def _clean(text: str) -> str:
        candidate = " ".join(text.strip().split())
        candidate = candidate.removeprefix("Translation:").strip()
        if len(candidate) >= 2 and candidate[0] == candidate[-1] and candidate[0] in "\"'":
            candidate = candidate[1:-1].strip()
        return candidate
