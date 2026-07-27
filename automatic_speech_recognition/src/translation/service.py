import time
import logging
import json
from typing import List, Optional, Union, Any
from pathlib import Path

from .translator import Translator
from .config import ModelConfig

logger = logging.getLogger(__name__)


class Translation:
    """
    High-level translation service with convenient defaults and multiple input modes.
    Designed for easy integration into engineering projects.
    """

    def __init__(
        self,
        translator: Optional[Translator] = None,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        src_lang: str = "eng_Latn",
        tgt_lang: str = "zho_Hans",
        max_length: int = 512,
        load_type: str = "custom",
        model_path: Optional[str] = None,
        generation_max_length: Optional[int] = None,
        num_beams: int = 1,
        temperature: float = 0.1,
        raise_on_error: bool = False,
    ):
        """
        Initialize the translation service.

        Args:
            translator: Optional pre-configured Translator instance.
            model_name: Hugging Face model identifier.
            src_lang: Default source language code.
            tgt_lang: Default target language code.
            max_length: Max input token length.
            load_type: 'custom' (Hugging Face) or 'local' (local directory).
            model_path: Required if load_type='local'.
            generation_max_length: Max output token length.
            num_beams: Beam search width (1 for greedy).
            temperature: Sampling temperature.
            raise_on_error: If True, exceptions are raised; if False, return empty string.
        """
        self.default_src_lang = src_lang
        self.default_tgt_lang = tgt_lang
        self.raise_on_error = raise_on_error

        # Timing attributes
        self.last_translation_time = 0.0
        self.last_batch_time = 0.0

        if translator is None:
            config = ModelConfig(
                model_name=model_name,
                load_type=load_type,
                model_path=model_path,
                max_length=max_length,
            )
            self.translator = Translator(
                config=config,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                max_length=max_length,
                generation_max_length=generation_max_length,
                num_beams=num_beams,
                temperature=temperature,
            )
        else:
            self.translator = translator

    def _translate_with_timing(
        self,
        text: str,
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None,
        **gen_kwargs: Any,
    ) -> tuple[str, float]:
        """Translate a single text and measure elapsed time."""
        src = src_lang if src_lang is not None else self.default_src_lang
        tgt = tgt_lang if tgt_lang is not None else self.default_tgt_lang

        start = time.perf_counter()
        try:
            result = self.translator.translate(text, src, tgt, **gen_kwargs)
            if result is None:
                if self.raise_on_error:
                    raise RuntimeError(f"Translation returned None for text: {text[:50]}...")
                result = ""
        except Exception as e:
            elapsed = time.perf_counter() - start
            logger.error(f"Translation failed: {e}")
            if self.raise_on_error:
                raise
            return "", elapsed

        elapsed = time.perf_counter() - start
        self.last_translation_time = elapsed
        return result, elapsed

    def translate(
        self,
        text: Union[str, List[str]],
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None,
        output_file: Optional[str] = None,
        output_mode: str = "w",
        json_output_path: Optional[str] = None,
        **gen_kwargs: Any,
    ) -> Union[str, List[str]]:
        """
        Unified translation entry point.

        Args:
            text: Input text (str) or list of texts.
            src_lang: Source language (default from __init__).
            tgt_lang: Target language (default from __init__).
            output_file: If provided, writes result(s) to a plain text file.
            output_mode: File write mode: 'w' (overwrite) or 'a' (append).
            json_output_path: If provided, writes structured JSON with results and timing.
            **gen_kwargs: Additional generation parameters.

        Returns:
            Translated string or list of strings.
        """
        if isinstance(text, str):
            result = self.translate_text(
                text, src_lang, tgt_lang, output_file, output_mode,
                json_output_path, **gen_kwargs
            )
            return result
        elif isinstance(text, list):
            results = self.translate_list(
                text, src_lang, tgt_lang, output_file, output_mode,
                json_output_path, **gen_kwargs
            )
            return results
        else:
            raise TypeError(f"Unsupported input type: {type(text)}. Expected str or List[str].")

    def translate_text(
        self,
        text: str,
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None,
        output_file: Optional[str] = None,
        output_mode: str = "w",
        json_output_path: Optional[str] = None,
        **gen_kwargs: Any,
    ) -> str:
        """
        Translate a single text string.

        Args:
            text: Text to translate.
            src_lang: Source language (default from __init__).
            tgt_lang: Target language (default from __init__).
            output_file: Optional plain text output file.
            output_mode: File write mode.
            json_output_path: Optional JSON output file for structured result.
            **gen_kwargs: Additional generation parameters.

        Returns:
            Translated text.
        """
        result, elapsed = self._translate_with_timing(text, src_lang, tgt_lang, **gen_kwargs)

        # Plain text output
        if output_file:
            self._write_to_file(output_file, result, mode=output_mode)

        # JSON output
        if json_output_path:
            data = {
                "source": text,
                "translation": result,
                "time_seconds": elapsed
            }
            self._write_json_file(json_output_path, data)

        return result

    def translate_list(
        self,
        texts: List[str],
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None,
        output_file: Optional[str] = None,
        output_mode: str = "w",
        json_output_path: Optional[str] = None,
        **gen_kwargs: Any,
    ) -> List[str]:
        """
        Translate a list of texts sequentially.

        Args:
            texts: List of texts to translate.
            src_lang: Source language (default from __init__).
            tgt_lang: Target language (default from __init__).
            output_file: Optional plain text output file.
            output_mode: File write mode.
            json_output_path: Optional JSON output file for structured result.
            **gen_kwargs: Additional generation parameters.

        Returns:
            List of translated strings.
        """
        results = []
        details = []          # for JSON
        total_start = time.perf_counter()
        total = len(texts)

        for i, text in enumerate(texts, 1):
            print(f"[{i}/{total}] sentence translating...")
            result, elapsed = self._translate_with_timing(text, src_lang, tgt_lang, **gen_kwargs)
            results.append(result)
            details.append({
                "source": text,
                "translation": result,
                "time_seconds": elapsed
            })
            logger.info(f"Translated item {i}/{total} in {elapsed:.3f}s")

        total_elapsed = time.perf_counter() - total_start
        self.last_batch_time = total_elapsed
        logger.info(f"Batch translation finished in {total_elapsed:.3f}s total")

        # Plain text output
        if output_file:
            content = "\n".join(results)
            self._write_to_file(output_file, content, mode=output_mode)

        # JSON output
        if json_output_path:
            json_data = {
                "summary": {
                    "total_time_seconds": total_elapsed,
                    "count": total
                },
                "details": details
            }
            self._write_json_file(json_output_path, json_data)

        return results

    def translate_file(
        self,
        input_file: str,
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None,
        output_file: Optional[str] = None,
        *,
        line_by_line: bool = True,
        encoding: str = "utf-8",
        output_mode: str = "w",
        json_output_path: Optional[str] = None,
        **gen_kwargs: Any,
    ) -> List[str]:
        """
        Translate the content of a file.

        Args:
            input_file: Path to input file.
            src_lang: Source language (default from __init__).
            tgt_lang: Target language (default from __init__).
            output_file: Optional plain text output file.
            line_by_line: If True, translate each line separately; else whole file as one block.
            encoding: File encoding.
            output_mode: File write mode.
            json_output_path: Optional JSON output file for structured result.
            **gen_kwargs: Additional generation parameters.

        Returns:
            List of translated lines (or single element if not line_by_line).
        """
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        with open(input_path, "r", encoding=encoding) as f:
            content = f.read()

        if line_by_line:
            lines = content.splitlines()
            results = []
            details = []
            total_start = time.perf_counter()
            total = len(lines)

            for i, line in enumerate(lines, 1):
                print(f"[{i}/{total}] sentence translating...")
                result, elapsed = self._translate_with_timing(line, src_lang, tgt_lang, **gen_kwargs)
                results.append(result)
                details.append({
                    "source": line,
                    "translation": result,
                    "time_seconds": elapsed
                })
                logger.info(f"Translated line {i}/{total} in {elapsed:.3f}s")

            total_elapsed = time.perf_counter() - total_start
            self.last_batch_time = total_elapsed
            logger.info(f"File translation (line-by-line) finished in {total_elapsed:.3f}s")
            output_lines = results

            # JSON output for line-by-line
            if json_output_path:
                json_data = {
                    "summary": {
                        "total_time_seconds": total_elapsed,
                        "count": total
                    },
                    "details": details
                }
                self._write_json_file(json_output_path, json_data)

        else:
            # Whole file as one block
            result, elapsed = self._translate_with_timing(content, src_lang, tgt_lang, **gen_kwargs)
            logger.info(f"Whole file translated in {elapsed:.3f}s")
            self.last_batch_time = elapsed
            output_lines = [result]

            # JSON output for whole block
            if json_output_path:
                json_data = {
                    "source": content,
                    "translation": result,
                    "time_seconds": elapsed
                }
                self._write_json_file(json_output_path, json_data)

        # Plain text output
        if output_file is None:
            stem = input_path.stem
            suffix = input_path.suffix
            output_file = f"{stem}_translated{suffix}"

        self._write_to_file(output_file, "\n".join(output_lines), mode=output_mode)

        return output_lines

    @staticmethod
    def _write_to_file(filepath: str, content: str, mode: str = "w") -> None:
        """Write content to a plain text file with UTF-8 encoding."""
        with open(filepath, mode, encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Output written to: {filepath}")

    @staticmethod
    def _write_json_file(filepath: str, data: dict) -> None:
        """Write structured data to a JSON file with UTF-8 encoding."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"JSON output written to: {filepath}")
