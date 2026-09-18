import argparse
import json
import logging
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import yaml
from openai import OpenAI, RateLimitError

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import log_and_print, setup_logging


class QwenMTTranslator:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "qwen-mt-plus",
        source_lang: str = "auto",
        target_lang: str = "Chinese",
        max_retries: int = 3,
        retry_base_delay: float = 2.0,
        request_interval: float = 0.0,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.request_interval = request_interval

    def _rate_limit_wait(self, attempt: int) -> float:
        base = min(self.retry_base_delay * (5.0 ** attempt), 60.0)
        return base + random.uniform(0.0, 1.0)

    def translate_one(self, text: str) -> str:
        last_error: Exception = RuntimeError("unknown error")
        for attempt in range(1, self.max_retries + 1):
            if self.request_interval > 0:
                time.sleep(self.request_interval)
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": text}],
                    extra_body={
                        "translation_options": {
                            "source_lang": self.source_lang,
                            "target_lang": self.target_lang,
                        }
                    },
                )
                translated = (completion.choices[0].message.content or "").strip()
                if not translated:
                    raise RuntimeError("empty translation returned")
                return translated
            except RateLimitError as exc:
                last_error = exc
                wait = self._rate_limit_wait(attempt)
                if attempt < self.max_retries:
                    log_and_print(
                        f"[translation] rate limited, waiting {wait:.1f}s "
                        f"(attempt {attempt}/{self.max_retries})",
                        level=logging.WARNING)
                    time.sleep(wait)
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_base_delay * (2 ** (attempt - 1)))
        raise RuntimeError(f"translate failed after {self.max_retries} attempts: {last_error}")

    def translate_unique(self, unique_texts: List[str], concurrency: int, progress, checkpoint_path: str = None, save_interval: int = 10) -> Dict[str, str]:
        pending = [t for t in unique_texts if t not in progress]
        log_and_print(f"[translation] unique={len(unique_texts)} pending={len(pending)}")
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(self.translate_one, t): t for t in pending}
            done = 0
            for future in as_completed(futures):
                text = futures[future]
                try:
                    progress[text] = future.result()
                except Exception as exc:
                    log_and_print(f"[translation] FAILED: {text!r} -> {exc}",
                                  level=logging.ERROR)
                done += 1
                if checkpoint_path and done % save_interval == 0:
                    save_checkpoint(checkpoint_path, progress)
                    log_and_print(f"[translation] checkpoint saved ({done}/{len(pending)})")
                log_and_print(f"[translation] progress {done}/{len(pending)}")
            if checkpoint_path:
                save_checkpoint(checkpoint_path, progress)
        return progress

def load_english_commands(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list) and all(isinstance(x, str) for x in data):
        return [x.strip() for x in data if x and x.strip()]
    raise ValueError(f"English-commands.json ERROR: not a pure string array")

def load_checkpoint(path: str) -> Dict[str, str]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_checkpoint(path: str, cache: Dict[str, str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def write_chinese_list(records: List[Dict[str, str]], path: str) -> None:
    lines: List[str] = [rec.get("translation", "") for rec in records]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(lines, fh, ensure_ascii=False, indent=2)
    log_and_print(f"[translation] wrote flat Chinese list -> {path} ({len(lines)} entries)")

def main() -> None:
    parser = argparse.ArgumentParser(description="Translate English commands to Chinese")
    parser.add_argument("--config", default="configs/translation/qwen_mt.yaml")
    parser.add_argument("--api-key", default=None, help="API Key")
    parser.add_argument("--input-file", default=None)
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--log-file", default=None, help="log file path; empty string disables it")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    log_file = args.log_file if args.log_file is not None else cfg.get("log_file")
    setup_logging(log_file, level=getattr(logging, args.log_level))

    api_key = args.api_key or os.getenv(cfg.get("api_key_env", "DASHSCOPE_API_KEY"))
    if not api_key:
        raise SystemExit("no API Key found")

    input_file = args.input_file or cfg["input_file"]
    output_file = args.output_file or cfg["output_file"]
    checkpoint_file = cfg.get("checkpoint_file", output_file + ".cache.json")

    texts = load_english_commands(input_file)
    log_and_print(f"[translation] loaded {len(texts)} English commands from {input_file}")

    translator = QwenMTTranslator(
        api_key=api_key,
        base_url=cfg.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        model=cfg.get("model", "qwen-mt-plus"),
        source_lang=cfg.get("source_lang", "auto"),
        target_lang=cfg.get("target_lang", "Chinese"),
        max_retries=cfg.get("max_retries", 6),
        retry_base_delay=cfg.get("retry_base_delay", 2.0),
        request_interval=cfg.get("request_interval", 1.0),
    )

    cache = load_checkpoint(checkpoint_file)
    log_and_print(f"[translation] resume from cache: {len(cache)} items")
    unique_texts = list(dict.fromkeys(texts))
    cache = translator.translate_unique(
        unique_texts, cfg.get("concurrency", 1), cache,
        checkpoint_path=checkpoint_file, save_interval=cfg.get("save_interval", 10))
    save_checkpoint(checkpoint_file, cache)

    records = []
    missing = 0
    for idx, original in enumerate(texts, 1):
        zh = cache.get(original, "")
        if not zh:
            missing += 1
            log_and_print(f"[translation] WARNING: no Chinese for index {idx}: {original!r}", level=logging.WARNING)
        records.append({
            "index": idx,
            "original": original,
            "translation": zh,
        })

    write_chinese_list(records, output_file)
    log_and_print(f"[translation] done -> {output_file} "
                  f"({len(records)} entries, {missing} missing)")

if __name__ == "__main__":
    main()
