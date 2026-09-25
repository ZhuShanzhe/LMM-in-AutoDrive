import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import DEFAULT_NUM_GPUS, resolve_device

from .utils import load_yaml, save_json

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
_LOADED: Dict[Tuple[str, str], Tuple] = {}

class Qwen3TranslatorService:

    SYSTEM_PROMPT = (
        "You are a professional translator for autonomous-driving commands. "
        "Translate the user text into the requested target language. "
        "Keep numbers, units, left/right, direction words and object words exact. "
        "Output only the translation, no explanations."
    )

    def __init__(
        self,
        model_id_or_path: str = "models/Qwen3-1.7B",
        device_map: Optional[str] = None,
        num_gpus: int = DEFAULT_NUM_GPUS,
        dtype: str = "bfloat16",
        attn_implementation: str = "",
        source_lang: str = "Chinese",
        target_lang: str = "English",
        generation_max_length: int = 512,
        enable_thinking: bool = False,
        do_sample: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        output_dir: str = "outputs",
        raise_on_error: bool = False,
        gen_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model_id_or_path = model_id_or_path
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.generation_max_length = generation_max_length
        self.enable_thinking = enable_thinking
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.output_dir = output_dir
        self.raise_on_error = raise_on_error
        self.gen_kwargs = dict(gen_kwargs or {})
        self.model, self.tokenizer = self._load(model_id_or_path, device_map or resolve_device(num_gpus), dtype, attn_implementation)

    @classmethod
    def from_yaml(cls, path: str) -> "Qwen3TranslatorService":
        cfg = load_yaml(path)
        params = {
            "model_id_or_path", "device_map", "num_gpus", "dtype", "attn_implementation",
            "source_lang", "target_lang", "generation_max_length",
            "enable_thinking", "do_sample", "temperature", "top_p",
            "output_dir", "raise_on_error", "gen_kwargs",
        }
        known = {k: v for k, v in cfg.items() if k in params}
        unknown = [k for k in cfg if k not in params]
        if unknown:
            logger.warning("ignoring unknown config keys: %s", unknown)
        return cls(**known)

    @staticmethod
    def _load(model_path: str, device_map: str, dtype: str, attn_implementation: str) -> Tuple:
        cache_key = (model_path, str(device_map))
        if cache_key not in _LOADED:
            torch_dtype = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }.get(dtype, "auto")
            kwargs: Dict[str, Any] = {"torch_dtype": torch_dtype, "trust_remote_code": True}
            if device_map:
                kwargs["device_map"] = device_map
            if attn_implementation:
                kwargs["attn_implementation"] = attn_implementation

            logger.info("Loading Qwen3 translator from %s ...", model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
            model.eval()
            _LOADED[cache_key] = (model, tokenizer)
            logger.info("Qwen3 translator ready.")
        return _LOADED[cache_key]

    def _messages(self, text: str, src: str, tgt: str) -> List[Dict[str, str]]:
        instruction = (
            f"Translate the following {src} text into {tgt}. "
            f"Output only the translation:\n\n{text}"
        )
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ]

    def _generate_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "max_new_tokens": self.generation_max_length,
            "pad_token_id": self.tokenizer.eos_token_id,
            "do_sample": self.do_sample,
        }
        if self.do_sample:
            if self.temperature is not None:
                kwargs["temperature"] = self.temperature
            if self.top_p is not None:
                kwargs["top_p"] = self.top_p
        kwargs.update(self.gen_kwargs)
        return kwargs

    @staticmethod
    def _strip_think(text: str) -> str:
        return _THINK_RE.sub("", text).strip()

    def _translate_one(self, text: str, src: str, tgt: str) -> Dict[str, Any]:
        start = time.perf_counter()
        try:
            if not text or not text.strip():
                out = ""
            else:
                messages = self._messages(text.strip(), src, tgt)
                apply_kwargs: Dict[str, Any] = {
                    "tokenize": True,
                    "add_generation_prompt": True,
                    "return_tensors": "pt",
                    "return_dict": True,
                }
                if not self.enable_thinking:
                    apply_kwargs["enable_thinking"] = False
                try:
                    inputs = self.tokenizer.apply_chat_template(messages, **apply_kwargs)
                except TypeError:
                    inputs = self.tokenizer.apply_chat_template(messages, **{k: v for k, v in apply_kwargs.items() if k != "enable_thinking"})
                device = next(self.model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                input_len = inputs["input_ids"].shape[-1]
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, **self._generate_kwargs())
                out = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
                out = self._strip_think(out)
            ok = bool(out)
        except Exception as exc:
            logger.error("translation failed: %s", exc)
            if self.raise_on_error: raise
            out, ok = "", False
        elapsed = round(time.perf_counter() - start, 4)
        return {"source": text, "translation": out, "success": ok, "source_lang": src, "target_lang": tgt, "time_seconds": elapsed}

    def _run(self, text: Union[str, List[str]], src: Optional[str], tgt: Optional[str]) -> Tuple[Union[str, List[str]], Dict[str, Any]]:
        src = src or self.source_lang
        tgt = tgt or self.target_lang
        if isinstance(text, str):
            meta = self._translate_one(text, src, tgt)
            return meta["translation"], meta
        metas = [self._translate_one(t, src, tgt) for t in text]
        merged = {
            "source": [m["source"] for m in metas],
            "translation": [m["translation"] for m in metas],
            "success": [m["success"] for m in metas],
            "time_seconds": [m["time_seconds"] for m in metas],
            "source_lang": src,
            "target_lang": tgt,
            "count": len(metas),
            "total_time_seconds": round(sum(m["time_seconds"] for m in metas), 4),
        }
        return [m["translation"] for m in metas], merged

    def translate(
        self, text: Union[str, List[str]],
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        output_json: Optional[str] = None,
        return_meta: bool = False
    ):
        """Translate one string or a list of strings.

        Args:
            text: text or list of texts to translate.
            source_lang / target_lang: override defaults.
            output_json: optional JSON result file path.
            return_meta: if True, also return the metadata dict.

        Returns:
            plain result (str / List[str]) by default, or (result, meta) when return_meta=True.
        """
        result, meta = self._run(text, source_lang, target_lang)
        if output_json:
            self.save_result(meta, output_json)
        if return_meta:
            return result, meta
        return result

    def translate_zh_to_en(self, text: Union[str, List[str]], output_json: Optional[str] = None, return_meta: bool = False):
        return self.translate(text, "Chinese", "English", output_json=output_json, return_meta=return_meta)

    def translate_en_to_zh(self, text: Union[str, List[str]], output_json: Optional[str] = None, return_meta: bool = False):
        return self.translate(text, "English", "Chinese", output_json=output_json, return_meta=return_meta)

    def translate_commands(self, input_file: str, output_json: Optional[str] = None, text_key: str = "text", **kwargs) -> List[Dict[str, Any]]:
        with open(input_file, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            items = [{"index": i + 1, "text": t} for i, t in enumerate(data)]
        else:
            items = data
        texts = [str(x.get(text_key, "")) for x in items]
        results, _ = self.translate(texts, **kwargs)
        records = []
        for item, res in zip(items, results):
            rec = dict(item)
            rec["translation"] = res
            records.append(rec)
        if output_json:
            save_json({"records": records}, output_json)
        return records

    def save_result(self, meta: Dict[str, Any], path: str) -> None:
        save_json({"model": self.model_id_or_path, **meta}, path)
        logger.info("translation results saved -> %s", path)
