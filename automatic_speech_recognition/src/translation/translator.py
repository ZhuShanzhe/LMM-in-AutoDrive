import torch
from typing import Optional, Any
from .config import ModelConfig


class Translator:
    """
    Qwen2.5-3B-Instruct translator supporting bidirectional English-Chinese translation.
    Uses chat template for instruction-style prompts.
    """

    # System prompt for translation tasks
    SYSTEM_PROMPT = "You are a professional translator. Translate the following text accurately and naturally. Only output the translation, no additional explanations."

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        src_lang: str = "eng_Latn",
        tgt_lang: str = "zho_Hans",
        max_length: int = 512,
        generation_max_length: Optional[int] = None,
        num_beams: int = 1,
        temperature: float = 0.1,
        top_p: float = 0.9,
    ):
        """
        Initialize the translator.

        Args:
            config: ModelConfig instance. If None, creates default.
            src_lang: Default source language (for reference only).
            tgt_lang: Default target language (for reference only).
            max_length: Max input token length.
            generation_max_length: Max output token length.
            num_beams: Number of beams for beam search (1 for greedy).
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
        """
        if config is None:
            config = ModelConfig()

        self.config = config
        self.model = self.config.get_model()
        self.tokenizer = self.config.get_tokenizer()
        self.device = self.config.device
        self.max_length = max_length
        self.generation_max_length = generation_max_length or 512
        self.num_beams = num_beams
        self.temperature = temperature
        self.top_p = top_p

        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def _build_translation_prompt(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """
        Build translation prompt with language direction specified.
        """
        # Map language codes to readable names
        lang_map = {
            "eng_Latn": "English",
            "zho_Hans": "Chinese",
            "zho_Hant": "Traditional Chinese",
            "fra_Latn": "French",
            "spa_Latn": "Spanish",
            "deu_Latn": "German",
            "jpn_Jpan": "Japanese",
            "kor_Hang": "Korean",
            "rus_Cyrl": "Russian",
        }

        src_name = lang_map.get(src_lang, src_lang)
        tgt_name = lang_map.get(tgt_lang, tgt_lang)

        if src_lang == "eng_Latn" and tgt_lang == "zho_Hans":
            instruction = f"Translate the following English text into Chinese. Only output the translation, no explanations."
        elif src_lang == "zho_Hans" and tgt_lang == "eng_Latn":
            instruction = f"Translate the following Chinese text into English. Only output the translation, no explanations."
        else:
            instruction = f"Translate the following text from {src_name} to {tgt_name}. Only output the translation, no explanations."

        return f"{instruction}\n\nText: {text}\n\nTranslation:"

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        **gen_kwargs: Any,
    ) -> Optional[str]:
        """
        Translate text from source_lang to target_lang.

        Args:
            text: Text to translate.
            source_lang: Source language code (e.g., 'eng_Latn').
            target_lang: Target language code (e.g., 'zho_Hans').
            **gen_kwargs: Additional generation parameters.

        Returns:
            Translated text, or None on failure.
        """
        if not text or not text.strip():
            return ""

        try:
            # Build prompt
            prompt = self._build_translation_prompt(text, source_lang, target_lang)

            # Apply chat template
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]

            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.device)

            # Prepare generation parameters
            gen_params = {
                "max_new_tokens": self.generation_max_length,
                "num_beams": self.num_beams,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "do_sample": self.num_beams == 1 and self.temperature > 0,
                "pad_token_id": self.tokenizer.eos_token_id,
            }
            gen_params.update(gen_kwargs)

            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_params)

            # Decode only the generated part (skip input)
            generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
            translation = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            return translation.strip()

        except Exception as e:
            print(f"Translation request failed: {e}")
            return None

    def translate_en_to_zh(self, text: str, **gen_kwargs: Any) -> Optional[str]:
        """Translate English text to Simplified Chinese."""
        return self.translate(text, "eng_Latn", "zho_Hans", **gen_kwargs)

    def translate_zh_to_en(self, text: str, **gen_kwargs: Any) -> Optional[str]:
        """Translate Simplified Chinese text to English."""
        return self.translate(text, "zho_Hans", "eng_Latn", **gen_kwargs)
