import logging
from typing import Any, Dict, List

import librosa
import torch
from torch.utils.data import Dataset

logger = logging.getLogger("training")

class ASRManifestDataset(Dataset):
    """Load audio at the given sample rate and build ASR chat messages."""

    def __init__(
        self, 
        rows: List[Dict[str, Any]],
        prompt: str = "Transcribe the audio into Chinese text.",
        sample_rate: int = 16000
    ):
        self.rows = rows
        self.prompt = prompt
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        audio_path = row.get("audio") or row.get("audio_file")
        audio, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        text = row.get("text", "")
        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": [{"type": "audio", "audio": audio}]},
            {"role": "assistant", "content": text},
        ]
        return {"messages": messages, "text": text}

class ASRCollator:
    """Build model inputs from chat messages and mask the prompt labels."""

    def __init__(self, processor: Any, max_length: int = 2048):
        self.processor = processor
        self.max_length = max_length
        if not hasattr(self.processor, "apply_chat_template"):
            raise AttributeError("processor has no apply_chat_template; check the qwen-asr version")

    def _apply(self, messages, add_generation_prompt: bool) -> Dict[str, torch.Tensor]:
        return self.processor.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=add_generation_prompt,
            return_dict=True, 
            return_tensors="pt",
        )

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        inputs_list: List[Dict[str, torch.Tensor]] = []
        labels_list: List[torch.Tensor] = []

        for item in batch:
            messages = item["messages"]
            full = self._apply(messages, add_generation_prompt=False)
            prompt_only = self._apply(messages[:-1], add_generation_prompt=True)

            input_ids = full["input_ids"][0][: self.max_length]
            labels = input_ids.clone()
            prompt_len = min(prompt_only["input_ids"].shape[-1], input_ids.shape[-1])
            labels[:prompt_len] = -100

            feat = {k: v[0][: self.max_length] for k, v in full.items() if isinstance(v, torch.Tensor)}
            inputs_list.append(feat)
            labels_list.append(labels)

        pad_token_id = getattr(self.processor.tokenizer, "pad_token_id", 0) or 0
        max_len = max(f["input_ids"].shape[-1] for f in inputs_list)
        keys = inputs_list[0].keys()

        out: Dict[str, torch.Tensor] = {}
        for key in keys:
            if key == "labels":
                continue
            pad_value = pad_token_id if "input_ids" in key else 0
            padded = torch.full((len(inputs_list), max_len), pad_value, dtype=torch.long)
            for i, feat in enumerate(inputs_list):
                seq = feat[key]
                padded[i, : seq.shape[-1]] = seq
            out[key] = padded

        out["labels"] = torch.full((len(labels_list), max_len), -100, dtype=torch.long)
        for i, lab in enumerate(labels_list):
            out["labels"][i, : lab.shape[-1]] = lab

        if "attention_mask" not in out:
            out["attention_mask"] = (out["input_ids"] != pad_token_id).long()
        return out