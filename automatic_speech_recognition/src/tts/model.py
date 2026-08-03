import ChatTTS
from typing import List, Any
from .config import ChatTTSConfig


class ChatTTSModel:
    """Wrapper for ChatTTS model, using your tested loading method."""

    def __init__(self, config: ChatTTSConfig):
        self.config = config
        self._chat = None

    def _load_model(self):
        if self._chat is None:
            chat = ChatTTS.Chat()
            chat.load(
                source="local",
                custom_path=self.config.model_path,
                compile=self.config.compile_model,
            )
            self._chat = chat
            print("ChatTTS model loaded successfully.")
        return self._chat

    def synthesize(self, texts: List[str], **infer_kwargs) -> List[Any]:
        """Infer speech from texts, returns list of numpy arrays."""
        if not texts:
            return []
        chat = self._load_model()
        wavs = chat.infer(texts, **infer_kwargs)
        return wavs
