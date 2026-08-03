from typing import List, Dict, Any, Optional
from funasr import AutoModel

from .config import FunASRConfig


class FunASRModel:
    def __init__(self, config: Optional[FunASRConfig] = None):
        """
        Initialize the model with a configuration.

        Args:
            config: FunASRConfig instance. If None, creates a default config.
        """
        if config is None:
            config = FunASRConfig()

        self.config = config
        self._model = None  # Lazy loading

    def _load_model(self) -> AutoModel:
        """Load the model if not already loaded."""
        if self._model is None:
            model_config = self.config.get_model_config()
            device = self.config.get_device()
            print(f"Loading FunASR model with config: {model_config}, device: {device}")
            self._model = AutoModel(**model_config, device=device)
            print("Model loaded successfully.")
        return self._model

    def generate(self, audio_path: str, **kwargs: Any) -> List[Dict[str, Any]]:
        """
        Perform speech recognition on an audio file.

        Args:
            audio_path: Path to the input audio file (WAV recommended).
            **kwargs: Additional arguments for model.generate() (e.g., hotword, batch_size_s).

        Returns:
            A list of result dictionaries (typically one element per audio file).
        """
        if not audio_path:
            raise ValueError("audio_path cannot be empty.")

        model = self._load_model()
        results = model.generate(input=audio_path, **kwargs)
        return results
