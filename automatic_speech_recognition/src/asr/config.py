from typing import Optional, Dict, Any


class FunASRConfig:
    # Default model parameters for each mode
    _DEFAULT_MODELS = {
        "single": {
            "model": "paraformer-zh",
            "vad_model": "fsmn-vad",
            "punc_model": "ct-punc",
            "spk_model": None,
        },
        "speaker": {
            "model": "iic/SenseVoiceSmall",
            "vad_model": "fsmn-vad",
            "punc_model": "ct-punc",
            "spk_model": "cam++",
        }
    }

    def __init__(
        self,
        mode: str = "single",
        device: str = "cuda:0",
        model: Optional[str] = None,
        vad_model: Optional[str] = None,
        punc_model: Optional[str] = None,
        spk_model: Optional[str] = None,
        **extra_kwargs: Any,
    ):
        """
        Initialize FunASR configuration.

        Args:
            mode: Recognition mode, either 'single' or 'speaker'.
            device: Device to run on, e.g., 'cuda:0' or 'cpu'.
            model: ASR model name (overrides default for the mode).
            vad_model: VAD model name.
            punc_model: Punctuation model name.
            spk_model: Speaker diarization model name (used only in 'speaker' mode).
            **extra_kwargs: Additional parameters passed to AutoModel.
        """
        if mode not in self._DEFAULT_MODELS:
            raise ValueError(f"Invalid mode: {mode}. Choose from {list(self._DEFAULT_MODELS.keys())}")

        self.mode = mode
        self.device = device

        default_config = self._DEFAULT_MODELS[mode].copy()

        if model is not None:
            default_config["model"] = model
        if vad_model is not None:
            default_config["vad_model"] = vad_model
        if punc_model is not None:
            default_config["punc_model"] = punc_model
        if spk_model is not None:
            default_config["spk_model"] = spk_model

        self._model_config = default_config
        self._model_config.update(extra_kwargs)  # Allow extra keys

    def get_model_config(self) -> Dict[str, Any]:
        """Return the configuration dict to be passed to AutoModel."""
        return self._model_config.copy()

    def get_device(self) -> str:
        """Return the device string."""
        return self.device

    def get_mode(self) -> str:
        """Return the recognition mode."""
        return self.mode
