import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelConfig:
    """
    Qwen2.5-3B-Instruct model configuration and loader.
    Supports loading from Hugging Face Hub or local directory.
    """
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        load_type: str = "custom",  # 'custom' (Hugging Face) or 'local'
        model_path: str = None,
        device: str = None,
        max_length: int = 512,
    ):
        """
        Args:
            model_name: Hugging Face model identifier (used when load_type='custom')
            load_type: 'custom' for Hugging Face Hub, 'local' for local directory
            model_path: Required if load_type='local'
            device: 'cuda', 'cpu', or None (auto-detect)
            max_length: Maximum input token length
        """
        if load_type == "local" and model_path is None:
            raise ValueError("model_path must be provided when load_type='local'.")

        self.model_name = model_name if load_type == "custom" else model_path
        self.load_type = load_type
        self.model_path = model_path
        self.max_length = max_length
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        self._model = None
        self._tokenizer = None

    def load_model_and_tokenizer(self):
        """Load model and tokenizer from Hugging Face or local path."""
        if self._model is None or self._tokenizer is None:
            print(f"Loading model: {self.model_name}, device: {self.device}")

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )

            # Ensure pad_token is set for generation
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
            )

            if self.device == "cpu":
                self._model = self._model.to("cpu")

            print("Model loaded successfully!")
        return self._model, self._tokenizer

    def get_model(self):
        if self._model is None:
            self.load_model_and_tokenizer()
        return self._model

    def get_tokenizer(self):
        if self._tokenizer is None:
            self.load_model_and_tokenizer()
        return self._tokenizer
