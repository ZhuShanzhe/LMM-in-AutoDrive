from typing import Optional, Dict, Any


class ChatTTSConfig:
    """Configuration for ChatTTS TTS service."""

    def __init__(
        self,
        model_path: str = "./models/rvcmd_linux_amd64",
        sample_rate: int = 24000,
        batch_size: int = 8,
        compile_model: bool = True,
        output_dir: str = "outputs",
        # 噪声参数（默认关闭）
        noise_enabled: bool = False,
        noise_type: str = "white",          # 'white', 'pink', 'brown', 'from_file'
        noise_level: float = 0.005,
        noise_file: Optional[str] = None,
        **extra_kwargs,
    ):
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.compile_model = compile_model
        self.output_dir = output_dir
        self.noise_enabled = noise_enabled
        self.noise_type = noise_type
        self.noise_level = noise_level
        self.noise_file = noise_file
        self.extra_kwargs = extra_kwargs

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatTTSConfig":
        known_params = {
            "model_path", "sample_rate", "batch_size", "compile_model",
            "output_dir", "noise_enabled", "noise_type", "noise_level", "noise_file"
        }
        known_data = {k: v for k, v in data.items() if k in known_params}
        extra_data = {k: v for k, v in data.items() if k not in known_params}
        return cls(**known_data, **extra_data)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "model_path": self.model_path,
            "sample_rate": self.sample_rate,
            "batch_size": self.batch_size,
            "compile_model": self.compile_model,
            "output_dir": self.output_dir,
            "noise_enabled": self.noise_enabled,
            "noise_type": self.noise_type,
            "noise_level": self.noise_level,
            "noise_file": self.noise_file,
        }
        if self.extra_kwargs:
            result["extra_kwargs"] = self.extra_kwargs
        return result

    def update(self, **kwargs) -> "ChatTTSConfig":
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.extra_kwargs[key] = value
        return self
