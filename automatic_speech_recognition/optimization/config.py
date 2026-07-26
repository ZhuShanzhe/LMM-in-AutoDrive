import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class Config:
    """
    Configuration class for the TTS pipeline.
    All attributes are lower_case with underscores.
    """
    model_dir: str = "pretrained_models/CosyVoice2-0.5B"
    default_speaker: str = "中文女"  # Supported: 中文女, 中文男, 英文女, 英文男, etc.

    sample_rate: int = 16000

    output_dir: str = "outputs"

    noise_enabled: bool = True
    noise_type: str = "traffic"  # white, traffic, cafe, rain
    noise_snr_db: float = 15.0  # Signal-to-noise ratio in dB
    noise_dir: str = "noise_samples"

    num_beams: int = 4
    temperature: float = 0.7
    top_p: float = 0.9

    @classmethod
    def from_env(cls, prefix: str = "COSYVOICE_") -> "Config":
        """
        Create a Config instance from environment variables.
        Environment variable names are uppercase with the given prefix.
        Example: COSYVOICE_MODEL_DIR, COSYVOICE_DEFAULT_SPEAKER, etc.
        """
        kwargs = {}
        for field_name in cls.__dataclass_fields__:
            env_var = f"{prefix}{field_name.upper()}"
            if env_var in os.environ:
                value = os.environ[env_var]
                # Attempt to convert to the correct type (int, float, bool)
                field_type = cls.__dataclass_fields__[field_name].type
                if field_type is bool:
                    value = value.lower() in ("true", "1", "yes")
                elif field_type is int:
                    value = int(value)
                elif field_type is float:
                    value = float(value)
                kwargs[field_name] = value
        return cls(**kwargs)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create a Config instance from a dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def update(self, **kwargs) -> None:
        """Update configuration attributes in-place."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
