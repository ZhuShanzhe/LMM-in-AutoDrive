from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class DenoiserConfig:
    """
    Configuration for DeepFilterNet speech enhancement.
    """
    model_name: str = "DeepFilterNet3"
    device: Optional[str] = None

    output_sr: int = 16000
    output_dir: str = "data"

    extra: Dict[str, Any] = field(default_factory=dict)
