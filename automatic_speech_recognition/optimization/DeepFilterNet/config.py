from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any


MODEL_ROOT = Path(__file__).resolve().parents[3] / "models"
DEFAULT_DEEPFILTER_MODEL_PATH = str(
    MODEL_ROOT / "pretrained" / "DeepFilterNet3"
)


@dataclass
class DenoiserConfig:
    """
    Configuration for DeepFilterNet speech enhancement.
    """
    model_name: str = DEFAULT_DEEPFILTER_MODEL_PATH
    device: Optional[str] = None

    output_sr: int = 16000
    output_dir: str = "data"

    extra: Dict[str, Any] = field(default_factory=dict)
