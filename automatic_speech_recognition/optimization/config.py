from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class AugmentConfig:
    """Configuration for audio noise augmentation."""

    noise_type: str = "white"          # "white", "from_file", "vehicle"
    noise_file: Optional[str] = None   # Path to custom noise WAV (if noise_type == "from_file")
    snr_db: float = 20.0               # Signal-to-noise ratio in dB

    sample_rate: int = 16000           # Target sample rate (resamples if needed)
    output_dir: str = "augmented"      # Output directory for augmented files
    seed: Optional[int] = None         # Random seed for reproducibility
    copy_original: bool = False        # Whether to copy original files to output

    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.seed is not None:
            import random
            import numpy as np
            random.seed(self.seed)
            np.random.seed(self.seed)
