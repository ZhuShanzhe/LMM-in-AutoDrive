import os
import time
from typing import Optional


def generate_timestamp_filename(prefix: str = "recording", ext: str = "wav") -> str:
    """Generate a filename with a timestamp."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{ext}"


def ensure_directory(path: str) -> None:
    """Ensure that a directory exists."""
    os.makedirs(path, exist_ok=True)
