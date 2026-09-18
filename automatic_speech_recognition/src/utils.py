import logging
import os
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_LOG_FILE = "logs/run.log"
DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"

DEFAULT_NUM_GPUS = 1

def resolve_num_gpus(num_gpus: Optional[int] = None) -> int:
    count = DEFAULT_NUM_GPUS if num_gpus in (None, "") else int(num_gpus)
    return max(1, count)

def resolve_device(num_gpus: Optional[int] = None, rank: int = 0) -> str:
    count = resolve_num_gpus(num_gpus)
    if count <= 1:
        return "cuda:0"
    return f"cuda:{int(rank) % count}"

def resolve_devices(num_gpus: Optional[int] = None) -> List[str]:
    count = resolve_num_gpus(num_gpus)
    return [resolve_device(count, rank) for rank in range(count)]

def resolve_log_path(log_file: Optional[str] = None) -> str:
    path = log_file or DEFAULT_LOG_FILE
    return path if os.path.isabs(path) else os.path.join(str(ROOT), path)

def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(level)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    if log_file == "":
        return logger
    path = resolve_log_path(log_file)
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
    logger.addHandler(handler)
    return logger

def log_and_print(message: str, level: int = logging.INFO) -> None:
    print(message, flush=True)
    logger = logging.getLogger()
    logger.log(level, message)
    for handler in logger.handlers:
        handler.flush()
