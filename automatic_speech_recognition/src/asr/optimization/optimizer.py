from typing import Any, Dict, Optional

from .dialect import DialectNormalizer
from .frontend import AudioFrontend

class ASROptimizer:
    def __init__(
        self, 
        frontend: Optional[AudioFrontend] = None,
        dialect: Optional[DialectNormalizer] = None,
        enable_frontend: bool = True,
        enable_dialect: bool = True
    ):
        self.frontend = frontend
        self.dialect = dialect
        self.enable_frontend = enable_frontend and frontend is not None
        self.enable_dialect = enable_dialect and dialect is not None

    def process_audio(self, audio, sr: int):
        if self.enable_frontend:
            return self.frontend(audio, sr)
        return audio

    def process_text(self, text: str) -> str:
        if self.enable_dialect:
            return self.dialect.normalize(text)
        return text

    def as_hooks(self):
        fe = self.process_audio if self.enable_frontend else None
        dn = self.process_text if self.enable_dialect else None
        return fe, dn

def build_optimizer(cfg: Dict[str, Any]) -> ASROptimizer:
    fe_cfg = dict(cfg.get("frontend", {}) or {})
    dl_cfg = dict(cfg.get("dialect", {}) or {})

    fe_enabled = bool(fe_cfg.pop("enabled", True))
    dl_enabled = bool(dl_cfg.pop("enabled", True))

    frontend = AudioFrontend(**fe_cfg) if fe_enabled else None
    dialect = DialectNormalizer(**dl_cfg) if dl_enabled else None

    return ASROptimizer(
        frontend=frontend, 
        dialect=dialect,
        enable_frontend=fe_enabled, 
        enable_dialect=dl_enabled
    )
