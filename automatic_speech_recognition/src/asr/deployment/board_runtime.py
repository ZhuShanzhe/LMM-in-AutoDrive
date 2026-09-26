import logging
import os
import shutil
import subprocess
import tempfile
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

class HbmSession:
    def __init__(self, hbm_path: str, input_name: str = "input_features") -> None:
        if not hbm_path or not os.path.exists(hbm_path):
            raise FileNotFoundError("hbm model not found: " + str(hbm_path))
        self.hbm_path = os.path.abspath(hbm_path)
        self.input_name = input_name
        self._model = None
        self._model_name = None
        self._exec = shutil.which("hrt_model_exec")
        self._binding = self._load_binding()
        if self._binding is None and self._exec is None:
            raise RuntimeError(
                "no BPU runtime found: install the board (aarch64) runtime so that either the "
                "'hbm_runtime' python module or 'hrt_model_exec' is on PATH; a .hbm only runs on the J6P target."
            )

    def _load_binding(self) -> Optional[str]:
        try:
            from hbm_runtime import HB_HBMRuntime
        except ImportError:
            return None
        self._model = HB_HBMRuntime(self.hbm_path)
        self._model_name = self._model.model_names[0]
        logger.info("hbm session ready via hbm_runtime binding: %s", self.hbm_path)
        return "hbm_runtime"

    @property
    def mel_frames(self) -> Optional[int]:
        if self._model is None:
            return None
        shape = self._model.input_shapes[self._model_name][self.input_name]
        return int(shape[-1])

    def run(self, feats: np.ndarray) -> np.ndarray:
        feats = np.ascontiguousarray(feats, dtype=np.float32)
        if self._model is not None:
            out = self._model.run(feats)
            arr = out[self._model_name] if isinstance(out, dict) else out
            return np.asarray(arr, dtype=np.float32)
        return self._run_cli(feats)

    def _run_cli(self, feats: np.ndarray) -> np.ndarray:
        with tempfile.TemporaryDirectory() as tmp:
            in_bin = os.path.join(tmp, "input.bin")
            feats.tofile(in_bin)
            cmd = [self._exec, "infer", "--model_file=" + self.hbm_path, "--input_file=" + in_bin, "--enable_dump=true"]
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp, timeout=600)
            if proc.returncode != 0:
                raise RuntimeError("hrt_model_exec failed: " + (proc.stderr or proc.stdout)[-1000:])
            dumps = sorted(f for f in os.listdir(tmp) if f.endswith(".bin") and f != os.path.basename(in_bin))
            if not dumps:
                raise RuntimeError("hrt_model_exec produced no output dump (cwd=" + tmp + ")")
            return np.fromfile(os.path.join(tmp, dumps[0]), dtype=np.float32)