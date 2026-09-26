import logging
import os
import tempfile
import time
from typing import Any, Dict, List, Optional

from ...utils import DEFAULT_NUM_GPUS, resolve_device, resolve_num_gpus

logger = logging.getLogger(__name__)

BACKENDS = ("pytorch", "onnx", "j6p")

_GPU_PROVIDERS = ("CUDAExecutionProvider",)

def _resolve_providers(ort, requested=None) -> List[str]:
    available = ort.get_available_providers()
    if requested:
        chosen = [p for p in requested if p in available]
        missing = [p for p in requested if p not in available]
        if missing:
            logger.warning("requested execution providers unavailable: %s (available: %s)", missing, available)
        if chosen:
            return chosen
    gpu = [p for p in _GPU_PROVIDERS if p in available]
    if gpu:
        return gpu + ["CPUExecutionProvider"]
    logger.warning(
        "no GPU execution provider found (%s); the ONNX encoder will run on CPU and use many cores. "
        "Install onnxruntime-gpu to accelerate it.",
        available,
    )
    return ["CPUExecutionProvider"]

def _make_onnx_audio_tower(session, input_names, mel_frames, target_dtype=None, run_fn=None):
    import torch

    input_name = list(input_names)[0]

    class _EncoderOutput(tuple):
        @property
        def last_hidden_state(self):
            return self[0]

    class _OnnxAudioTower(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.session = session
            self.input_name = input_name
            self.mel_frames = mel_frames
            self.target_dtype = target_dtype
            self.run_fn = run_fn

        def forward(self, input_features, feature_lens=None, aftercnn_lens=None, *args, **kwargs):
            if input_features.dim() == 3:
                if input_features.shape[0] > 1:
                    logger.warning(
                        "onnx encoder received batch size %d; only the first item is encoded",
                        input_features.shape[0],
                    )
                feats = input_features[0]
            else:
                feats = input_features
            feats = feats.detach().to("cpu").to(torch.float32)
            frames = feats.shape[-1]
            if frames < self.mel_frames:
                pad = torch.zeros(feats.shape[0], self.mel_frames - frames, dtype=feats.dtype)
                feats = torch.cat([feats, pad], dim=-1)
            elif frames > self.mel_frames:
                logger.warning(
                    "mel frames %d exceed the exported length %d; the tail is discarded. "
                    "Re-export with a larger mel_frames to keep the full audio.",
                    frames, self.mel_frames,
                )
                feats = feats[..., : self.mel_frames]
            array = feats.contiguous().numpy()
            if self.run_fn is not None:
                outputs = [self.run_fn(array)]
            else:
                outputs = self.session.run(None, {self.input_name: array})
            dtype = self.target_dtype or input_features.dtype
            tensor = torch.from_numpy(outputs[0]).to(device=input_features.device, dtype=dtype)
            return _EncoderOutput((tensor,))

    return _OnnxAudioTower()

class ASRRuntime:
    def __init__(
        self, 
        backend: str = "pytorch",
        model_id_or_path: str = "models/Qwen3-ASR-1.7B",
        onnx_path: Optional[str] = None, 
        device: Optional[str] = None,
        num_gpus: int = DEFAULT_NUM_GPUS,
        rank: int = 0,
        dtype: str = "bfloat16", 
        language: Optional[str] = None,
        frontend=None, 
        dialect_normalizer=None,
        target_sample_rate: int = 16000, 
        raise_on_error: bool = False,
        hbm_path: Optional[str] = None,
        hbm_mel_frames: int = 3000,
        service=None,
        **backend_kwargs
    ) -> None:
        if backend not in BACKENDS:
            raise ValueError(f"unsupported backend: {backend}; expected {BACKENDS}")
        self.backend = backend
        self.raise_on_error = raise_on_error
        self.language = language
        self.target_sample_rate = target_sample_rate
        self.frontend = frontend
        self.dialect_normalizer = dialect_normalizer
        self.backend_kwargs = backend_kwargs
        self._service = service
        self._sess = None
        self._onnx_attached = False
        self._hbm_attached = False
        self._onnx_path = onnx_path
        self._hbm_path = hbm_path
        self._hbm_mel_frames = int(hbm_mel_frames)
        self._model_id = model_id_or_path
        self.num_gpus = resolve_num_gpus(num_gpus)
        self.rank = rank
        self.device = device or resolve_device(self.num_gpus, rank)
        self.dtype = dtype

    def _get_service(self):
        if self._service is None:
            from .. import Qwen3ASRService
            self._service = Qwen3ASRService(
                model_id_or_path=self._model_id, device=self.device,
                dtype=self.dtype, language=self.language,
                frontend=self.frontend, dialect_normalizer=self.dialect_normalizer,
                target_sample_rate=self.target_sample_rate
            )
        return self._service

    def _get_session(self):
        if self._sess is None:
            if not self._onnx_path or not os.path.exists(self._onnx_path):
                raise FileNotFoundError("onnx model not found: " + str(self._onnx_path))
            import onnxruntime as ort

            def _build_opts():
                options = ort.SessionOptions()
                options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                return options

            providers = _resolve_providers(ort, self.backend_kwargs.get("providers"))
            try:
                self._sess = ort.InferenceSession(self._onnx_path, sess_options=_build_opts(), providers=providers)
            except Exception as exc:
                logger.warning(
                    "Failed to create the onnx session with %s (%s); retrying with CPUExecutionProvider. "
                    "IntegerOps quantization such as ConvInteger/MatMulInteger is not implemented by GPU providers.",
                    providers, exc,
                )
                self._sess = ort.InferenceSession(self._onnx_path, sess_options=_build_opts(), providers=["CPUExecutionProvider"])
            logger.info("onnx session active providers: %s", self._sess.get_providers())
        return self._sess

    def _attach_onnx_encoder(self) -> None:
        from ..utils import module_tree_hint, resolve_submodule

        model = self._get_service().model
        found = resolve_submodule(model)
        if found is None:
            raise RuntimeError("could not locate the audio encoder; " + module_tree_hint(model))
        owner, attr_name, dotted = found
        encoder = getattr(owner, attr_name)
        try:
            target_dtype = next(encoder.parameters()).dtype
        except (StopIteration, AttributeError):
            target_dtype = None
        session = self._get_session()
        input_names = [i.name for i in session.get_inputs()]
        shape = session.get_inputs()[0].shape
        mel_frames = int(shape[-1]) if isinstance(shape[-1], int) else 3000
        setattr(owner, attr_name, _make_onnx_audio_tower(session, input_names, mel_frames, target_dtype))
        logger.info("onnx encoder attached at %s (inputs=%s, mel_frames=%d)", dotted, input_names, mel_frames)

    def _transcribe_onnx(self, audio_path: str) -> Dict[str, Any]:
        if not self._onnx_attached:
            self._attach_onnx_encoder()
            self._onnx_attached = True
        rec = self._get_service().transcribe_file(audio_path)
        return {
            "text": rec.get("text", ""), 
            "language": rec.get("language"),
            "onnx_encoder": self._onnx_path,
        }

    def _attach_hbm_encoder(self) -> None:
        from ..utils import module_tree_hint, resolve_submodule
        from .board_runtime import HbmSession

        session = HbmSession(self._hbm_path)
        mel_frames = session.mel_frames or self._hbm_mel_frames
        model = self._get_service().model
        found = resolve_submodule(model)
        if found is None:
            raise RuntimeError("could not locate the audio encoder; " + module_tree_hint(model))
        owner, attr_name, dotted = found
        encoder = getattr(owner, attr_name)
        try:
            target_dtype = next(encoder.parameters()).dtype
        except (StopIteration, AttributeError):
            target_dtype = None
        tower = _make_onnx_audio_tower(None, [session.input_name], mel_frames, target_dtype, run_fn=session.run)
        setattr(owner, attr_name, tower)
        logger.info("hbm encoder attached at %s (mel_frames=%d, hbm=%s)", dotted, mel_frames, self._hbm_path)

    def _transcribe_j6p(self, audio_path: str) -> Dict[str, Any]:
        if not self._hbm_path:
            raise RuntimeError("the j6p backend needs hbm_path pointing to a compiled .hbm; build it with scripts/run_j6p_deploy.sh")
        if not self._hbm_attached:
            self._attach_hbm_encoder()
            self._hbm_attached = True
        rec = self._get_service().transcribe_file(audio_path)
        return {
            "text": rec.get("text", ""),
            "language": rec.get("language"),
            "hbm_model": self._hbm_path,
        }

    def transcribe_file(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        if self.backend == "onnx" and not self._onnx_attached:
            self._attach_onnx_encoder()
            self._onnx_attached = True
        elif self.backend == "j6p" and not self._hbm_attached:
            if not self._hbm_path:
                raise RuntimeError("the j6p backend needs hbm_path pointing to a compiled .hbm; build it with scripts/run_j6p_deploy.sh")
            self._attach_hbm_encoder()
            self._hbm_attached = True
        return self._get_service().transcribe_file(audio_path, **kwargs)

    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        start = time.perf_counter()
        error = None
        try:
            if self.backend == "pytorch":
                rec = self._get_service().transcribe_file(audio_path)
            elif self.backend == "onnx":
                rec = self._transcribe_onnx(audio_path)
            else:
                rec = self._transcribe_j6p(audio_path)
            text = rec.get("text", "")
            lang = rec.get("language")
        except Exception as exc:
            if self.raise_on_error:
                raise
            logger.error("transcription failed on backend %s for %s: %s", self.backend, audio_path, exc)
            rec, text, lang, error = {}, "", None, str(exc)
        elapsed = round(time.perf_counter() - start, 4)
        record = {
            "audio_file": audio_path, 
            "text": text, 
            "language": lang,
            "backend": self.backend, 
            "latency_seconds": elapsed,
            "success": bool(text), 
        }
        if error:
            record["error"] = error
        record.update({k: v for k, v in rec.items() if k.startswith(("onnx_", "hbm_"))})
        return record

    def transcribe_batch(self, audio_paths: List[str]) -> List[Dict[str, Any]]:
        return [self.transcribe(p) for p in audio_paths]
