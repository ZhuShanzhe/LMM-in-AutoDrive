import json
import logging
import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_CFG = {
    "model_type": "onnx",
    "march": "nash-p",            
    "input_type_rt": "featuremap",
    "norm_type": "no_preprocess",
    "input_type_train": "featuremap",
    "calibration_type": "default",
    "compile_mode": "latency",
    "output_model_file_prefix": "asr_encoder",
}

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

def _resolve(path: Optional[str]) -> Optional[str]:
    if not path:
        return path
    return path if os.path.isabs(path) else os.path.join(_ROOT, path)

def toolchain_available() -> bool:
    return shutil.which("hb_compile") is not None

def _onnx_input_spec(onnx_path: str, fallback_name: str = "input_features", input_name: Optional[str] = None) -> Any:
    try:
        import onnx
        graph = onnx.load(onnx_path, load_external_data=False).graph
        initializers = {init.name for init in graph.initializer}
        inputs = [i for i in graph.input if i.name not in initializers]
        target = next((i for i in inputs if i.name == input_name), inputs[0])
        dims = [d.dim_value if d.dim_value > 0 else 1 for d in target.type.tensor_type.shape.dim]
        return target.name, dims
    except Exception as exc:
        logger.warning("could not read the onnx input signature (%s); falling back to %s [128, 3000]", exc, fallback_name)
        return fallback_name, [128, 3000]

def write_hb_config(
    onnx_path: str, 
    config_path: str, 
    input_name: str = "input_features",
    calib_dir: Optional[str] = None, 
    overrides: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None
) -> str:
    cfg = dict(DEFAULT_CFG)
    cfg.update(overrides or {})
    resolved_name, input_shape = _onnx_input_spec(onnx_path, input_name, cfg.pop("input_name", None))
    shape_str = "x".join(str(int(d)) for d in input_shape)
    base = os.path.dirname(os.path.abspath(config_path)) or "."

    def _rel(path: Optional[str]) -> str:
        if not path:
            return ""
        return os.path.relpath(os.path.abspath(path), base).replace(os.sep, "/")

    lines: List[str] = [
        "model_parameters:",
        f"  onnx_model: {_rel(onnx_path)}",
        f"  march: {cfg['march']}",
        f"  output_model_file_prefix: {cfg['output_model_file_prefix']}",
    ]
    if output_dir:
        lines.append(f"  working_dir: {_rel(output_dir)}")
    lines += [
        "input_parameters:",
        f"  input_name: {resolved_name}",
        f"  input_type_train: {cfg['input_type_train']}",
        f"  input_type_rt: {cfg['input_type_rt']}",
        f"  norm_type: {cfg['norm_type']}",
        f"  input_shape: {shape_str}",
        "calibration_parameters:",
        f"  cal_data_dir: {_rel(calib_dir)}",
        f"  calibration_type: {cfg['calibration_type']}",
        "compiler_parameters:",
        f"  compile_mode: {cfg['compile_mode']}",
    ]
    folder = os.path.dirname(config_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    logger.info("hb_compile config written: %s", config_path)
    return config_path

def convert(
    onnx_path: str, 
    output_dir: str, 
    config_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    calib_dir: Optional[str] = None,
    timeout: Optional[float] = None
) -> Dict[str, Any]:
    if not toolchain_available():
        raise RuntimeError("Horizon toolchain not found (hb_mapper/hb_compile). Run this step on a machine with the official J6P toolchain installed.")
    os.makedirs(output_dir, exist_ok=True)
    cfg = config_path or os.path.join(output_dir, "hb_compile.yaml")
    if not config_path:
        write_hb_config(onnx_path, cfg, calib_dir=calib_dir, overrides=overrides, output_dir=output_dir)

    from ..utils import to_rel_path

    cmd = ["hb_compile", "-c", os.path.relpath(cfg, output_dir)]
    logger.info("running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=output_dir, timeout=timeout)
    report = {
        "config": to_rel_path(cfg), 
        "output_dir": to_rel_path(output_dir),
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-2000:], 
        "stderr_tail": proc.stderr[-2000:]
    }
    if proc.returncode != 0:
        logger.error("hb_compile failed (rc=%d)", proc.returncode)
    else:
        report["models"] = sorted(
            to_rel_path(os.path.join(root, name))
            for root, _, files in os.walk(output_dir)
            for name in files if name.endswith((".hbm", ".bin"))
        )
        if not report["models"]:
            logger.warning("hb_compile succeeded but no .hbm/.bin found under %s", output_dir)
    return report

def verify_hbm_alignment(
    onnx_path: str,
    hbm_path: str,
    calib_dir: str,
    input_name: str = "input_features",
    cos_threshold: float = 0.99,
    max_samples: int = 8,
) -> Dict[str, Any]:
    import numpy as np
    import onnxruntime as ort

    from ..utils import to_rel_path
    from .board_runtime import HbmSession

    files = sorted(f for f in os.listdir(calib_dir) if f.endswith(".npy"))[: max(1, int(max_samples))]
    if not files:
        raise FileNotFoundError("no calibration .npy found under " + calib_dir)

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_out = sess.get_outputs()[0].name
    try:
        hbm = HbmSession(hbm_path, input_name=input_name)
    except RuntimeError as exc:
        reason = ("no BPU runtime available on this host; run the alignment check on the J6P target where a .hbm can actually execute (" + str(exc) + ")")
        logger.warning("hbm alignment skipped: %s", reason)
        return {"hbm_path": to_rel_path(hbm_path), "skipped": True, "reason": reason}
    cosines: List[float] = []
    max_diffs: List[float] = []
    for name in files:
        feats = np.load(os.path.join(calib_dir, name)).astype(np.float32)
        ref = sess.run([ort_out], {input_name: feats})[0].reshape(-1)
        got = hbm.run(feats).reshape(-1)
        n = min(ref.size, got.size)
        ref, got = ref[:n], got[:n]
        denom = float(np.linalg.norm(ref) * np.linalg.norm(got)) or 1e-12
        cosines.append(float(np.dot(ref, got) / denom))
        max_diffs.append(float(np.max(np.abs(ref - got))))

    mean_cos = float(np.mean(cosines))
    report = {
        "onnx_path": to_rel_path(onnx_path),
        "hbm_path": to_rel_path(hbm_path),
        "samples": len(files),
        "mean_cosine": round(mean_cos, 5),
        "max_abs_diff": round(float(np.max(max_diffs)), 5),
        "cosine_threshold": cos_threshold,
        "within_budget": mean_cos >= cos_threshold,
    }
    logger.info("hbm alignment report: %s", report)
    return report

def dump_calibration_npy(
    manifest: str,
    out_dir: str,
    model_id_or_path: str,
    input_shape: List[int],
    num_samples: int = 100,
    seed: int = 42,
) -> str:
    import random

    import numpy as np
    from transformers import WhisperFeatureExtractor

    from ..compression.quantize import load_manifest_records
    from ..utils import load_audio

    records = load_manifest_records(manifest)
    paths = [r.get("audio") or r.get("audio_file") for r in records]
    paths = [p for p in paths if p]
    if not paths:
        raise ValueError("no calibration audio found in manifest: " + manifest)
    random.seed(seed)
    random.shuffle(paths)
    chosen = paths[: max(1, int(num_samples))]

    fe = WhisperFeatureExtractor.from_pretrained(
        model_id_or_path, local_files_only=os.path.isdir(model_id_or_path))
    num_mel = int(input_shape[0])
    num_frames = int(input_shape[-1])
    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    for path in chosen:
        resolved = _resolve(path)
        if not resolved or not os.path.exists(resolved):
            logger.warning("calibration audio missing, skipping: %s", path)
            continue
        audio = load_audio(resolved, fe.sampling_rate)
        feats = fe(audio, sampling_rate=fe.sampling_rate, return_tensors="np").input_features[0]
        feats = feats[:num_mel]
        frames = feats.shape[-1]
        if frames < num_frames:
            pad = np.zeros((feats.shape[0], num_frames - frames), dtype=feats.dtype)
            feats = np.concatenate([feats, pad], axis=-1)
        elif frames > num_frames:
            feats = feats[:, :num_frames]
        np.save(os.path.join(out_dir, f"calib_{saved:04d}.npy"),
                feats.reshape(input_shape).astype(np.float32))
        saved += 1
    if saved == 0:
        raise RuntimeError("no usable calibration audio; check the manifest paths")
    logger.info("calibration data written: %d samples -> %s (shape=%s)", saved, out_dir, input_shape)
    return out_dir

_OVERRIDE_KEYS = (
    "march", "compile_mode", "input_type_train", "input_type_rt",
    "norm_type", "calibration_type", "output_model_file_prefix", "input_name",
)

def _load_deployment_config(config_path: str) -> Dict[str, Any]:
    from ..utils import load_yaml

    return load_yaml(config_path) or {}

def main() -> None:
    import argparse

    from src.utils import log_and_print, setup_logging

    parser = argparse.ArgumentParser(description="Compile the float32 ONNX ASR encoder into a J6P .hbm via the Horizon toolchain")
    parser.add_argument("--config", default="configs/asr/deployment.yaml")
    parser.add_argument("--onnx", default=None, help="override the float32 onnx path")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--calib-dir", default=None, help="reuse an existing calibration .npy directory")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--verify", action="store_true", help="check the float-vs-hbm encoder alignment after compiling")
    parser.add_argument("--log-file", default="logs/j6p_convert.log", help="run log path; empty string disables it")
    args = parser.parse_args()
    setup_logging(args.log_file)

    dep = _load_deployment_config(_resolve(args.config))
    j6p: Dict[str, Any] = dict(dep.get("j6p", {}) or {})
    onnx_path = _resolve(args.onnx or dep.get("onnx_path"))
    if not onnx_path or not os.path.exists(onnx_path):
        raise FileNotFoundError("float32 onnx not found: " + str(onnx_path))
    output_dir = _resolve(args.output_dir or j6p.get("output_dir", "outputs/deployment/j6p"))
    model_id = dep.get("model_id_or_path", "models/Qwen3-ASR-1.7B")

    overrides = {k: j6p[k] for k in _OVERRIDE_KEYS if k in j6p}
    input_name = overrides.get("input_name", "input_features")
    _, input_shape = _onnx_input_spec(onnx_path, input_name, input_name)

    calib_dir = _resolve(args.calib_dir) if args.calib_dir else None
    manifest = j6p.get("calibration_manifest")
    if calib_dir is None and manifest:
        calib_dir = os.path.join(output_dir, "calibration")
        num_samples = args.num_samples if args.num_samples is not None else int(j6p.get("calibration_num_samples", 100))
        dump_calibration_npy(_resolve(manifest), calib_dir, model_id, input_shape,
                             num_samples=num_samples, seed=int(j6p.get("calibration_seed", 42)))
    if calib_dir is None:
        logger.warning("no calibration data configured; calibration_type=%s may fail", j6p.get("calibration_type", "default"))

    from ..utils import to_rel_path

    log_and_print(f"[j6p] onnx={to_rel_path(onnx_path)} output_dir={to_rel_path(output_dir)} calib_dir={to_rel_path(calib_dir) if calib_dir else '(none)'}")
    report = convert(onnx_path, output_dir, overrides=overrides or None, calib_dir=calib_dir, timeout=j6p.get("timeout_seconds"))
    log_and_print(f"[j6p] returncode={report['returncode']} models={report.get('models', [])}")
    if report["returncode"] != 0:
        raise SystemExit(report["returncode"])

    if args.verify:
        if not calib_dir:
            raise SystemExit("--verify needs calibration data; set j6p.calibration_manifest or pass --calib-dir")
        hbm_files = [m for m in report.get("models", []) if m.endswith(".hbm")]
        if not hbm_files:
            raise SystemExit("--verify found no .hbm to check")
        vcfg = dict(j6p.get("verify", {}) or {})
        align = verify_hbm_alignment(
            onnx_path, 
            _resolve(hbm_files[0]), 
            calib_dir,
            input_name = input_name,
            cos_threshold = float(vcfg.get("cosine_threshold", 0.99)),
            max_samples = int(vcfg.get("max_samples", 8)),
        )
        log_and_print("[j6p] alignment=" + json.dumps(align, ensure_ascii=False))
        if align.get("skipped"):
            logger.warning("alignment check skipped: %s", align.get("reason"))
        elif not align["within_budget"]:
            raise SystemExit("hbm encoder degraded beyond the cosine budget; inspect the calibration data")

if __name__ == "__main__":
    main()
