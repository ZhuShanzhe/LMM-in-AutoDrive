import json
import logging
import os
from typing import Any, Dict, List, Optional

from ..text_metrics import corpus_cer
from ..utils import to_rel_path
from src.utils import DEFAULT_NUM_GPUS, log_and_print, resolve_device, setup_logging

logger = logging.getLogger(__name__)

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

def _resolve(path: str) -> str:
    if not path:
        return path
    return path if os.path.isabs(path) else os.path.join(_ROOT, path)

_CONV_OPS = ("Conv", "ConvTranspose")

def _detect_op_types(onnx_path: str) -> set:
    import onnx

    found = set()

    def _walk(graph) -> None:
        for node in graph.node:
            found.add(node.op_type)
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    _walk(attr.g)
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for sub in attr.graphs:
                        _walk(sub)

    _walk(onnx.load(onnx_path).graph)
    return found

def resolve_weight_type(input_onnx: str, weight_type: Optional[str] = None):
    from onnxruntime.quantization import QuantType

    name = str(weight_type or "auto").lower()
    if name == "auto":
        conv_ops = sorted(_detect_op_types(input_onnx) & set(_CONV_OPS))
        chosen = QuantType.QUInt8 if conv_ops else QuantType.QInt8
        logger.info(
            "auto weight type: %s (Conv-family ops: %s); onnxruntime only supports uint8 weights for ConvInteger",
            chosen.name, conv_ops or "none",
        )
        return chosen
    mapping = {"qint8": QuantType.QInt8, "quint8": QuantType.QUInt8}
    if name not in mapping:
        raise ValueError("unsupported weight_type: " + str(weight_type) + "; expected auto, qint8 or quint8")
    return mapping[name]

def load_manifest_records(path: str) -> List[Dict[str, Any]]:
    with open(_resolve(path), "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        return data.get("records", [])
    return data

def build_calibration_set(eval_manifest: str, pool_manifests: List[str], num_samples: int = 100, seed: int = 42) -> List[str]:
    import random

    eval_paths = {r.get("audio") or r.get("audio_file") for r in load_manifest_records(eval_manifest)}
    pool: List[str] = []
    for manifest in pool_manifests:
        try:
            records = load_manifest_records(manifest)
        except FileNotFoundError:
            logger.warning("calibration pool missing: %s", manifest)
            continue
        for rec in records:
            path = rec.get("audio") or rec.get("audio_file")
            if path and path not in eval_paths:
                pool.append(path)
    random.seed(seed)
    random.shuffle(pool)
    chosen = pool[:num_samples]
    logger.info("calibration set: %d samples (pool=%d, eval=%d)", len(chosen), len(pool), len(eval_paths))
    return chosen

def split_calibration_holdout(manifest: str, num_samples: int = 100, seed: int = 42):
    import random

    records = list(load_manifest_records(manifest))
    random.seed(seed)
    random.shuffle(records)
    cut = min(max(int(num_samples), 0), len(records))
    calib_records, eval_records = records[:cut], records[cut:]
    calib_paths = [r.get("audio") or r.get("audio_file") for r in calib_records]
    logger.info("holdout split: %d calibration / %d evaluation (seed=%d)", len(calib_paths), len(eval_records), seed)
    return calib_paths, eval_records

def quantize_onnx(
    input_onnx: str, 
    output_onnx: str,
    calibration_files: Optional[List[str]] = None,
    mode: str = "dynamic", 
    per_channel: bool = True,
    weight_type: Optional[str] = None,
    feature_extractor_path: Optional[str] = None,
    input_name: str = "input_features",
    mel_frames: int = 3000,
) -> Dict[str, Any]:
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType, quantize_dynamic, quantize_static)

    folder = os.path.dirname(output_onnx)
    if folder:
        os.makedirs(folder, exist_ok=True)

    weight_name = "qint8"
    if mode == "dynamic":
        chosen = resolve_weight_type(input_onnx, weight_type)
        weight_name = chosen.name
        quantize_dynamic(input_onnx, output_onnx, weight_type=chosen, per_channel=per_channel)
    elif mode == "static":
        if not calibration_files:
            raise ValueError("static quantization requires calibration_files")
        import numpy as np
        import soundfile as sf
        from transformers import WhisperFeatureExtractor

        if not feature_extractor_path:
            raise ValueError("static quantization requires feature_extractor_path to build mel features")
        local_only = os.path.isdir(feature_extractor_path)
        fe = WhisperFeatureExtractor.from_pretrained(feature_extractor_path, local_files_only=local_only)

        class _Reader(CalibrationDataReader):
            def __init__(self, files):
                self.files = list(files)
                self.idx = 0

            def get_next(self):
                if self.idx >= len(self.files):
                    return None
                audio, _ = sf.read(_resolve(self.files[self.idx]), dtype="float32", always_2d=False)
                self.idx += 1
                feats = fe(
                    np.asarray(audio, dtype=np.float32),
                    sampling_rate=fe.sampling_rate,
                    return_tensors="np",
                ).input_features[0]
                frames = feats.shape[-1]
                if frames < mel_frames:
                    pad = np.zeros((feats.shape[0], mel_frames - frames), dtype=feats.dtype)
                    feats = np.concatenate([feats, pad], axis=-1)
                elif frames > mel_frames:
                    feats = feats[..., :mel_frames]
                return {input_name: np.asarray(feats, dtype=np.float32)}

        quantize_static(
            input_onnx, 
            output_onnx, 
            _Reader(calibration_files),
            quant_format=QuantFormat.QDQ, 
            per_channel=per_channel,
            activation_type=QuantType.QUInt8, 
            weight_type=QuantType.QInt8
        )
    else:
        raise ValueError("unsupported quantization mode: " + str(mode))

    import onnxruntime as ort
    return {
        "quantized_onnx": output_onnx, 
        "mode": mode,
        "weight_type": weight_name,
        "providers": ort.get_available_providers(),
        "size_mb": round(os.path.getsize(output_onnx) / (1024 * 1024), 3)
    }

def accuracy_regression(
    baseline_texts: List[str],    
    quantized_texts: List[str],
    references: List[str], 
    budget: float = 0.03
) -> Dict[str, Any]:
    base_cer = corpus_cer(references, baseline_texts)
    quant_cer = corpus_cer(references, quantized_texts)
    delta = quant_cer - base_cer
    return {
        "baseline_cer": round(base_cer, 4),
        "quantized_cer": round(quant_cer, 4),
        "cer_delta": round(delta, 4),
        "budget": budget,
        "within_budget": float(delta) <= budget,
    }

def load_baseline_details(path: str) -> Dict[str, Dict[str, Any]]:
    with open(_resolve(path), "r", encoding="utf-8") as fh:
        rows = json.load(fh)
    return {os.path.basename(r["audio_file"]): r for r in rows if r.get("audio_file")}

def verify_quantized_accuracy(
    onnx_path: str,
    eval_manifest: str,
    baseline_details: str,
    model_id_or_path: str = "models/Qwen3-ASR-1.7B",
    limit: Optional[int] = None,
    budget: float = 0.03,
    output_json: Optional[str] = None,
    device: Optional[str] = None,
    num_gpus: int = DEFAULT_NUM_GPUS,
    dtype: str = "bfloat16",
    providers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    from ..deployment.runtime import ASRRuntime

    records = list(load_manifest_records(eval_manifest))
    if limit:
        records = records[: int(limit)]
    baseline = load_baseline_details(baseline_details)
    device = device or resolve_device(num_gpus)
    runtime = ASRRuntime(
        backend = "onnx", 
        model_id_or_path = model_id_or_path,
        onnx_path = _resolve(onnx_path), 
        device = device, 
        dtype = dtype,
        providers = providers,
    )

    references: List[str] = []
    baseline_texts: List[str] = []
    quantized_texts: List[str] = []
    skipped = 0
    total = len(records)
    for i, rec in enumerate(records, 1):
        audio = rec.get("audio") or rec.get("audio_file")
        row = baseline.get(os.path.basename(audio))
        if row is None:
            skipped += 1
            continue
        references.append(row["reference"])
        baseline_texts.append(row["hypothesis"])
        quantized_texts.append(runtime.transcribe(_resolve(audio))["text"])
        if i % 10 == 0 or i == total:
            log_and_print(f"[verify] progress {i}/{total}")

    report = accuracy_regression(baseline_texts, quantized_texts, references, budget=budget)
    report["evaluated"] = len(references)
    report["skipped_no_baseline"] = skipped
    report["onnx_path"] = to_rel_path(onnx_path)
    if output_json:
        folder = os.path.dirname(output_json)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        logger.info("quantized accuracy report saved -> %s", output_json)
    return report

def run_pipeline(config_path: str) -> Dict[str, Any]:
    import yaml
    from .export_onnx import export_asr_onnx

    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    model_id = cfg.get("model_id_or_path", "models/Qwen3-ASR-1.7B")
    export_dir = cfg.get("export_dir", "outputs/compression/onnx")
    quant_dir = cfg.get("quantized_dir", "outputs/compression/quantized")
    opset = int(cfg.get("opset", 17))
    dtype = cfg.get("dtype", "float32")

    export_report = export_asr_onnx(model_id, export_dir, opset=opset, dtype=dtype, mel_frames=int(cfg.get("mel_frames", 3000)))
    encoder_onnx = export_report["encoder"]["onnx_path"]

    qcfg = cfg.get("quantization", {}) or {}
    mode = qcfg.get("mode", "dynamic")
    per_channel = bool(qcfg.get("per_channel", True))
    weight_type = qcfg.get("weight_type", "auto")
    input_name = cfg.get("input_name", "input_features")

    ccfg = cfg.get("calibration", {}) or {}
    eval_cfg = cfg.get("eval", {}) or {}
    eval_manifest = eval_cfg.get("manifest", "")

    calib_files: Optional[List[str]] = None
    if mode == "static":
        num_samples = int(ccfg.get("num_samples", 100))
        seed = int(ccfg.get("seed", 42))
        if ccfg.get("holdout"):
            calib_files, eval_records = split_calibration_holdout(eval_manifest, num_samples, seed)
            eval_manifest = os.path.join(quant_dir, "eval_holdout_manifest.json")
            os.makedirs(quant_dir, exist_ok=True)
            with open(eval_manifest, "w", encoding="utf-8") as fh:
                json.dump(eval_records, fh, ensure_ascii=False, indent=2)
        else:
            calib_files = build_calibration_set(
                eval_manifest, 
                list(ccfg.get("pool_manifests", []) or []),
                num_samples = num_samples, 
                seed = seed,
            )
            if not calib_files:
                raise ValueError(
                    "calibration pool is empty; set calibration.holdout: true or point "
                    "calibration.pool_manifests at a manifest that does not overlap eval.manifest"
                )

    out_onnx = os.path.join(quant_dir, os.path.basename(encoder_onnx))
    quant_report = quantize_onnx(
        encoder_onnx, 
        out_onnx,
        calibration_files = calib_files,
        mode = mode, 
        per_channel = per_channel,
        weight_type = weight_type,
        feature_extractor_path = model_id,
        input_name = input_name,
        mel_frames = int(export_report["encoder"].get("mel_frames", int(cfg.get("mel_frames", 3000)))),
    )
    report = {"config": config_path, "export": export_report, "quantization": quant_report, "eval_manifest": eval_manifest}
    report_path = os.path.join(quant_dir, "compression_report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    logger.info("compression report saved -> %s", report_path)
    return report

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Export ONNX and quantize ASR model")
    parser.add_argument("--config", default="configs/asr/compression.yaml")
    parser.add_argument("--verify", action="store_true", help="run the quantized accuracy regression after quantizing")
    parser.add_argument("--log-file", default="logs/compression.log", help="run log path; empty string disables it")
    args = parser.parse_args()
    setup_logging(args.log_file)
    log_and_print(f"[compression] config={args.config} verify={args.verify}")
    report = run_pipeline(args.config)

    if args.verify:
        import yaml
        with open(args.config, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        eval_cfg = cfg.get("eval", {}) or {}
        verify = verify_quantized_accuracy(
            report["quantization"]["quantized_onnx"],
            report["eval_manifest"],
            eval_cfg.get("baseline_details", "outputs/tests/standard/details.json"),
            model_id_or_path = cfg.get("model_id_or_path", "models/Qwen3-ASR-1.7B"),
            limit = eval_cfg.get("limit"),
            budget = float(eval_cfg.get("budget", 0.03)),
            output_json = eval_cfg.get("output_json", "outputs/compression/accuracy_regression.json"),
            device = eval_cfg.get("device"),
            num_gpus = int(eval_cfg.get("num_gpus", 1)),
            dtype = eval_cfg.get("dtype", "bfloat16"),
            providers = eval_cfg.get("providers") or None,
        )
        log_and_print("[compression] accuracy regression: " + json.dumps(verify, ensure_ascii=False))
    log_and_print("[compression] done: " + json.dumps(report.get("quantization", {}), ensure_ascii=False))


if __name__ == "__main__":
    main()
