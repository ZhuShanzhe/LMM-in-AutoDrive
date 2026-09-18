import logging
import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_CFG = {
    "model_type": "onnx",
    "march": "bayes",            
    "input_type_rt": "nv12",
    "input_layout_rt": "NHWC",
    "calibration_type": "default",
    "compile_mode": "latency",
}

def toolchain_available() -> bool:
    return shutil.which("hb_mapper") is not None or shutil.which("hb_compile") is not None

def _onnx_input_spec(onnx_path: str, fallback_name: str) -> Any:
    """Return the first graph input's (name, shape) and fall back when onnx is missing."""
    try:
        import onnx
        graph = onnx.load(onnx_path, load_external_data=False).graph
        inp = graph.input[0]
        dims = [d.dim_value if d.dim_value > 0 else 1 for d in inp.type.tensor_type.shape.dim]
        return inp.name, dims
    except Exception as exc:
        logger.warning("could not read the onnx input signature (%s); falling back to %s [128, 3000]", exc, fallback_name)
        return fallback_name, [128, 3000]

def write_hb_config(
    onnx_path: str, 
    config_path: str, 
    input_name: str = "input_features",
    calib_dir: Optional[str] = None, 
    overrides: Optional[Dict[str, Any]] = None
) -> str:
    cfg = dict(DEFAULT_CFG)
    cfg.update(overrides or {})
    input_name, input_shape = _onnx_input_spec(onnx_path, input_name)
    lines: List[str] = [
        f"model_type: {cfg['model_type']}",
        f"march: {cfg['march']}",
        f"onnx_model: {onnx_path}",
        f"input_type_rt: {cfg['input_type_rt']}",
        f"input_layout_rt: {cfg['input_layout_rt']}",
        f"calibration_type: {cfg['calibration_type']}",
        f"compile_mode: {cfg['compile_mode']}",
        "input_parameters:",
        f"  {input_name}:",
        "    type: float32",
        f"    shape: {input_shape}",
    ]
    if calib_dir:
        lines += ["calibration_parameters:", f"  cal_data_dir: {calib_dir}"]
    folder = os.path.dirname(config_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    logger.info("hb_mapper config written: %s", config_path)
    return config_path

def convert(
    onnx_path: str, 
    output_dir: str, 
    config_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Convert ONNX to a J6P BPU model. Raises if the toolchain is unavailable."""
    if not toolchain_available():
        raise RuntimeError("Horizon toolchain not found (hb_mapper/hb_compile). Run this step on a machine with the official J6P toolchain installed.")
    os.makedirs(output_dir, exist_ok=True)
    cfg = config_path or os.path.join(output_dir, "hb_mapper.yaml")
    if not config_path:
        write_hb_config(onnx_path, cfg, overrides=overrides)

    cmd = ["hb_mapper", "makertbin", "--config", cfg, "--model-type", "onnx", "--output-dir", output_dir]
    logger.info("running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    report = {
        "config": cfg, 
        "output_dir": output_dir,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-2000:], 
        "stderr_tail": proc.stderr[-2000:]
    }
    if proc.returncode != 0:
        logger.error("hb_mapper failed (rc=%d)", proc.returncode)
    else:
        report["bin_files"] = [n for n in os.listdir(output_dir) if n.endswith(".bin")]
    return report
