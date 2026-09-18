from typing import Any

_EXPORT = ("export_asr_onnx", "export_module_to_onnx")
_QUANTIZE = (
    "quantize_onnx", 
    "build_calibration_set", 
    "split_calibration_holdout",
    "load_manifest_records", 
    "load_baseline_details", 
    "accuracy_regression",
    "verify_quantized_accuracy", 
    "run_pipeline",
)

__all__ = list(_EXPORT) + list(_QUANTIZE)

_LAZY = {name: ".export_onnx" for name in _EXPORT}
_LAZY.update({name: ".quantize" for name in _QUANTIZE})

def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    value = getattr(importlib.import_module(module_name, __name__), name)
    globals()[name] = value
    return value
