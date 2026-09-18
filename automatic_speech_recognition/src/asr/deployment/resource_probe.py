import logging
import os
import statistics
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[k]

def memory_rss_mb() -> float:
    try:
        import psutil
        return round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 2)
    except Exception:
        return 0.0

def gpu_power_w() -> Optional[float]:
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return round(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0, 2)
    except Exception:
        logger.warning("NVML power probe unavailable; power not measured.")
        return None

def j6p_power_w() -> Optional[float]:
    import shutil
    import subprocess

    tool = shutil.which("hrut_somstatus")
    if not tool:
        return None
    try:
        out = subprocess.run([tool], capture_output=True, text=True, timeout=5).stdout
        for line in out.splitlines():
            if "power" in line.lower():
                return float("".join(c for c in line if c.isdigit() or c == ".") or 0)
    except Exception:
        logger.warning("J6P power probe failed.")
    return None

class ResourceProbe:
    def __init__(
        self, 
        warmup: int = 3, 
        repeats: int = 20,
        power_probe: Optional[Callable[[], Optional[float]]] = None
    ) -> None:
        self.warmup = warmup
        self.repeats = repeats
        self.power_probe = power_probe or j6p_power_w

    def run(self, fn: Callable[[], Any]) -> Dict[str, Any]:
        for _ in range(self.warmup):
            fn()
        latencies: List[float] = []
        mem_peak = 0.0
        power_samples: List[float] = []
        for _ in range(self.repeats):
            start = time.perf_counter()
            fn()
            latencies.append(time.perf_counter() - start)
            mem_peak = max(mem_peak, memory_rss_mb())
            p = self.power_probe() if self.power_probe else None
            if p is not None:
                power_samples.append(p)
        return {
            "repeats": self.repeats,
            "latency_mean_ms": round(statistics.mean(latencies) * 1000, 2),
            "latency_p50_ms": round(_percentile(latencies, 50) * 1000, 2),
            "latency_p95_ms": round(_percentile(latencies, 95) * 1000, 2),
            "latency_p99_ms": round(_percentile(latencies, 99) * 1000, 2),
            "latency_max_ms": round(max(latencies) * 1000, 2),
            "memory_rss_peak_mb": mem_peak,
            "power_mean_w": round(statistics.mean(power_samples), 2) if power_samples else None,
            "power_max_w": round(max(power_samples), 2) if power_samples else None,
        }
