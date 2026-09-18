import logging
import os
from typing import Any, Dict, List, Optional

from src.asr.utils import save_json

from .metrics import summarize

logger = logging.getLogger("tests.report")

def compare_backends(references: List[str], baseline_texts: List[str], optimized_texts: List[str], budget: float = 0.03, include_slots: bool = False) -> Dict[str, Any]:
    base = summarize(references, baseline_texts, include_slots=include_slots)
    opt = summarize(references, optimized_texts, include_slots=include_slots)
    deltas = {k: round(opt[k] - base[k], 4) for k in base if k != "samples"}
    result: Dict[str, Any] = {
        "samples": base["samples"],
        "baseline": base,
        "optimized": opt,
        "delta": deltas,
        "budget": budget,
        "cer_within_budget": deltas["cer"] <= budget,
    }
    if include_slots:
        result["critical_within_budget"] = deltas["critical_score"] >= -budget
    return result

def to_markdown(result: Dict[str, Any], latency: Optional[Dict[str, Any]] = None, title: str = "ASR Optimization Comparison") -> str:
    base, opt, delta = result["baseline"], result["optimized"], result["delta"]
    lines = [f"# {title}", "",
             f"- samples: {result['samples']}",
             f"- budget: {result['budget']}",
             f"- CER within budget: {result['cer_within_budget']}"]
    if "critical_within_budget" in result:
        lines.append(f"- critical within budget: {result['critical_within_budget']}")
    lines += ["", "| metric | baseline | optimized | delta |", "|---|---|---|---|"]
    keys = ["cer", "ser"]
    if "critical_score" in base:
        keys += ["critical_score", "direction_match", "negation_match", "quantity_recall", "action_match"]
    for key in keys:
        lines.append(f"| {key} | {base.get(key)} | {opt.get(key)} | {delta.get(key)} |")
    if latency:
        lines += ["", "## latency", "", "| metric | value |", "|---|---|"]
        for key in ("latency_mean_ms", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms", "memory_rss_peak_mb", "power_mean_w"):
            if key in latency:
                lines.append(f"| {key} | {latency[key]} |")
    return "\n".join(lines) + "\n"

def save_report(result: Dict[str, Any], path: str, latency: Optional[Dict[str, Any]] = None) -> str:
    json_path = path if path.endswith(".json") else path + ".json"
    md_path = (path[:-5] if path.endswith(".json") else path) + ".md"
    save_json(result, json_path)
    folder = os.path.dirname(md_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(to_markdown(result, latency))
    logger.info("report written: %s", md_path)
    return md_path
