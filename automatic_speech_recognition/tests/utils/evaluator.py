import json
import logging
import os
import statistics
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.asr.utils import save_json, to_rel_path
from src.asr.text_metrics import is_empty, is_same
from src.utils import log_and_print

from .metrics import summarize

logger = logging.getLogger("tests.evaluator")

class ASREvaluator:
    def __init__(
        self,
        transcribe: Callable[[str], Dict[str, Any]],
        records: List[Dict[str, Any]],
        enable_slots: bool = False,
        transcribes: Optional[List[Callable[[str], Dict[str, Any]]]] = None,
    ):
        self.transcribes = list(transcribes) if transcribes else [transcribe]
        self.transcribe = self.transcribes[0]
        self.records = records
        self.enable_slots = enable_slots

    def _run_one(self, rec: Dict[str, Any], transcribe: Optional[Callable[[str], Dict[str, Any]]] = None) -> Tuple[str, str, Dict[str, Any]]:
        result = (transcribe or self.transcribe)(rec["audio_file"])
        return rec["text"], result.get("text", ""), result

    def _run_parallel(self, workers: int):
        lanes = self.transcribes

        def _task(i: int) -> Tuple[str, str, Dict[str, Any]]:
            return self._run_one(self.records[i], lanes[i % len(lanes)])

        log_and_print(f"[evaluator] {workers} workers across {len(lanes)} device lane(s)")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            yield from pool.map(_task, range(len(self.records)))

    def _build_summary(self, refs: List[str], hyps: List[str], latencies: List[float], failures: int) -> Dict[str, Any]:
        summary: Dict[str, Any] = summarize(refs, hyps, include_slots=self.enable_slots)
        summary["empty_hypotheses"] = failures
        if latencies:
            summary["latency_mean_ms"] = round(statistics.mean(latencies) * 1000, 2)
            summary["latency_p95_ms"] = round(sorted(latencies)[max(0, int(0.95 * (len(latencies) - 1)))] * 1000, 2)
        return summary

    def run(
        self, 
        save_dir: Optional[str] = None,
        latency_key: str = "processing_time_seconds",
        progress: bool = True, 
        progress_every: int = 10,
        save_every: int = 50,
        stream_file: Optional[str] = None,
        label: str = "asr_test",
        workers: int = 1,
    ) -> Dict[str, Any]:
        refs, hyps, details = [], [], []
        latencies: List[float] = []
        failures = 0
        total = len(self.records)

        summary_path = details_path = None
        stream = None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            summary_path = os.path.join(save_dir, "summary.json")
            details_path = os.path.join(save_dir, "details.json")
            stream_path = stream_file or os.path.join(save_dir, "details.jsonl")
            stream = open(stream_path, "w", encoding="utf-8")

        parallel = workers > 1 and len(self.transcribes) > 1
        if workers > 1 and not parallel:
            log_and_print(f"[{label}] warning: {workers} workers requested but only one device lane; running sequentially")
        outcomes = self._run_parallel(workers) if parallel else (self._run_one(rec) for rec in self.records)

        try:
            for i, (rec, outcome) in enumerate(zip(self.records, outcomes), 1):
                ref, hyp, result = outcome
                refs.append(ref)
                hyps.append(hyp)
                if is_empty(hyp):
                    failures += 1
                latency = result.get(latency_key)
                if isinstance(latency, (int, float)):
                    latencies.append(float(latency))
                detail = {
                    "index": rec.get("index"),
                    "audio_file": to_rel_path(rec["audio_file"]),
                    "reference": ref,
                    "hypothesis": hyp,
                    "correct": is_same(ref, hyp),
                    "latency_seconds": latency,
                }
                if rec.get("dialect"):
                    detail["dialect"] = rec["dialect"]
                details.append(detail)

                if stream is not None:
                    stream.write(json.dumps(detail, ensure_ascii=False) + "\n")
                    stream.flush()

                if progress and (i % progress_every == 0 or i == total):
                    interim = summarize(refs, hyps)
                    log_and_print(f"[{label}] progress {i}/{total} | CER {interim['cer']} | "
                                  f"SER {interim['ser']} | empty {failures} | last: {hyp}")

                if summary_path and (i % save_every == 0 or i == total):
                    save_json(self._build_summary(refs, hyps, latencies, failures),
                              summary_path)
        finally:
            if hasattr(outcomes, "close"):
                outcomes.close()
            if stream is not None:
                stream.close()

        summary = self._build_summary(refs, hyps, latencies, failures)
        if save_dir:
            save_json(summary, summary_path)
            save_json(details, details_path)
            logger.info("results saved -> %s", save_dir)
            log_and_print(f"[{label}] results saved -> {summary_path}")
        return summary

def report(summary: Dict[str, Any], title: str = "ASR Test") -> str:
    lines = [f"== {title} ==",
             f"samples        : {summary.get('samples')}",
             f"CER            : {summary.get('cer')}",
             f"SER            : {summary.get('ser')}"]
    if "critical_score" in summary:
        lines += [f"critical_score : {summary.get('critical_score')}",
                  f"direction_match: {summary.get('direction_match')}",
                  f"negation_match : {summary.get('negation_match')}",
                  f"quantity_recall: {summary.get('quantity_recall')}"]
    if "latency_mean_ms" in summary:
        lines.append(f"latency_mean_ms: {summary.get('latency_mean_ms')}")
    if "empty_hypotheses" in summary:
        lines.append(f"empty_hypotheses: {summary.get('empty_hypotheses')}")
    return "\n".join(lines)