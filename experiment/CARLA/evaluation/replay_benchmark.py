"""Offline, same-source model invocation; not a closed-loop driving score."""

from __future__ import annotations

import argparse
from copy import deepcopy
import importlib
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

from evaluation.sensor_replay import SynchronizedReplayDataset


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def run(dataset: SynchronizedReplayDataset, adapter: Any, output: Path, *,
        adapter_id: str, adapter_config: dict, deadline_ms: float = 150.0,
        max_frames: int | None = None, collect_resources: bool = False,
        selection_path: Path | None = None, scene_id: str | None = None) -> dict:
    """Adapter: predict(frame)->JSON object, optional reset/synchronize/close."""
    if not math.isfinite(deadline_ms) or deadline_ms <= 0:
        raise ValueError("deadline_ms must be finite and positive")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")
    # Hash before invoking user code, preserving the original input identity.
    manifest = dataset.integrity_manifest()
    selected_frames = None
    if selection_path is not None:
        if scene_id is None or max_frames is not None:
            raise ValueError("same-source selection requires a scene and full-sequence replay")
        from evaluation.challenge_capture_audit import validate_replay_selection
        selected_frames = validate_replay_selection(selection_path, scene_id, dataset, manifest)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    synchronize = getattr(adapter, "synchronize", None)
    latencies: list[float] = []
    selected_latencies: list[float] = []
    failed = 0
    count = 0
    fatal_error = None
    close_error = None
    resource_summary = None
    resource_sampler = None
    try:
        if collect_resources:
            from evaluation.resources import ResourceSampler
            resource_sampler = ResourceSampler(output / 'resources.jsonl').start()
        reset = getattr(adapter, "reset", None)
        if reset is not None:
            reset()
        with (output / "predictions.jsonl").open("w", encoding="utf-8") as log:
            for frame in dataset:
                if max_frames is not None and count >= max_frames:
                    break
                row = {"simulation_frame": frame.simulation_frame,
                       "timestamp_s": frame.timestamp_s,
                       "request_id": frame.driving_intent_request_id}
                if selected_frames is not None:
                    row["selected_for_evaluation"] = frame.simulation_frame in selected_frames
                started = None
                try:
                    payload = deepcopy(frame)
                    if synchronize is not None:
                        synchronize()
                    started = time.perf_counter()
                    prediction = adapter.predict(payload)
                    if synchronize is not None:
                        synchronize()
                    elapsed = (time.perf_counter() - started) * 1000
                    if not isinstance(prediction, dict):
                        raise ValueError("predict must return a JSON object")
                    # Round-trip detaches mutable outputs and rejects NaN/Inf.
                    prediction = json.loads(json.dumps(prediction, allow_nan=False))
                    latencies.append(elapsed)
                    if selected_frames is not None and frame.simulation_frame in selected_frames:
                        selected_latencies.append(elapsed)
                    row.update(status="OK", latency_ms=elapsed, output=prediction,
                               deadline_exceeded=elapsed > deadline_ms)
                except Exception as error:
                    failed += 1
                    row.update(status="ERROR", error_type=type(error).__name__, error=str(error),
                               latency_ms=None if started is None else (time.perf_counter() - started) * 1000)
                log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                log.flush()
                count += 1
                # Stateful models must not silently continue after a corrupt frame.
                if row["status"] == "ERROR":
                    break
    except Exception as error:
        fatal_error = {"type": type(error).__name__, "message": str(error)}
    finally:
        if resource_sampler is not None:
            resource_summary = resource_sampler.close()
        close = getattr(adapter, "close", None)
        if close is not None:
            try:
                close()
            except Exception as error:
                close_error = {"type": type(error).__name__, "message": str(error)}
    expected = min(len(dataset), max_frames) if max_frames else len(dataset)
    selection_incomplete = selected_frames is not None and len(selected_latencies) != len(selected_frames)
    summary = {
        "schema_version": "offline_replay_benchmark/1.0",
        "status": "FAILED" if failed or fatal_error or close_error or count != expected or not expected or selection_incomplete else "COMPLETED",
        "evaluation_scope": "offline_model_invocation_only",
        "closed_loop_success": None,
        "adapter": adapter_id, "adapter_config": adapter_config,
        "dataset_sha256": manifest["dataset_sha256"],
        "dataset_frames": len(dataset), "requested_frames": expected,
        "attempted_frames": count, "successful_outputs": len(latencies), "failed_frames": failed,
        "fatal_error": fatal_error, "close_error": close_error,
        "resources": resource_summary,
        "timing_scope": "predict_with_adapter_synchronization" if synchronize else "predict_host_call_only",
        "warmup_policy": "none; first prediction included; adapter initialization excluded",
        "latency_ms": {"p50": _percentile(latencies, .5), "p95": _percentile(latencies, .95),
                       "max": max(latencies) if latencies else None},
        "deadline_ms": deadline_ms,
        "deadline_exceeded_outputs": sum(value > deadline_ms for value in latencies),
    }
    if selected_frames is not None:
        summary["evaluation_selection"] = {
            "scene_id": scene_id, "selection_file": str(selection_path),
            "requested_frames": len(selected_frames), "successful_outputs": len(selected_latencies),
            "latency_ms": {"p50": _percentile(selected_latencies, .5),
                           "p95": _percentile(selected_latencies, .95),
                           "max": max(selected_latencies) if selected_latencies else None},
            "deadline_exceeded_outputs": sum(value > deadline_ms for value in selected_latencies),
            "scope": "selected evaluation frames after full chronological model replay",
        }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                                  allow_nan=False) + "\n", encoding="utf-8")
    return summary


def run_repeated(dataset, factory, output, *, repetitions, adapter_id, adapter_config, **kwargs):
    """Fresh adapter instances with identical inputs; stop on first invalid run."""
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError('repetitions must be a positive integer')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    result = dict(schema_version='replay_repetition/1.0', requested_repetitions=repetitions,
                  runs=rows, status='RUNNING', closed_loop_success=None,
                  comparison_scope='exact JSON predictions, not semantic correctness',
                  rng_policy='factory/config responsibility; no implicit seed changes',
                  initialization_scope='fresh adapter per run in the same process; not a cold GPU process')
    def save():
        (output/'repetitions.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    save()
    for index in range(repetitions):
        directory=output/f'run_{index+1:03d}'
        try:
            adapter=factory(deepcopy(adapter_config))
            summary=run(dataset,adapter,directory,adapter_id=adapter_id,
                        adapter_config=deepcopy(adapter_config),**kwargs)
            del adapter
            digest=hashlib.sha256()
            with (directory/'predictions.jsonl').open(encoding='utf-8') as stream:
                for line in stream:
                    row=json.loads(line)
                    stable={k:row.get(k) for k in ('simulation_frame','request_id','status','output','error_type','error')}
                    digest.update((json.dumps(stable,sort_keys=True,allow_nan=False)+'\n').encode('utf-8'))
            rows.append(dict(run=index+1,status=summary['status'],dataset_sha256=summary['dataset_sha256'],
                             prediction_sha256=digest.hexdigest(),latency_ms=summary['latency_ms']))
            if summary['status']!='COMPLETED':
                result['status']='FAILED'
                break
            if rows[0]['dataset_sha256']!=summary['dataset_sha256']:
                result.update(status='FAILED',reason='input dataset changed between repetitions')
                break
        except Exception as error:
            rows.append(dict(run=index+1,status='FAILED',error=str(error),error_type=type(error).__name__))
            result['status']='FAILED'
            break
        finally:
            save()
    if result['status']=='RUNNING':
        result['status']='COMPLETED'
    result['exact_predictions_repeatable']=(len({r.get('prediction_sha256') for r in rows})==1
        if result['status']=='COMPLETED' and repetitions>1 else None)
    save()
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--adapter", required=True, help="Python module:factory")
    parser.add_argument("--config", type=Path, required=True, help="Adapter configuration JSON")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--deadline-ms", type=float, default=150)
    parser.add_argument("--allow-missing-calibration", action="store_true")
    parser.add_argument("--format", choices=("synchronized", "model-rig"), default="synchronized")
    parser.add_argument("--selection", type=Path, help="Three-scene same-source selection manifest")
    parser.add_argument("--scene", choices=("scene_1", "scene_2", "scene_3"))
    parser.add_argument("--no-resources", action="store_true")
    parser.add_argument("--repeat", type=int, default=1, help="Fresh model instances on identical replay inputs")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output directory already exists")
    if args.repeat < 1 or not math.isfinite(args.deadline_ms) or args.deadline_ms <= 0 or (args.max_frames is not None and args.max_frames <= 0):
        parser.error("deadline and frame limit must be positive finite values")
    if args.selection is not None and (args.format != "model-rig" or args.scene is None or args.max_frames is not None):
        parser.error("--selection requires --format model-rig, --scene and full-sequence replay")
    if args.format == "model-rig":
        from evaluation.model_rig_replay import ModelRigReplayDataset
        dataset = ModelRigReplayDataset(args.dataset)
    else:
        dataset = SynchronizedReplayDataset(args.dataset, require_calibration=not args.allow_missing_calibration)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        parser.error("adapter configuration must be a JSON object")
    module, separator, factory = args.adapter.partition(":")
    if not separator or not module or not factory:
        parser.error("adapter must be module:factory")
    factory_fn = getattr(importlib.import_module(module), factory)
    if args.repeat > 1:
        summary = run_repeated(dataset, factory_fn, args.output, repetitions=args.repeat,
                  adapter_id=args.adapter, adapter_config=config, deadline_ms=args.deadline_ms,
                  max_frames=args.max_frames, collect_resources=not args.no_resources,
                  selection_path=args.selection, scene_id=args.scene)
    else:
        adapter = factory_fn(deepcopy(config))
        summary = run(dataset, adapter, args.output, adapter_id=args.adapter,
                  adapter_config=config, deadline_ms=args.deadline_ms, max_frames=args.max_frames,
                  collect_resources=not args.no_resources,
                  selection_path=args.selection, scene_id=args.scene)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
