"""Exact-frame shadow evaluation for semantic and control predictions.

This evaluator never actuates CARLA.  It joins model predictions to independent
ground truth by ``simulation_frame`` and refuses claim status when observed
truth coverage or exact-frame prediction coverage is insufficient.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

try:
    from evaluation.ground_truth import (
        CONTROL_ACTIONS,
        FRAME_GROUND_TRUTH_SCHEMA,
        observed_control_contract,
        observed_event_ids,
        observed_risk_labels,
        unique_frames,
    )
except ImportError:
    from ground_truth import (
        CONTROL_ACTIONS,
        FRAME_GROUND_TRUTH_SCHEMA,
        observed_control_contract,
        observed_event_ids,
        observed_risk_labels,
        unique_frames,
    )


SHADOW_REPORT_SCHEMA = "ShadowEvaluationReport/1.0.0"
SEMANTIC_PREDICTION_SCHEMA = "SemanticPrediction/1.0.0"
CONTROL_SHADOW_SCHEMA = "ControlDecisionShadow/1.0.0"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as stream:
        for line_number, line in enumerate(
            stream,
            start=1,
        ):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    "{0}:{1}: invalid JSON: {2}".format(
                        path,
                        line_number,
                        error,
                    )
                ) from error
            if not isinstance(payload, dict):
                raise ValueError(
                    "{0}:{1}: JSONL record must be an object".format(
                        path,
                        line_number,
                    )
                )
            if "simulation_frame" not in payload:
                raise ValueError(
                    "{0}:{1}: simulation_frame is required".format(
                        path,
                        line_number,
                    )
                )
            records.append(payload)
    return records


def _percentile(
    values: Sequence[float],
    percentile: float,
) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return (
        ordered[lower] * (1.0 - fraction)
        + ordered[upper] * fraction
    )


def _ratio(
    numerator: int,
    denominator: int,
) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 6)


def _set_scores(
    truth_sets: Iterable[set[str]],
    predicted_sets: Iterable[set[str]],
) -> dict[str, Any]:
    true_positive = 0
    false_positive = 0
    false_negative = 0
    exact = 0
    samples = 0
    labels: set[str] = set()
    truth_list = list(truth_sets)
    prediction_list = list(predicted_sets)
    for truth, predicted in zip(
        truth_list,
        prediction_list,
    ):
        labels.update(truth)
        labels.update(predicted)
        true_positive += len(truth & predicted)
        false_positive += len(predicted - truth)
        false_negative += len(truth - predicted)
        exact += int(truth == predicted)
        samples += 1
    precision = _ratio(
        true_positive,
        true_positive + false_positive,
    )
    recall = _ratio(
        true_positive,
        true_positive + false_negative,
    )
    f1 = None
    if (
        precision is not None
        and recall is not None
        and precision + recall > 0.0
    ):
        f1 = round(
            2.0
            * precision
            * recall
            / (precision + recall),
            6,
        )
    per_label: dict[str, Any] = {}
    for label in sorted(labels):
        label_tp = sum(
            label in truth and label in predicted
            for truth, predicted in zip(
                truth_list,
                prediction_list,
            )
        )
        label_fp = sum(
            label not in truth and label in predicted
            for truth, predicted in zip(
                truth_list,
                prediction_list,
            )
        )
        label_fn = sum(
            label in truth and label not in predicted
            for truth, predicted in zip(
                truth_list,
                prediction_list,
            )
        )
        label_precision = _ratio(
            label_tp,
            label_tp + label_fp,
        )
        label_recall = _ratio(
            label_tp,
            label_tp + label_fn,
        )
        label_f1 = None
        if (
            label_precision is not None
            and label_recall is not None
            and label_precision + label_recall > 0.0
        ):
            label_f1 = round(
                2.0
                * label_precision
                * label_recall
                / (
                    label_precision
                    + label_recall
                ),
                6,
            )
        per_label[label] = {
            "precision": label_precision,
            "recall": label_recall,
            "f1": label_f1,
            "support": sum(
                label in truth
                for truth in truth_list
            ),
        }
    supported_f1 = [
        metrics["f1"]
        for metrics in per_label.values()
        if (
            metrics["support"] > 0
            and metrics["f1"] is not None
        )
    ]
    return {
        "samples": samples,
        "exact_match_accuracy": _ratio(
            exact,
            samples,
        ),
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "macro_f1": (
            round(
                sum(supported_f1)
                / len(supported_f1),
                6,
            )
            if supported_f1
            else None
        ),
        "per_label": per_label,
    }


def _prediction_set(
    prediction: Mapping[str, Any],
    plural_key: str,
    singular_key: str,
) -> set[str]:
    value = prediction.get(plural_key)
    if value is None:
        singular = prediction.get(singular_key)
        return (
            {str(singular)}
            if singular not in (None, "")
            else set()
        )
    if not isinstance(value, list):
        raise ValueError(
            plural_key + " must be a list"
        )
    return {
        str(item)
        for item in value
        if item not in (None, "")
    }


def _validate_ground_truth_records(
    records: Sequence[Mapping[str, Any]],
) -> str:
    if not records:
        raise ValueError(
            "ground-truth input is empty"
        )
    scene_ids: set[str] = set()
    for record in records:
        if (
            record.get("schema_version")
            != FRAME_GROUND_TRUTH_SCHEMA
        ):
            raise ValueError(
                "unsupported ground-truth schema_version"
            )
        scene_id = str(record.get("scene_id", ""))
        if not scene_id:
            raise ValueError(
                "ground-truth scene_id is required"
            )
        scene_ids.add(scene_id)
        provenance = record.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError(
                "ground-truth provenance is required"
            )
        if provenance.get("model_output_used") is not False:
            raise ValueError(
                "ground truth must be independent "
                "from model output"
            )
        if (
            provenance.get(
                "adjacent_frame_fill_used"
            )
            is not False
        ):
            raise ValueError(
                "adjacent-frame ground-truth fill is forbidden"
            )
    if len(scene_ids) != 1:
        raise ValueError(
            "ground-truth input must contain one scene_id"
        )
    return next(iter(scene_ids))


def _validate_semantic_predictions(
    records: Sequence[Mapping[str, Any]],
    scene_id: str,
) -> None:
    for record in records:
        if (
            record.get("schema_version")
            != SEMANTIC_PREDICTION_SCHEMA
        ):
            raise ValueError(
                "unsupported semantic prediction schema_version"
            )
        if record.get("scene_id") != scene_id:
            raise ValueError(
                "semantic prediction scene_id mismatch"
            )
        for name in (
            "active_event_ids",
            "risk_labels",
        ):
            if not isinstance(record.get(name), list):
                raise ValueError(
                    "semantic prediction "
                    + name
                    + " must be a list"
                )


def _validate_control_decisions(
    records: Sequence[Mapping[str, Any]],
    scene_id: str,
) -> None:
    for record in records:
        if (
            record.get("schema_version")
            != CONTROL_SHADOW_SCHEMA
        ):
            raise ValueError(
                "unsupported control shadow schema_version"
            )
        if record.get("scene_id") != scene_id:
            raise ValueError(
                "control shadow scene_id mismatch"
            )
        action = str(
            record.get("action", "")
        ).lower()
        if action not in CONTROL_ACTIONS:
            raise ValueError(
                "unsupported control shadow action: "
                + action
            )
        if record.get("safety_gate_status") not in {
            "APPROVED",
            "OVERRIDDEN",
            "REJECTED",
        }:
            raise ValueError(
                "invalid control shadow safety_gate_status"
            )
        latency = record.get("latency_ms")
        if latency is None or float(latency) < 0.0:
            raise ValueError(
                "control shadow latency_ms must be non-negative"
            )


def _coverage_status(
    *,
    observed_frames: int,
    observed_events: int,
    matched_frames: int,
    minimum_observed_frames: int,
    minimum_observed_events: int,
    minimum_prediction_coverage: float,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if observed_frames < minimum_observed_frames:
        reasons.append(
            "observed truth frames {0} < required {1}".format(
                observed_frames,
                minimum_observed_frames,
            )
        )
    if observed_events < minimum_observed_events:
        reasons.append(
            "observed event classes {0} < required {1}".format(
                observed_events,
                minimum_observed_events,
            )
        )
    prediction_coverage = (
        matched_frames / observed_frames
        if observed_frames
        else 0.0
    )
    if prediction_coverage < minimum_prediction_coverage:
        reasons.append(
            "exact-frame prediction coverage "
            "{0:.3f} < required {1:.3f}".format(
                prediction_coverage,
                minimum_prediction_coverage,
            )
        )
    return (
        (
            "MEASURED"
            if not reasons
            else "INSUFFICIENT_EVIDENCE"
        ),
        reasons,
    )


def evaluate_shadow_records(
    *,
    ground_truth_records: Sequence[Mapping[str, Any]],
    semantic_predictions: (
        Sequence[Mapping[str, Any]] | None
    ) = None,
    control_decisions: (
        Sequence[Mapping[str, Any]] | None
    ) = None,
    minimum_observed_frames: int = 30,
    minimum_observed_events: int = 3,
    minimum_prediction_coverage: float = 0.95,
) -> dict[str, Any]:
    scene_id = _validate_ground_truth_records(
        ground_truth_records
    )
    if semantic_predictions is not None:
        _validate_semantic_predictions(
            semantic_predictions,
            scene_id,
        )
    if control_decisions is not None:
        _validate_control_decisions(
            control_decisions,
            scene_id,
        )
    truth = unique_frames(ground_truth_records)
    semantic = (
        unique_frames(semantic_predictions)
        if semantic_predictions is not None
        else {}
    )
    controls = (
        unique_frames(control_decisions)
        if control_decisions is not None
        else {}
    )
    observed_frames = sorted(
        frame
        for frame, record in truth.items()
        if record.get("claim_eligible") is True
        and observed_event_ids(record)
    )
    observed_event_classes = sorted(
        {
            event_id
            for frame in observed_frames
            for event_id in observed_event_ids(
                truth[frame]
            )
        }
    )
    quality_counts = Counter(
        str(
            record.get(
                "frame_truth_quality",
                "UNKNOWN",
            )
        )
        for record in truth.values()
    )

    semantic_report: dict[str, Any] | None = None
    if semantic_predictions is not None:
        matched = [
            frame
            for frame in observed_frames
            if frame in semantic
        ]
        event_truth = [
            observed_event_ids(truth[frame])
            for frame in matched
        ]
        event_predictions = [
            _prediction_set(
                semantic[frame],
                "active_event_ids",
                "event_id",
            )
            for frame in matched
        ]
        risk_truth = [
            observed_risk_labels(truth[frame])
            for frame in matched
        ]
        risk_predictions = [
            _prediction_set(
                semantic[frame],
                "risk_labels",
                "risk_label",
            )
            for frame in matched
        ]
        status, reasons = _coverage_status(
            observed_frames=len(observed_frames),
            observed_events=len(
                observed_event_classes
            ),
            matched_frames=len(matched),
            minimum_observed_frames=(
                minimum_observed_frames
            ),
            minimum_observed_events=(
                minimum_observed_events
            ),
            minimum_prediction_coverage=(
                minimum_prediction_coverage
            ),
        )
        semantic_report = {
            "status": status,
            "insufficient_evidence_reasons": reasons,
            "observed_truth_frames": len(
                observed_frames
            ),
            "matched_exact_frames": len(matched),
            "exact_frame_coverage": _ratio(
                len(matched),
                len(observed_frames),
            ),
            "unknown_prediction_frames": sorted(
                frame
                for frame in semantic
                if frame not in truth
            ),
            "event_detection": _set_scores(
                event_truth,
                event_predictions,
            ),
            "risk_label_alignment": _set_scores(
                risk_truth,
                risk_predictions,
            ),
        }

    control_report: dict[str, Any] | None = None
    if control_decisions is not None:
        control_truth_frames = [
            frame
            for frame in observed_frames
            if observed_control_contract(
                truth[frame]
            )[0]
        ]
        matched = [
            frame
            for frame in control_truth_frames
            if frame in controls
        ]
        compatible = 0
        unsafe_approved = 0
        forbidden_attempts = 0
        latencies: list[float] = []
        for frame in matched:
            allowed, forbidden = (
                observed_control_contract(
                    truth[frame]
                )
            )
            decision = controls[frame]
            action = str(
                decision.get("action", "")
            ).lower()
            compatible += int(action in allowed)
            if action in forbidden:
                forbidden_attempts += 1
                if (
                    decision.get(
                        "safety_gate_status"
                    )
                    == "APPROVED"
                ):
                    unsafe_approved += 1
            latency = decision.get("latency_ms")
            if latency is not None:
                latency_value = float(latency)
                if latency_value < 0.0:
                    raise ValueError(
                        "latency_ms must be non-negative"
                    )
                latencies.append(latency_value)
        status, reasons = _coverage_status(
            observed_frames=len(
                control_truth_frames
            ),
            observed_events=len(
                observed_event_classes
            ),
            matched_frames=len(matched),
            minimum_observed_frames=(
                minimum_observed_frames
            ),
            minimum_observed_events=(
                minimum_observed_events
            ),
            minimum_prediction_coverage=(
                minimum_prediction_coverage
            ),
        )
        control_report = {
            "status": status,
            "insufficient_evidence_reasons": reasons,
            "observed_truth_frames": len(
                control_truth_frames
            ),
            "matched_exact_frames": len(matched),
            "exact_frame_coverage": _ratio(
                len(matched),
                len(control_truth_frames),
            ),
            "action_compatibility_rate": _ratio(
                compatible,
                len(matched),
            ),
            "forbidden_action_attempts": (
                forbidden_attempts
            ),
            "unsafe_action_false_approvals": (
                unsafe_approved
            ),
            "unsafe_action_false_approval_rate": (
                _ratio(
                    unsafe_approved,
                    forbidden_attempts,
                )
            ),
            "latency_ms": {
                "samples": len(latencies),
                "median": (
                    round(median(latencies), 3)
                    if latencies
                    else None
                ),
                "p95": (
                    round(
                        _percentile(
                            latencies,
                            0.95,
                        ),
                        3,
                    )
                    if latencies
                    else None
                ),
            },
        }

    return {
        "schema_version": SHADOW_REPORT_SCHEMA,
        "mode": "SHADOW_ONLY",
        "closed_loop_actuation_performed": False,
        "closed_loop_claim_allowed": False,
        "join_key": "simulation_frame",
        "adjacent_frame_fill_used": False,
        "ground_truth": {
            "records": len(truth),
            "frame_quality_counts": dict(
                sorted(quality_counts.items())
            ),
            "claim_eligible_frames": len(
                observed_frames
            ),
            "observed_event_classes": (
                observed_event_classes
            ),
        },
        "semantic_alignment": semantic_report,
        "control": control_report,
        "interpretation": (
            "MEASURED means the configured evidence and exact-frame "
            "coverage gates passed. It does not by itself prove a "
            "competition threshold or closed-loop driving success."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--semantic-predictions",
        type=Path,
    )
    parser.add_argument(
        "--control-decisions",
        type=Path,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--minimum-observed-frames",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--minimum-observed-events",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--minimum-prediction-coverage",
        type=float,
        default=0.95,
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    if (
        args.semantic_predictions is None
        and args.control_decisions is None
    ):
        raise ValueError(
            "at least one prediction file is required"
        )
    if args.minimum_observed_frames < 1:
        raise ValueError(
            "--minimum-observed-frames must be positive"
        )
    if args.minimum_observed_events < 1:
        raise ValueError(
            "--minimum-observed-events must be positive"
        )
    if not (
        0.0
        <= args.minimum_prediction_coverage
        <= 1.0
    ):
        raise ValueError(
            "--minimum-prediction-coverage must be in [0, 1]"
        )
    report = evaluate_shadow_records(
        ground_truth_records=load_jsonl(
            args.ground_truth
        ),
        semantic_predictions=(
            load_jsonl(args.semantic_predictions)
            if args.semantic_predictions is not None
            else None
        ),
        control_decisions=(
            load_jsonl(args.control_decisions)
            if args.control_decisions is not None
            else None
        ),
        minimum_observed_frames=(
            args.minimum_observed_frames
        ),
        minimum_observed_events=(
            args.minimum_observed_events
        ),
        minimum_prediction_coverage=(
            args.minimum_prediction_coverage
        ),
    )
    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        "SHADOW EVALUATION REPORT:",
        args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
