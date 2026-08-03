"""Calibrate per-class confidence thresholds on a YOLO validation split."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def box_iou(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def load_truth(
    label_dir: Path,
    image_dir: Path,
) -> tuple[dict[int, dict[str, list[tuple[float, ...]]]], dict[int, int]]:
    import cv2

    truth: dict[int, dict[str, list[tuple[float, ...]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    counts: dict[int, int] = defaultdict(int)
    for image_path in sorted(image_dir.iterdir()):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"cannot read image: {image_path}")
        height, width = image.shape[:2]
        label_path = label_dir / f"{image_path.stem}.txt"
        for line in label_path.read_text(encoding="ascii").splitlines():
            class_id_text, cx_text, cy_text, w_text, h_text = line.split()
            class_id = int(class_id_text)
            cx, cy = float(cx_text), float(cy_text)
            box_width, box_height = float(w_text), float(h_text)
            box = (
                (cx - box_width / 2) * width,
                (cy - box_height / 2) * height,
                (cx + box_width / 2) * width,
                (cy + box_height / 2) * height,
            )
            truth[class_id][image_path.stem].append(box)
            counts[class_id] += 1
    return truth, dict(counts)


def best_threshold(
    predictions: list[tuple[float, str, tuple[float, ...]]],
    truth_by_frame: dict[str, list[tuple[float, ...]]],
    *,
    truth_count: int,
    iou_threshold: float,
    beta: float,
    floor: float,
    minimum_precision: float,
) -> dict:
    matched: dict[str, set[int]] = defaultdict(set)
    true_positives = 0
    false_positives = 0
    best = None
    beta_squared = beta * beta
    ordered = sorted(predictions, key=lambda item: item[0], reverse=True)
    for index, (score, frame_id, box) in enumerate(ordered):
        if score < floor:
            break
        candidates = []
        for truth_index, truth_box in enumerate(truth_by_frame.get(frame_id, [])):
            if truth_index in matched[frame_id]:
                continue
            overlap = box_iou(box, truth_box)
            if overlap >= iou_threshold:
                candidates.append((overlap, truth_index))
        if candidates:
            _, truth_index = max(candidates)
            matched[frame_id].add(truth_index)
            true_positives += 1
        else:
            false_positives += 1
        next_score = ordered[index + 1][0] if index + 1 < len(ordered) else -1.0
        if next_score == score:
            continue
        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / max(truth_count, 1)
        denominator = beta_squared * precision + recall
        f_beta = (
            (1 + beta_squared) * precision * recall / denominator
            if denominator
            else 0.0
        )
        if precision < minimum_precision:
            continue
        candidate = {
            "threshold": max(floor, float(score)),
            "precision": precision,
            "recall": recall,
            f"f{beta:g}": f_beta,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "truth": truth_count,
        }
        if best is None or candidate[f"f{beta:g}"] > best[f"f{beta:g}"]:
            best = candidate
    if best is None:
        return {
            "threshold": floor,
            "precision": 0.0,
            "recall": 0.0,
            f"f{beta:g}": 0.0,
            "true_positives": 0,
            "false_positives": 0,
            "truth": truth_count,
        }
    return best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch", type=int, default=96)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--device", default="0")
    parser.add_argument("--inference-floor", type=float, default=0.01)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--minimum-precision", type=float, default=0.45)
    return parser.parse_args()


def main() -> int:
    from ultralytics import YOLO

    args = parse_args()
    image_dir = args.dataset / "images" / "val"
    label_dir = args.dataset / "labels" / "val"
    truth, truth_counts = load_truth(label_dir, image_dir)
    model = YOLO(str(args.weights))
    predictions: dict[int, list[tuple[float, str, tuple[float, ...]]]] = (
        defaultdict(list)
    )
    results = model.predict(
        source=str(image_dir),
        stream=True,
        imgsz=args.image_size,
        conf=args.inference_floor,
        iou=0.7,
        max_det=300,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        verbose=False,
    )
    names = model.names
    for result in results:
        if result.boxes is None:
            continue
        frame_id = Path(result.path).stem
        for box, score, class_id in zip(
            result.boxes.xyxy.detach().cpu().tolist(),
            result.boxes.conf.detach().cpu().tolist(),
            result.boxes.cls.detach().cpu().int().tolist(),
        ):
            predictions[class_id].append(
                (float(score), frame_id, tuple(float(value) for value in box))
            )
    calibration = {}
    thresholds = {}
    for class_id, name in sorted(names.items()):
        metrics = best_threshold(
            predictions[class_id],
            truth.get(class_id, {}),
            truth_count=truth_counts.get(class_id, 0),
            iou_threshold=args.iou_threshold,
            beta=args.beta,
            floor=args.inference_floor,
            minimum_precision=args.minimum_precision,
        )
        calibration[name] = {
            key: round(value, 6) if isinstance(value, float) else value
            for key, value in metrics.items()
        }
        thresholds[name] = calibration[name]["threshold"]
    output = {
        "weights": str(args.weights),
        "dataset": str(args.dataset),
        "objective": f"maximize F{args.beta:g} with precision >= {args.minimum_precision}",
        "inference_floor": args.inference_floor,
        "iou_threshold": args.iou_threshold,
        "by_category": thresholds,
        "calibration_metrics": calibration,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
