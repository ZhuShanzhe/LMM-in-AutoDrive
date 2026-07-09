#!/usr/bin/env python3
import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmdet.datasets import build_dataset
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


COMMAND_NAMES = {0: "turn_right", 1: "turn_left", 2: "go_straight"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="projects/configs/sparsedrive_small_stage2_mini.py",
    )
    parser.add_argument("--results", default="work_dirs/mini/results.pkl")
    parser.add_argument("--work-dir", default="work_dirs/mini")
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def read_text(path):
    path = Path(path)
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def first_float(text, pattern):
    match = re.search(pattern, text, flags=re.MULTILINE)
    return float(match.group(1)) if match else None


def parse_summary_metrics(work_dir):
    evaluation = read_text(work_dir / "evaluation.log")
    benchmark = read_text(work_dir / "benchmark.log")
    metrics = {}

    patterns = {
        "perception/mAP": r"^mAP:\s*([-+0-9.eE]+)",
        "perception/mATE": r"^mATE:\s*([-+0-9.eE]+)",
        "perception/mASE": r"^mASE:\s*([-+0-9.eE]+)",
        "perception/mAOE": r"^mAOE:\s*([-+0-9.eE]+)",
        "perception/mAVE": r"^mAVE:\s*([-+0-9.eE]+)",
        "perception/mAAE": r"^mAAE:\s*([-+0-9.eE]+)",
        "perception/NDS": r"^NDS:\s*([-+0-9.eE]+)",
        "tracking/AMOTA": r"^AMOTA:\s*([-+0-9.eE]+)",
        "tracking/AMOTP": r"^AMOTP:\s*([-+0-9.eE]+)",
        "tracking/recall": r"^RECALL:\s*([-+0-9.eE]+)",
        "tracking/MOTAR": r"^MOTAR:\s*([-+0-9.eE]+)",
        "tracking/MOTA": r"^MOTA:\s*([-+0-9.eE]+)",
        "tracking/MOTP": r"^MOTP:\s*([-+0-9.eE]+)",
        "tracking/IDS": r"^IDS:\s*([-+0-9.eE]+)",
        "mapping/ped_crossing": r"^ped_crossing=\s*([-+0-9.eE]+)",
        "mapping/divider": r"^divider=\s*([-+0-9.eE]+)",
        "mapping/boundary": r"^boundary=\s*([-+0-9.eE]+)",
        "mapping/mAP": r"^mAP_normal=\s*([-+0-9.eE]+)",
        "planning/L2": r"^L2:\s*([-+0-9.eE]+)",
        "planning/collision_percent": r"^obj_box_col:\s*([-+0-9.eE]+)%",
        "performance/FPS": r"Overall fps:\s*([-+0-9.eE]+)",
    }
    for name, pattern in patterns.items():
        value = first_float(evaluation if name != "performance/FPS" else benchmark, pattern)
        if value is not None:
            metrics[name] = value

    paired_patterns = {
        "motion/EPA": r"^epa=\s*([-+0-9.eE]+)\s*/\s*([-+0-9.eE]+)",
        "motion/minADE": r"^ade=\s*([-+0-9.eE]+)\s*/\s*([-+0-9.eE]+)",
        "motion/minFDE": r"^fde=\s*([-+0-9.eE]+)\s*/\s*([-+0-9.eE]+)",
        "motion/miss_rate": r"^mr=\s*([-+0-9.eE]+)\s*/\s*([-+0-9.eE]+)",
    }
    for name, pattern in paired_patterns.items():
        match = re.search(pattern, evaluation, flags=re.MULTILINE)
        if match:
            metrics[f"{name}_car"] = float(match.group(1))
            metrics[f"{name}_pedestrian"] = float(match.group(2))

    memory_values = [
        int(value)
        for value in re.findall(r"gpu mem:\s*(\d+)\s*M", benchmark)
    ]
    if memory_values:
        metrics["performance/peak_torch_memory_mb"] = max(memory_values)
    if "performance/FPS" in metrics and metrics["performance/FPS"] > 0:
        metrics["performance/mean_latency_ms"] = 1000.0 / metrics["performance/FPS"]
    return metrics


def parse_gpu_csv(path):
    path = Path(path)
    if not path.exists():
        return {}
    memory, utilization, power = [], [], []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            for key, target in [
                ("memory.used [MiB]", memory),
                ("utilization.gpu [%]", utilization),
                ("power.draw [W]", power),
            ]:
                raw = row.get(key)
                if raw:
                    match = re.search(r"[-+0-9.]+", raw)
                    if match:
                        target.append(float(match.group(0)))
    metrics = {}
    if memory:
        metrics["performance/nvidia_smi_peak_memory_mb"] = max(memory)
    if utilization:
        metrics["performance/mean_gpu_utilization_percent"] = float(np.mean(utilization))
        metrics["performance/peak_gpu_utilization_percent"] = max(utilization)
    if power:
        metrics["performance/mean_power_w"] = float(np.mean(power))
        metrics["performance/peak_power_w"] = max(power)
    return metrics


def case_collision(prediction, ground_truth, mask, future_boxes):
    from projects.mmdet3d_plugin.datasets.evaluation.planning.planning_eval import (
        PlanningMetric,
    )

    boxes = []
    for step in range(6):
        if step < len(future_boxes):
            value = np.asarray(future_boxes[step], dtype=np.float32)
            if value.size == 0:
                value = np.zeros((0, 7), dtype=np.float32)
        else:
            value = np.zeros((0, 7), dtype=np.float32)
        boxes.append([torch.from_numpy(value)])

    metric = PlanningMetric()
    pred_tensor = torch.from_numpy(prediction.copy()).float().unsqueeze(0)
    gt_tensor = torch.from_numpy(ground_truth.copy()).float().unsqueeze(0)
    mask_tensor = torch.from_numpy(
        np.repeat(mask[:, None], 2, axis=1)[None].astype(np.float32)
    )
    metric.update(pred_tensor, gt_tensor, mask_tensor, boxes)
    return metric.obj_box_col.numpy().astype(bool)


def build_case_rows(config_path, results_path):
    import projects.mmdet3d_plugin  # noqa: F401

    cfg = Config.fromfile(str(config_path))
    dataset = build_dataset(cfg.data.val)
    results = mmcv.load(str(results_path))
    rows = []

    if len(dataset) != len(results):
        raise RuntimeError(f"Dataset/results length mismatch: {len(dataset)} != {len(results)}")

    for index, result in enumerate(results):
        info = dataset.get_data_info(index)
        mask = np.asarray(info["gt_ego_fut_masks"], dtype=bool)[:6]
        pred = as_numpy(result["img_bbox"]["final_planning"])[:6, :2].astype(np.float32)
        gt = np.cumsum(
            np.asarray(info["gt_ego_fut_trajs"], dtype=np.float32)[:6, :2],
            axis=0,
        )
        valid = len(mask) == 6 and bool(mask.all()) and len(pred) == 6 and len(gt) == 6
        l2 = np.linalg.norm(pred - gt, axis=1) if valid else np.full(6, np.nan)
        cumulative_l2 = np.cumsum(l2) / np.arange(1, 7) if valid else l2
        official_avg = (
            float(np.mean(cumulative_l2[[1, 3, 5]])) if valid else float("nan")
        )
        collisions = (
            case_collision(pred, gt, mask, info.get("fut_boxes", []))
            if valid
            else np.zeros(6, dtype=bool)
        )
        command_id = int(np.argmax(np.asarray(info["gt_ego_fut_cmd"])))
        planning_score = result["img_bbox"].get("planning_score")
        score_max = (
            float(np.max(as_numpy(planning_score)))
            if planning_score is not None
            else float("nan")
        )
        rows.append(
            {
                "index": index,
                "token": info.get("token", ""),
                "command": COMMAND_NAMES.get(command_id, str(command_id)),
                "valid": valid,
                "l2_0_5s": float(l2[0]),
                "l2_1_0s": float(l2[1]),
                "l2_1_5s": float(l2[2]),
                "l2_2_0s": float(l2[3]),
                "l2_2_5s": float(l2[4]),
                "l2_3_0s": float(l2[5]),
                "l2_official_avg": official_avg,
                "collision_any": bool(collisions.any()),
                "collision_steps": json.dumps(
                    [round((step + 1) * 0.5, 1) for step, hit in enumerate(collisions) if hit]
                ),
                "planning_score_max": score_max,
                "image_paths": json.dumps(info.get("img_filename", []), ensure_ascii=False),
                "predicted_trajectory": json.dumps(pred.tolist()),
                "ground_truth_trajectory": json.dumps(gt.tolist()),
                "case_label": "",
                "reason": "",
            }
        )
    return rows


def select_cases(rows, top_k):
    valid = [row for row in rows if row["valid"]]
    successes = sorted(
        [row for row in valid if not row["collision_any"]],
        key=lambda row: row["l2_official_avg"],
    )[:top_k]

    failure_pool = sorted(
        valid,
        key=lambda row: (not row["collision_any"], -row["l2_official_avg"]),
    )
    success_indices = {row["index"] for row in successes}
    failures = [
        row for row in failure_pool if row["index"] not in success_indices
    ][:top_k]

    for row in successes:
        row["case_label"] = "success"
        row["reason"] = (
            "No predicted collision and low planning trajectory error "
            f"(official-style L2={row['l2_official_avg']:.4f} m)."
        )
    for row in failures:
        row["case_label"] = "failure"
        if row["collision_any"]:
            row["reason"] = (
                "Predicted ego trajectory intersects a future object box; "
                f"collision times={row['collision_steps']}, "
                f"L2={row['l2_official_avg']:.4f} m."
            )
        else:
            row["reason"] = (
                "No collision was detected, but this sample has one of the largest "
                f"planning trajectory errors (L2={row['l2_official_avg']:.4f} m)."
            )
    return successes, failures


def write_csv(path, rows):
    path = Path(path)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def copy_case_images(work_dir, selected):
    source = work_dir / "visualization" / "combine"
    target = work_dir / "cases"
    target.mkdir(parents=True, exist_ok=True)
    copied = []
    for row in selected:
        src = source / f"{row['index']:04d}.jpg"
        dst = target / f"{row['case_label']}_{row['index']:04d}.jpg"
        if src.exists():
            shutil.copy2(src, dst)
            copied.append((row, dst))
    return copied


def write_tensorboard(work_dir, metrics, rows, copied_images):
    log_dir = work_dir / "tensorboard"
    writer = SummaryWriter(str(log_dir))
    for name, value in sorted(metrics.items()):
        if value is not None and np.isfinite(value):
            writer.add_scalar(name, value, 0)

    valid_l2 = [row["l2_official_avg"] for row in rows if row["valid"]]
    if valid_l2:
        writer.add_histogram(
            "planning/per_sample_l2_official_avg",
            np.asarray(valid_l2, dtype=np.float32),
            0,
        )
    for row, image_path in copied_images:
        image = np.asarray(Image.open(image_path).convert("RGB"))
        writer.add_image(
            f"cases/{row['case_label']}/sample_{row['index']:04d}",
            image,
            0,
            dataformats="HWC",
        )
        writer.add_text(
            f"cases/{row['case_label']}/sample_{row['index']:04d}_details",
            row["reason"],
            0,
        )
    writer.flush()
    writer.close()
    return log_dir


def main():
    args = parse_args()
    repo_root = Path.cwd().resolve()
    sys.path.insert(0, str(repo_root))
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    metrics = parse_summary_metrics(work_dir)
    metrics.update(parse_gpu_csv(work_dir / "gpu_usage.csv"))
    rows = build_case_rows(Path(args.config), Path(args.results))
    successes, failures = select_cases(rows, args.top_k)
    selected = successes + failures

    valid_rows = [row for row in rows if row["valid"]]
    if valid_rows:
        metrics["dataset/validation_samples"] = len(rows)
        metrics["planning/valid_samples"] = len(valid_rows)
        metrics["planning/per_sample_mean_l2"] = float(
            np.mean([row["l2_official_avg"] for row in valid_rows])
        )
        metrics["planning/per_sample_collision_rate_percent"] = 100.0 * float(
            np.mean([row["collision_any"] for row in valid_rows])
        )

    write_csv(work_dir / "cases_all.csv", rows)
    write_csv(work_dir / "selected_cases.csv", selected)
    write_csv(
        work_dir / "metrics.csv",
        [{"metric": key, "value": value} for key, value in sorted(metrics.items())],
    )
    (work_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    copied_images = copy_case_images(work_dir, selected)
    log_dir = write_tensorboard(work_dir, metrics, rows, copied_images)

    print(f"Wrote {work_dir / 'metrics.json'}")
    print(f"Wrote {work_dir / 'metrics.csv'}")
    print(f"Wrote {work_dir / 'cases_all.csv'}")
    print(f"Wrote {work_dir / 'selected_cases.csv'}")
    print(f"Copied {len(copied_images)} selected case images to {work_dir / 'cases'}")
    print(f"TensorBoard log dir: {log_dir}")
    print("Selected success indices:", [row["index"] for row in successes])
    print("Selected failure indices:", [row["index"] for row in failures])


if __name__ == "__main__":
    main()
