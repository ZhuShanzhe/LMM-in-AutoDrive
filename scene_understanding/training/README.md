# 专项检测器训练

目标类别固定为：

```text
vehicle, pedestrian, cyclist, motorcycle,
traffic_light, traffic_sign, road_barrier, traffic_cone
```

数据和权重放在 `/root/autodl-tmp`，不提交 Git。

## 1. 构建通用训练集

```bash
python -m scene_understanding.training.build_specialized_yolo_dataset \
  --bdd-train /root/autodl-tmp/datasets/scene_understanding/bdd100k/organized/manifests/train.jsonl \
  --bdd-val /root/autodl-tmp/datasets/scene_understanding/bdd100k/organized/manifests/val.jsonl \
  --nuscenes-train /root/autodl-tmp/datasets/scene_understanding/nuscenes_full/organized/manifests/train_cam_front.jsonl \
  --nuscenes-train /root/autodl-tmp/datasets/scene_understanding/nuscenes_full/organized/manifests/train_cam_front_left.jsonl \
  --nuscenes-train /root/autodl-tmp/datasets/scene_understanding/nuscenes_full/organized/manifests/train_cam_front_right.jsonl \
  --nuscenes-val /root/autodl-tmp/datasets/scene_understanding/nuscenes_full/organized/manifests/val_cam_front.jsonl \
  --output /root/autodl-tmp/datasets/scene_understanding/specialized_yolo_v1
```

- 训练集：50,000 张、804,695 个框；
- 验证集：5,000 张、66,974 个框；
- 训练/验证原图交集：0；
- 坏链接、越界坐标和非法类别：0。

## 2. 训练通用 YOLO11s

```bash
python -m scene_understanding.training.train_yolo11_specialized \
  --data /root/autodl-tmp/datasets/scene_understanding/specialized_yolo_v1/dataset.yaml \
  --model /root/autodl-tmp/models/YOLO11/yolo11s.pt \
  --project /root/autodl-tmp/models/scene_understanding \
  --name yolo11s_specialized_v1 \
  --epochs 40 --batch 48 --image-size 640
```

最佳权重：

```text
/root/autodl-tmp/models/scene_understanding/yolo11s_specialized_v1/weights/best.pt
```

验证结果：

| Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|
| 69.71% | 49.81% | 54.82% | 26.99% |

## 3. 构建 CARLA 域适配集

先用 `experiment/CARLA/run_control_experiment.py --scene-capture` 采集训练场景。
交通灯必须使用修正后的 `TrafficLight.get_light_boxes()` 逐灯头投影。不要把最终
straight 测试场景的帧加入训练。

```bash
python -m scene_understanding.training.build_carla_domain_dataset \
  --capture-index experiment/CARLA/outputs/runs/domain_adaptation_v1/pedestrian_crossing/scene_understanding/capture_index.jsonl \
  --capture-index experiment/CARLA/outputs/runs/domain_adaptation_v1/emergency_brake/scene_understanding/capture_index.jsonl \
  --base-dataset /root/autodl-tmp/datasets/scene_understanding/specialized_yolo_v1 \
  --output /root/autodl-tmp/datasets/scene_understanding/specialized_yolo_carla_v1 \
  --train-repeats 30 \
  --base-train-limit 12000
```

当前构建结果：

- CARLA 源帧 86：训练 69、域内验证 17；
- 训练帧重复增强采样后为 2,070 项、12,270 个框；
- 混合 12,000 张固定抽样的原训练图，降低灾难性遗忘；
- 混合验证集为原验证 5,000 张加 CARLA 17 帧；
- straight 36 帧完全隔离用于最终测试。

构建日志必须显示真实标签数量。若出现大批 `backgrounds`，先检查
`images/` 到 `labels/` 的路径映射，不得继续训练。

## 4. 训练 CARLA 域适配权重

```bash
python -m scene_understanding.training.train_yolo11_specialized \
  --data /root/autodl-tmp/datasets/scene_understanding/specialized_yolo_carla_v1/dataset.yaml \
  --model /root/autodl-tmp/models/scene_understanding/yolo11s_specialized_v1/weights/best.pt \
  --project /root/autodl-tmp/models/scene_understanding \
  --name yolo11s_specialized_carla_v1 \
  --epochs 12 --batch 48 --image-size 640
```

最佳权重：

```text
/root/autodl-tmp/models/scene_understanding/yolo11s_specialized_carla_v1/weights/best.pt
```

混合验证结果：

| Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|
| 67.86% | 49.74% | 54.05% | 26.43% |

域适配模型在隔离 CARLA straight 上把灯头召回/精度从通用模型 640 的
13.87%/47.69% 提高到 17.67%/56.43%。它是 CARLA 部署权重，不替代通用研究权重。

## 5. 独立评测

```bash
python -m scene_understanding.realtime_perception.run_dataset \
  --manifest /path/to/test.jsonl \
  --output /path/to/perception.jsonl \
  --summary /path/to/summary.json \
  --backend yolop_yolo11 \
  --yolo11-weights /root/autodl-tmp/models/scene_understanding/yolo11s_specialized_carla_v1/weights/best.pt \
  --image-size 640 \
  --object-image-size 640 \
  --score-threshold 0.10

python -m scene_understanding.realtime_perception.evaluate_dataset \
  --results /path/to/perception.jsonl \
  --manifest /path/to/test.jsonl \
  --output /path/to/metrics.json \
  --iou-threshold 0.5
```

推荐配置：

- CARLA 实时稳健：`--object-image-size 640`，CARLA P95 41.36 ms；
- 独占 RTX 5090 高精度：`--object-image-size 768`，CARLA P95 48.14 ms；
- `--infrastructure-tiles`：仅离线分析，CARLA P95 60.01 ms；
- 置信度保持 0.10，0.05/0.025 未得到更好折中。

完整独立集结果见 `../PERCEPTION_EXPERIMENT_REPORT.md`。

## 6. 许可

权重源自 Ultralytics YOLO11，分发和部署须遵守 AGPL-3.0 或相应商业许可。BDD100K、
nuScenes 和 CARLA 数据只保存在数据盘，不随代码或权重仓库重新分发。
