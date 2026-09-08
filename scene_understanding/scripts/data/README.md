# 场景理解数据准备

以下命令均从仓库根目录执行。数据与模型不提交到 Git；路径通过参数传入，清单内的
Waymo 图像路径为相对路径，便于把数据目录整体移动到 Docker 挂载点。

```bash
python -m pip install -r scene_understanding/requirements-data.txt
DATA_ROOT=${DATA_ROOT:-../datasets/scene_understanding}
mkdir -p "$DATA_ROOT"
```

## nuScenes

完整 trainval 可使用原下载脚本；开发与回归可使用官方 mini 包。不要把
`v1.0-mini` 当成正式泛化指标，只用于验证转换和短程微调。

```bash
source /etc/network_turbo
bash scene_understanding/scripts/data/download_nuscenes.sh

python scene_understanding/scripts/data/prepare_nuscenes.py \
  --dataroot "$DATA_ROOT/nuscenes/raw" \
  --output "$DATA_ROOT/nuscenes/organized" \
  --version v1.0-trainval

# mini 包对应：--version v1.0-mini
```

转换器使用官方 scene split，把六路相机的 3D 标注投影为归一化 2D 框。统一类别为
`vehicle/pedestrian/cyclist/motorcycle/road_barrier/traffic_cone`。

## Waymo Open Dataset v2

先按 Waymo 条款取得授权，并用 Google Cloud CLI 登录默认账号。Google 下载不要执行
`source /etc/network_turbo`，该代理只面向 GitHub/Hugging Face。仓库配置不是盲选：它
由官方 `stats` 组件的 798 个训练片段和 199 个验证片段统计后确定，优先覆盖行人、
骑行者和 Dawn/Dusk，且验证片段与训练片段独立。

```bash
python scene_understanding/scripts/data/download_waymo_v2_subset.py \
  --config scene_understanding/configs/waymo_v2_perception_subset.json \
  --output-dir "$DATA_ROOT/waymo_v2_subset" \
  --gcloud-bin "${GCLOUD_BIN:-gcloud}"

python scene_understanding/scripts/data/prepare_waymo_v2.py \
  --input-root "$DATA_ROOT/waymo_v2_subset" \
  --output "$DATA_ROOT/waymo_v2_organized"
```

下载器只取 `camera_image/camera_box/camera_calibration/stats`，逐文件计算 SHA-256；
转换器按 `(segment_context_name, frame_timestamp_micros, camera_name)` 精确连接图像与
框，并保留 Waymo 的车辆、行人、骑行者类别。JPEG 不重复编码。

## 构建和训练八类 YOLO 数据集

```bash
python scene_understanding/training/build_manifest_yolo_dataset.py \
  --train-manifest "$DATA_ROOT/waymo_v2_organized/manifests/training.jsonl" \
  --train-manifest "$DATA_ROOT/nuscenes/organized/manifests/train_cam_front.jsonl" \
  --val-manifest "$DATA_ROOT/waymo_v2_organized/manifests/validation.jsonl" \
  --val-manifest "$DATA_ROOT/nuscenes/organized/manifests/val_cam_front.jsonl" \
  --output "$DATA_ROOT/driving_yolo_v2"

python scene_understanding/training/train_yolo11_specialized.py \
  --data "$DATA_ROOT/driving_yolo_v2/dataset.yaml" \
  --model "${BASE_DETECTOR:?set BASE_DETECTOR to the submitted checkpoint}" \
  --project "${MODEL_ROOT:-../models}/scene_understanding" \
  --name yolo11s_driving_v2 \
  --epochs 15 --batch 24 --learning-rate 0.001 --device 0
```

八类顺序固定为：车辆、行人、骑行者、摩托车、交通灯、交通标志、道路护栏、
交通锥。Waymo/nuScenes 不标注全部八类，因此必须从已有八类 checkpoint 低学习率
微调，并同时保留 CARLA 场景回归；不能只看外部数据验证集 mAP。

## 质量门

- 训练/验证按场景或 segment 分离，禁止相邻帧跨 split。
- 清单中的图像必须存在，框必须在 `[0,1]` 内且宽高为正。
- 外部数据 mAP 与 CARLA 场景存在性召回分别报告；后者不是 2D mAP 的替代品。
- 数据、最终权重和 Docker 镜像单独提交，仓库只保存配置、转换代码、测试与清单格式。
