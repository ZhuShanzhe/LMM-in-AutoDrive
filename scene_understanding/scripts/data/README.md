# 场景理解数据准备

数据统一放在 `/root/autodl-tmp/datasets/scene_understanding`，不提交到 Git。

数据处理环境：

```bash
conda activate /root/autodl-tmp/conda_envs/command_parser
python -m pip install -r scene_understanding/requirements-data.txt
```

## nuScenes

```bash
source /etc/network_turbo
bash scene_understanding/scripts/data/download_nuscenes.sh
```

脚本从 Motional 的 AWS CloudFront 镜像并行下载完整 `v1.0-trainval` 和地图扩展包，校验压缩包后解压到 `nuscenes_full/raw`。

生成官方 train/val、六相机和 2D 投影清单：

```bash
python scene_understanding/scripts/data/prepare_nuscenes.py \
  --dataroot /root/autodl-tmp/datasets/scene_understanding/nuscenes_full/raw \
  --output /root/autodl-tmp/datasets/scene_understanding/nuscenes_full/organized
```

`organized/manifests/{train,val}_cam_*.jsonl` 直接引用 `raw` 中图像，不复制约 398GB
原始数据。每条记录含场景、位置、相机、时间戳、归一化 2D 框、可见度和统一驾驶类别。
最终清单为每路 28,130 条 train 和 6,019 条 val，共 204,894 条相机记录、
1,416,337 个可见投影框；逐条校验结果为缺失图像 0、越界框 0。

## BDD100K

```bash
source /etc/network_turbo
python scene_understanding/scripts/data/download_bdd100k_mirror.py \
  --endpoint https://hf-mirror.com \
  --output /root/autodl-tmp/datasets/scene_understanding/bdd100k/source_parquet
python scene_understanding/scripts/data/prepare_bdd100k.py \
  --source /root/autodl-tmp/datasets/scene_understanding/bdd100k/source_parquet \
  --output /root/autodl-tmp/datasets/scene_understanding/bdd100k/organized
```

镜像版本包含 80,500 张训练/验证图像、二维检测标注、交通灯状态以及天气、道路场景和时间属性。车道和可行驶区域标注仍以 BDD100K 官方 ETH 归档为准，网络恢复后补齐。

整理后结构：

```text
bdd100k/organized/
├── images/train/           # 70,000 JPG
├── images/val/             # 10,500 JPG
├── manifests/train.jsonl
├── manifests/val.jsonl
└── inventory.json
```

清单字段包括 `image_path`、宽高、天气、场景、时段以及每个目标的类别、像素框、
遮挡、截断和交通灯颜色。完整性由总数 80,500 和 10 个 Parquet 分片双重校验。

## Waymo

Waymo 数据集使用独立的非商用许可。接受许可并完成 Google Cloud 登录后，只下载与本模块有关的 `camera_image`、`camera_box`、`camera_to_lidar_box_association` 和 `stats` 模块，不下载完整 LiDAR 数据。

## 最终目录

```text
/root/autodl-tmp/datasets/scene_understanding/
├── bdd100k/organized/
├── drivelm_nuscenes/
└── nuscenes_full/
    ├── raw/
    └── organized/
```

`archives/`、`source_parquet/` 和失败下载缓存只用于恢复下载；在原始数据和清单完成
总数校验后可删除。模型、数据和运行产物均由仓库根目录 `.gitignore` 排除。
