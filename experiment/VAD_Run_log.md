# VAD Run Log

当前状态：VAD 最小复现实验已完成。

本文件只记录实际执行过的命令、输出和问题。
未运行的内容不记录为实验结果。

---

# 2026-07-08

## 目标

根据项目规划完成 VAD（Vectorized Autonomous Driving）baseline 调研，并准备复现实验环境。


## 已完成

阅读并整理：

- VAD 论文：
  - VAD: Vectorized Scene Representation for Efficient Autonomous Driving

- 官方仓库：
  - https://github.com/hustvl/VAD


确认 VAD 作为自动驾驶：

```

感知
+
运动预测
+
规划

```

方向的 baseline。


完成基础调研内容：

- VAD 核心思想
- Vectorized Scene Representation
- Motion Prediction
- Planning Head
- nuScenes 数据格式
- 与 CARLA 项目结合方式


## 未执行

未完成：

- VAD完整训练
- nuScenes full 下载
- 自定义数据训练
- CARLA迁移


原因：

当前设备：

- RTX4060 Laptop
- 显存 8GB

无法满足官方完整训练需求。


## 下一步

- 配置 VAD 环境
- 准备 nuScenes mini 数据
- 下载官方 checkpoint
- 测试 inference 流程


---

# 2026-07-09

## 目标

完成 VAD 最小复现实验：

1. 配置运行环境
2. 准备 nuScenes mini 数据
3. 准备 CAN bus 和 map 数据
4. 生成 VAD annotation
5. 加载官方 checkpoint
6. 完成 inference 和 evaluation


---

# 环境

## 主机

- 主机：
  Windows 11 + WSL2 Ubuntu 22.04


## GPU

- GPU：
  NVIDIA GeForce RTX4060 Laptop


- 显存：

```

8GB

```


## CUDA

nvidia-smi:

```

CUDA Version: 13.2

```


实际 PyTorch CUDA:

```

CUDA 11.1

```


## Python

```

Python 3.7

```


## Conda 环境

```

vad

```


## 当前目录

```

/home/neonfox/VAD

```


## 代码版本

官方 VAD:

```

[https://github.com/hustvl/VAD](https://github.com/hustvl/VAD)

```


---

# 环境配置


## PyTorch


安装：

```

torch 1.9.1+cu111

````


验证：

```bash
python -c "
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
"
````

输出：

```text
1.9.1+cu111
11.1
True
```

结果：

CUDA 环境正常。

---

# MMDetection3D 环境

安装版本：

```
mmcv-full 1.4.0

mmdet 2.14.0

mmdet3d 0.17.1
```

验证：

```bash
python -c "import mmcv; print(mmcv.__version__)"
python -c "import mmdet; print(mmdet.__version__)"
python -c "import mmdet3d; print(mmdet3d.__version__)"
```

输出：

```text
mmcv 1.4.0

mmdet 2.14.0

mmdet3d 0.17.1
```

---

# 数据准备

## nuScenes 数据

由于设备限制：

未下载：

```
nuScenes v1.0-trainval full
```

使用：

```
nuScenes v1.0-mini
```

目录：

```
data/nuscenes
```

包含：

```
samples
sweeps
v1.0-mini
maps
```

---

# CAN bus 数据

## 数据来源

nuScenes CAN bus expansion

下载方式：

Windows 手动下载。

复制到 WSL：

```
VAD/data/can_bus
```

验证：

路径可被 converter 正常读取。

---

# Map 数据

首次测试出现：

```text
FileNotFoundError:

data/nuscenes/maps/expansion/boston-seaport.json
```

原因：

缺少：

```
nuScenes-map-expansion
```

处理：

下载：

```
nuScenes-map-expansion-v1.3
```

移动：

```
maps/nuScenes-map-expansion-v1.3/expansion

↓

maps/expansion
```

最终结构：

```
data/nuscenes/maps

├── expansion
│   ├── boston-seaport.json
│   ├── singapore-onenorth.json
│   ├── singapore-hollandvillage.json
│   └── singapore-queenstown.json
```

---

# VAD Annotation生成

## 执行命令

```bash
python tools/data_converter/vad_nuscenes_converter.py \
    nuscenes \
    --root-path ./data/nuscenes \
    --out-dir ./data/nuscenes \
    --extra-tag vad_nuscenes \
    --version v1.0-mini \
    --canbus ./data
```

## 关键输出

```text
Loading NuScenes tables for version v1.0-mini...


total scene num: 10

train scene: 8
val scene: 2


train sample: 323

val sample: 81
```

## 生成文件

```bash
ls -lh data/nuscenes/*.pkl
```

输出：

```text
vad_nuscenes_infos_temporal_train.pkl

9.0M


vad_nuscenes_infos_temporal_val.pkl

2.7M
```

## 结果

成功：

```
VAD annotation生成完成
```

---

# Checkpoint准备

## Backbone

文件：

```
ckpts/resnet50-19c8e357.pth
```

## VAD模型

下载官方：

```
VAD_tiny.pth
```

目录：

```
ckpts/

├── resnet50-19c8e357.pth

└── VAD_tiny.pth
```

---

# Inference测试

## 执行命令

```bash
python tools/test.py \
    projects/configs/VAD/VAD_tiny_stage_2.py \
    ckpts/VAD_tiny.pth \
    --eval bbox
```

---

# 关键输出

模型成功加载：

```text
load checkpoint from local path:
ckpts/VAD_tiny.pth
```

测试数据：

```text
81 samples
```

速度：

```text
6.3 task/s
```

---

# Motion Prediction结果

## Vehicle

```text
ADE_car:
0.9311


FDE_car:
1.0283


MR_car:
0
```

## Pedestrian

```text
ADE_pedestrian:
1.2387


FDE_pedestrian:
1.6274


MR_pedestrian:
0.357
```

---

# Planning结果

输出：

```text
plan_L2_1s:
1.417


plan_L2_2s:
2.350


plan_L2_3s:
3.322
```

碰撞指标：

```text
plan_obj_col_1s:
0


plan_obj_col_2s:
0


plan_obj_col_3s:
0
```

---

# Detection结果

输出：

```text
mAP:
0.0000


NDS:
0.0129
```

说明：

当前实验：

```
nuScenes mini
+
官方 VAD-Tiny checkpoint
```

主要验证：

* 数据读取
* checkpoint加载
* inference流程

不代表论文完整性能。

---

# 遇到的问题与处理

## 问题1：mmcv._ext缺失

错误：

```text
ModuleNotFoundError:
No module named 'mmcv._ext'
```

原因：

安装普通 mmcv：

```
mmcv 1.4.0
```

缺少 CUDA extension。

处理：

重新安装：

```
mmcv-full 1.4.0
```

解决。

---

## 问题2：缺少 similaritymeasures

错误：

```text
ModuleNotFoundError:
No module named similaritymeasures
```

处理：

```bash
pip install similaritymeasures
```

解决。

---

# 当前结果总结

| 项目            | 状态 |
| ------------- | -- |
| VAD环境配置       | ✅  |
| PyTorch CUDA  | ✅  |
| MMDetection3D | ✅  |
| nuScenes mini | ✅  |
| CAN bus       | ✅  |
| Map expansion | ✅  |
| annotation生成  | ✅  |
| checkpoint加载  | ✅  |
| inference     | ✅  |
| evaluation    | ✅  |

---

# 当前结论

已完成 VAD 最小复现实验。

当前设备：

```
RTX4060 Laptop 8GB
```

不进行：

* VAD重新训练
* nuScenes full训练

采用：

```
官方checkpoint

+

nuScenes mini

+

inference evaluation
```

验证模型运行流程。

---

# 下一步

1. 可视化 VAD 输出：

```
预测轨迹
+
地图
+
agent位置
```

2. 分析：

```
vector scene representation

motion prediction

planning trajectory
```

3. 研究迁移：

```
VAD trajectory prediction

↓

CARLA controller

↓

vehicle control
```
