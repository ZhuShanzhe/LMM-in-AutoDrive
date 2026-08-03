# 场景感知双链路架构

## 1. 目标

本架构服务于 `DrivingIntent 1.1.0` 的目标对齐和安全门控。同步路径保证控制循环不会
等待大模型，异步路径保留开放语义能力。任何视觉模型都不得生成或覆盖世界坐标、速度、
距离、TTC、碰撞事件、交通灯控制器状态和最终变道安全结论。

## 2. 数据流

```text
RGB / CARLA Actor / HD Map
       |
       +--> 同步实时路径 ------------------------------------------+
       |    YOLOP: vehicle + lane marking + drivable area          |
       |    域适配 YOLO11s: vehicle/person/cyclist/light/sign/cone  |
       |    ByteTrack: track_id + temporal association              |
       |    Map/Actor API: metric state + lane legality + signal     |
       |                         |                                   |
       |                         v                                   |
       |                 PerceptionFrame + WorldState               |
       |                         |                                   |
       |                         v                                   |
       |             semantic alignment / TTC / safety gate --------+
       |
       +--> 异步低频路径
            Qwen2.5-VL-3B 或 MiniCPM-V 4.6
            latest-frame queue, stale-result rejection
            scene summary / target description / open semantics
```

## 3. 指令目标覆盖

| 目标 | 同步来源 | 异步补充 | 说明 |
|---|---|---|---|
| `VEHICLE/SLOW_VEHICLE` | YOLOP/通用检测器 + ByteTrack + 度量融合 | 外观描述 | 慢车由相对速度判断 |
| `PEDESTRIAN/CYCLIST` | YOLO11/专用检测器 + ByteTrack | 行为描述 | 紧急制动不等待 VLM |
| `OBSTACLE/ROAD_HAZARD` | 专用检测器、深度或数据集标注 | 开放类别描述 | 仅文字识别不能产生 TTC |
| `TRAFFIC_CONE/CONSTRUCTION_ZONE` | 专用检测器、地图 | 区域摘要 | 施工区可由多帧语义补充 |
| `TRAFFIC_LIGHT/TRAFFIC_SIGN` | 检测器 + 信号/地图 API | OCR/细分类 | 灯色优先控制器或标注 |
| `LANE/ROAD` | HD Map + YOLOP 车道/可行驶区域 | 道路语义 | 视觉线不能单独证明合法性 |
| `CROSSWALK/STOP_LINE/JUNCTION` | 地图或像素分割 | 场景解释 | 用于 `YIELD/STOP/TURN` 对齐 |
| `CURB/PARKING_*` | 地图或专用分割 | 停车位描述 | 停车需要后续专门数据 |
| `LANDMARK/AREA` | 地图 | VLM | 不进入紧急安全规则 |
| `DESTINATION/PICKUP/DROPOFF/COORDINATE` | 路由地图 | 无 | 不由相机凭空定位 |

机器可读矩阵位于 `realtime_perception/taxonomy.py`。

## 4. 变道对齐

`CHANGE_LANE/MERGE/OVERTAKE/PULL_OVER` 至少读取：

- 地图：当前车道、左右相邻驾驶车道、允许变道方向、路口状态；
- 视觉：车道线和可行驶区域，仅用于一致性检查与降级定位；
- 跟踪：目标车道前后车辆轨迹；
- 度量：相对位置、相对速度、距离和 TTC；
- 规则：`LEFT/RIGHT_LANE_EXISTS`、`LANE_CHANGE_LEGAL`、
  `LEFT/RIGHT/TARGET_LANE_SAFE`。

当地图不可用或视觉/地图冲突时，`dynamic_safe` 保持 `unknown`，执行
`WAIT_FOR_SAFE` 或 `SAFE_STOP`，不得乐观放行。

## 5. 当前结果

### 实时路径

| 配置 | BDD100K R/P | nuScenes R/P | CARLA 灯 R/P | CARLA P95 |
|---|---:|---:|---:|---:|
| 通用专项 YOLO11s 640 | 61.16/64.96% | 57.36/45.01% | 13.87/47.69% | 49.10 ms |
| 通用专项 YOLO11s 768 | 63.29/65.20% | 58.79/45.23% | 15.21/37.99% | 44.17 ms |
| 域适配 YOLO11s 640 | 62.53/62.03% | 58.64/44.20% | 17.67/56.43% | 41.36 ms |
| 域适配 YOLO11s 768 | 65.10/62.57% | 59.91/44.35% | 19.91/59.33% | 48.14 ms |

CARLA 默认使用域适配 YOLO11s 640。独占 RTX 5090 时可切换 768，但时延余量较小。
双分块会把灯头召回提高到 23.94%，同时使 P95 升到 60.01 ms，因此只保留为离线选项。

CARLA 真值使用 `TrafficLight.get_light_boxes()` 逐灯头投影，旧 actor 大框结果无效。
闭环由 Actor、地图、信号控制器和确定性降级策略兜底。BDD100K 和 nuScenes 各
1,000 帧均为与训练/验证隔离的测试清单。

题目规定的 90% 是场景任务完成率，不是检测 mAP。当前直线行驶、行人横穿、紧急制动
为 3/3 成功，仍需通过批量路线、天气、速度和随机种子扩充后再形成正式达标结论。

### 异步路径保留结果

Qwen2.5-VL-3B 在 41 帧 CARLA 上 41/41 最终 Schema 合法，平均 6.325 s，峰值显存
约 7.16 GiB；车辆召回 8/8、行人 8/10、远距离交通灯 0/76。该结果作为异步 VLM
对照保留，不进入默认实时链路。

MiniCPM-V 4.6 在 CARLA 9 帧上 9/9 Schema 合法，平均 3.636 s、峰值 3.06 GiB；
DriveLM-nuScenes 10 帧上 9/10 合法，有效帧平均 4.604 s、峰值 3.31 GiB。16×与
4×模式都未产生可定位目标框，因此只保留摘要对照，不替代 Qwen 的历史结果，也不
进入安全链路。

## 6. 数据集安排

```text
/root/autodl-tmp/datasets/scene_understanding/
├── carla/                  # 同源闭环、Actor/Map 真值和语义相机
├── drivelm_nuscenes/       # DriveLM 图像与 QA，VLM 语义泛化
│   ├── archives/
│   ├── annotations/
│   ├── images/
│   ├── manifests/
│   └── logs/
├── nuscenes_full/
│   ├── raw/                # 完整传感器、3D 框、轨迹与 HD Map
│   └── organized/          # 官方 split、六相机和 2D 投影清单
├── bdd100k/
│   └── organized/          # 80,500 张图像和检测/属性 JSONL
└── waymo/                  # 只下载所需模块和分片
```

- 当前实际 2TB 数据盘足够保留 nuScenes trainval、BDD100K、CARLA、模型和训练缓存；
- nuScenes 解压目录约 398GB，下载归档约 294GB，确认清单后归档可删除；
- BDD100K 组织后共 80,500 张图；Parquet 中间源确认后可删除；
- 3TB 以上只在确实需要 Waymo 多任务全量时使用，本任务不应默认下载 Motion/Sim Agents。

nuScenes 和 Waymo 是真实道路数据，不是 CARLA 式仿真器。闭环控制仍在 CARLA 0.9.16
验证；真实数据用于检测、跟踪、地图对齐和跨域泛化评测。

## 7. 模型与许可

- YOLOP：MIT，外部代码和权重放在数据盘，不提交仓库；
- YOLO11/Ultralytics：当前代码按 AGPL-3.0 使用，其他部署方式需重新确认许可；
- torchvision SSDLite：作为失败对照保留评测摘要，不作为默认后端；
- Qwen2.5-VL-3B：保留已有实验结果；
- MiniCPM-V 4.6：Apache-2.0，使用隔离环境，只做异步对照；
- 数据集分别遵循 CARLA、nuScenes、BDD100K、Waymo 和 DriveLM 的原始许可。
