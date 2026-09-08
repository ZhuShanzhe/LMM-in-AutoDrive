# 模型运行目录

模型权重不写入 Git。运行脚本从相对路径 `models/` 查找权重，也允许用 `MODEL_ROOT` 指向服务器共享权重或容器只读挂载目录。下载入口与模型版本见各模块README。

```text
models/
├── modernbert-drive-command-compositional/
├── lightweight_vla_adapter/
│   └── universal_three_scene_v6_sensor_policy/
│       └── model.pt
└── scene_understanding/
    └── yolo11s_specialized_carla_v1/weights/best.pt   # 可选感知审核模块
```

权重和配置的固定校验值见 `lightweight_vla_adapter/UNIVERSAL_THREE_SCENE_MODEL.md`。三个场景使用同一份 VLA 权重；场景差异仅体现在题目规定的可用传感器和场景配置，不能切换成三个专用模型。

Linux 运行示例：

```bash
source submission_env.sh
bash experiment/CARLA/scripts/run_universal_vla.sh scene1
bash experiment/CARLA/scripts/run_universal_vla.sh scene2
bash experiment/CARLA/scripts/run_universal_vla.sh scene3
```

自定义 Docker 权重挂载目录：

```bash
MODEL_ROOT=/models bash experiment/CARLA/scripts/run_universal_vla.sh scene3 /outputs
```

需要记录本次权重版本时，可在 `models/` 内生成并校验哈希：

```bash
find models -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > models/SHA256SUMS
sha256sum -c models/SHA256SUMS
```

共享配置和脚本使用相对路径或环境变量；运行日志可记录本机解析后的绝对路径，便于定位问题。
