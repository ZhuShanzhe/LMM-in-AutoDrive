# 提交模型目录

模型权重与 Docker 镜像单独提交，不写入 Git。运行脚本从仓库根目录的相对路径 `models/` 查找权重，也允许用 `MODEL_ROOT` 指向容器只读挂载目录。

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

提交前在 `models/` 内生成并校验完整哈希：

```bash
find models -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > models/SHA256SUMS
sha256sum -c models/SHA256SUMS
```

不得在配置、脚本或清单中保留训练服务器绝对路径。运行日志可以记录本次容器解析后的绝对路径，便于审计。
