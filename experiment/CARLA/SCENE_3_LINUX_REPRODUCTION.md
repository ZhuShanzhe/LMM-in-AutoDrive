# 场景三 Linux / Docker 复现

场景代码不绑定服务器绝对路径。CARLA 0.9.16、Python 环境和 ffmpeg 由提交的 Docker 镜像提供；模型权重通过挂载目录或环境变量交给模型进程，不写入场景配置。

在仓库根目录运行严格基线：

```bash
PYTHON_BIN=python3 \
CARLA_ROOT=/opt/carla \
bash experiment/CARLA/tools/run_scene3_linux.sh \
  --duration 1400 \
  --fixed-delta-seconds 0.05 \
  --record-ground-truth \
  --ground-truth-every-n 20 \
  --require-complete-scene
```

`CARLA_ROOT` 可省略；启动器会按仓库相对位置和常见同级目录发现 CARLA。输出默认写到 `experiment/CARLA/outputs/scene3_linux_run/`，也可设置 `SCENE3_OUTPUT_DIR`。

后续替换多模态模型时，场景、传感器、真值和评测接口保持不变：

```bash
export MODEL_ROOT="${MODEL_ROOT:-$(pwd)/models}"
EGO_CONTROLLER=external bash experiment/CARLA/tools/run_scene3_linux.sh --duration 0
```

`external` 模式不会由场景脚本施加车辆控制，模型/ROS/CARLA 客户端接管 `role_name=hero` 的 ego。默认 `route-pid` 用于验证场景自身；`behavior-agent` 仅用于 CARLA 基线对照。

已有 7 个事件保持固定，不会在运行时随机新增变体。普通背景流只在切入到施工区窗口内活动；阻塞车道事件使用独立的前后间隙车辆，避免不同事件互相污染。
