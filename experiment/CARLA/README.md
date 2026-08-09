# CARLA 闭环仿真

本目录实现基础语音操控和复杂避障场景的构建、决策接入、车辆控制、日志与视频记录。

## 入口

场景三文本到 VLA 在线闭环已完成 6 km 严格测试；复现命令、结果和模型不足见 [SCENE_3_VLA_CLOSED_LOOP_TEST_20260805.md](SCENE_3_VLA_CLOSED_LOOP_TEST_20260805.md)。该测试不使用音频或 ASR，当前 BEV 输入来自实时 CARLA 状态代理，不等同于原始 RGB/LiDAR 端到端感知。

CARLA 控制端只消费稳定的 `ControlDecision 1.0`，因此同时兼容：

## 子目录

| 目录 | 作用 |
|---|---|
| `configs/` | 路线、指令、触发点、交通流和闭环参数 |
| `scenarios/` | 基础、复杂、应急、行人和验证场景 |
| `control/` | 决策接入、路线适配、安全监督、完成判定和 PID |
| `continuous/` | 连续路线、场景事件和交通流管理 |
| `evaluation/` | 相机、HUD、事件、日志、指标和视频 |
| `perception/` | CARLA 车辆状态与传感器数据结构 |
| `tests/` | 场景、协议、决策和控制回归测试 |
| `tools/` | 配置生成、路线检查、摘要与诊断工具 |

## 决策链

## 当前集成版本

- 场景代码由组员 CARLA 分支持续集成；当前 `main` 副本作为第一阶段统一控制和场景接口。
- CARLA 服务端与 Python API：统一使用 `0.9.16`，二者版本必须一致。
- 推荐 Python：`3.12.13`；当前 AutoDL 环境已验证 PyTorch 可识别 RTX 5090 的 `sm_120`。
- 默认 CARLA 路径：`$CARLA_ROOT`，也可通过 `CARLA_ROOT` 覆盖。
- 统一集成环境：Linux（当前验证系统为 Ubuntu 22.04）。

Linux 环境优先直接安装 CARLA 0.9.16 自带的 wheel；`carla_bootstrap.py` 也会从
`$CARLA_ROOT/PythonAPI/carla/dist` 查找 `.whl` 或 `.egg`：

```bash
export CARLA_ROOT=$CARLA_ROOT
python -m pip install "$CARLA_ROOT"/PythonAPI/carla/dist/carla-0.9.16-*.whl
python -c "from importlib.metadata import version; import carla; print(version('carla'))"
```

若只需先安装 Python 客户端，也可以使用官方 PyPI（AutoDL 的默认阿里云源不提供该包）：

```bash
python -m pip install -i https://pypi.org/simple carla==0.9.16
```

### AutoDL 服务端验证状态

当前容器已完成以下安装：

```text
CARLA 服务端：$CARLA_ROOT（约 19 GB）
CARLA Python API：0.9.16 / CPython 3.12
```

该容器最初因用户态 EGL/X Server 运行库不完整，`vulkaninfo` 返回
`ERROR_INCOMPATIBLE_DRIVER`。以下依赖组合已在当前 Ubuntu 22.04 / RTX 5090 容器验证，
安装后 `vulkaninfo --summary` 可识别 NVIDIA 580.105.08 和 RTX 5090：

```bash
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  acl libvulkan1 vulkan-tools mesa-utils libegl1 libgles2 libgbm1 \
  xserver-xorg-core xserver-xorg-video-dummy
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json vulkaninfo --summary
```

CARLA 服务端禁止以 root 身份运行，但 root 可以运行 Python 客户端。AutoDL 默认登录
用户为 root，因此使用专用 `carla` 用户，并只授予其穿过 `/root` 到数据盘的权限：

```bash
export CARLA_ROOT=$CARLA_ROOT
export CARLA_CACHE_DIR=/tmp/carla_cache
id carla >/dev/null 2>&1 || useradd -m -s /bin/bash carla
setfacl -m u:carla:--x /root
install -d -o carla -g carla "$CARLA_CACHE_DIR/runtime" "$CARLA_CACHE_DIR/logs"
chmod 700 "$CARLA_CACHE_DIR/runtime"

runuser -u carla -- env \
  HOME="$CARLA_CACHE_DIR" XDG_RUNTIME_DIR="$CARLA_CACHE_DIR/runtime" \
  bash -lc 'cd $CARLA_ROOT && \
    ./CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Low -carla-rpc-port=2000'
```

`-RenderOffScreen` 不显示窗口，但 RGB 摄像头仍正常渲染并把图像直接传给 Python 代码，
适合远程服务器闭环测试。无需安装桌面环境或通过远程桌面查看画面。

## 场景理解数据采集

`run_control_experiment.py` 已接入 `scene_understanding` 的同帧采集桥。它不会让视觉
模型参与紧急制动或 TTC 控制，只保存场景帧解释、语义对齐和离线评测所需的数据。

先启动 CARLA 0.9.16 服务端，再从仓库根目录执行：

```bash
export PYTHONPATH="$PWD"
cd experiment/CARLA
python run_control_experiment.py emergency_brake \
  --duration-s 25 \
  --scene-capture \
  --scene-capture-every-n 10 \
  --output-dir outputs/runs/emergency_scene_capture
```

同样可将场景名替换为 `straight_driving` 或 `pedestrian_crossing`。采集结果位于：

```text
实时感知 + DrivingIntent
  -> StructuredBEVRasterizer
  -> LightweightVLAPipeline
  -> safety_bridge
  -> ControlDecision
  -> safety_supervisor / PID
```

两条链路使用同一 `ControlDecision` 协议，场景代码无需感知上游决策来源。

## 环境

- Linux
- Python 3.12.13
- CARLA 0.9.16
- PyTorch 2.11.0 + CUDA 13.0
- NVIDIA RTX 5090，SM120

从提交包根目录加载统一路径：

```bash
cd ../..
python -m scene_understanding.core.prepare_carla_samples \
  --capture-index experiment/CARLA/outputs/runs/emergency_scene_capture/scene_understanding/capture_index.jsonl \
  --prompt scene_understanding/prompts/scene_understanding.txt \
  --output experiment/CARLA/outputs/runs/emergency_scene_capture/scene_manifest.jsonl

python -m scene_understanding.core.run_qwen_scene_inference \
  --manifest experiment/CARLA/outputs/runs/emergency_scene_capture/scene_manifest.jsonl \
  --model-path $MODEL_ROOT/Qwen2.5-VL-3B-Instruct \
  --output experiment/CARLA/outputs/runs/emergency_scene_capture/scene_results.jsonl \
  --limit 10 \
  --fail-fast
```

CARLA 服务端应先启动并监听默认 `127.0.0.1:2000`；端口和运行参数以各入口
`--help` 及 `configs/` 内配置为准。

## 随附实测

| 场景 | 配置 | 结果 |
|---|---|---|
| 场景一 | `configs/basic_voice_urban_5km.json` | 约 5.03 km，状态 `SUCCESS`，碰撞 0，非法压线 0 |
| 场景二 | `configs/scene_2_submission_8_runtime.json` | 约 8.00 km，计划 `COMPLETED`，碰撞 0，车道侵入 0 |

场景二抽样闭环记录共 127 帧，帧管线时延 p50/p95/max 为
`29.360/35.188/65.276 ms`。其中感知 p95 为 `29.159 ms`，语义对齐 p95 为
`0.482 ms`，VLA 推理 p95 为 `3.935 ms`，VLA + FSM p95 为 `4.559 ms`。

原始摘要、指令、事件和抽样管线日志位于
`基础赛道提交材料/06_仿真测试全量报告/原始时序数据/`。

## 结果边界

随附场景二日志使用 8 条安全闭环指令配置，复杂障碍事件处于关闭状态。目录仍保留
`configs/scene_2_town05_runtime.json` 的 15 条组合指令与车辆、行人、公交站和骑行者事件设计，
但提交材料不把当前 8 km 日志表述为完整复杂避障验收。

## 回归测试

在提交包根目录执行：

```bash
python -m pytest -q experiment/CARLA/tests
```

也可按根目录 `README.md` 统一运行指令解析、场景理解、VLA 和 CARLA 四部分回归测试。
