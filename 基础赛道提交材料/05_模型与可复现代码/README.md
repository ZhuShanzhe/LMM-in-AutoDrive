# 模型与可复现代码

## 1. 源码位置

整个分支即源码包，不在本目录复制第二份代码：

| 模块 | 目录 | 内容 |
|---|---|---|
| ASR/NLU | `automatic_speech_recognition/` | 录音、降噪、ASR、翻译、测试 |
| 指令解析 | `structured_command_parser/` | 数据构建、训练、校准、推理、Schema 和测试 |
| 场景理解 | `scene_understanding/` | 数据准备、训练、实时推理、语义对齐、风险和测试 |
| 轻量 VLA | `lightweight_vla_adapter/` | 数据构建、训练、蒸馏、推理、评测、安全门和测试 |
| CARLA | `experiment/CARLA/` | 场景、传感器、协议、控制、日志、指标和视频 |

## 2. 环境

统一基线：

```text
Ubuntu 22.04
Python 3.12.13
CARLA 0.9.16
CUDA 13.0
RTX 5090 / sm_120
```

依赖按模块安装：

```bash
python -m pip install -r automatic_speech_recognition/requirements.txt
python -m pip install -r structured_command_parser/requirements-modernbert.txt
python -m pip install -r scene_understanding/requirements-realtime.txt
python -m pip install -r lightweight_vla_adapter/requirements.txt
```

部分模块依赖冲突，正式复现建议建立独立环境。详细命令和模型下载方式见各模块 README。

## 3. 最小验证

```bash
python -m pytest -q \
  lightweight_vla_adapter/tests \
  structured_command_parser/tests \
  scene_understanding/tests

(
  cd experiment/CARLA
  python -m unittest discover -s tests -v
)
```

2026-07-30 在提交包工作树复核结果：

- 指令解析、场景理解与轻量 VLA：`342 passed`，另有 `137 subtests passed`；
- CARLA 协议、指标、安全回归和数据采集：`21 tests passed`；
- 提交包固定权重路径检查：ASR、翻译、DeepFilterNet3、ModernBERT、YOLO11 和 VLA 路径均存在。

CARLA 服务端启动和闭环命令见 `experiment/CARLA/README.md`。

## 4. 权重

权重不进入普通 Git，但服务器生成的最终邮件 ZIP 在根目录 `models/` 中直接附带
预训练权重和最终训练权重。模型 ID、本地目录和校验信息见
[模型权重与参数统计.md](模型权重与参数统计.md)。

```bash
source submission_env.sh
test -f "$QWEN3_ASR_MODEL_PATH/config.json"
test -f "$MODERNBERT_MODEL_PATH/model.safetensors"
test -f "$YOLOP_ROOT/weights/End-to-end.pth"
test -f "$YOLO11_CARLA_MODEL_PATH"
test -f "$VLA_MODEL_PATH"
sha256sum -c models/SHA256SUMS
```

源码默认根据文件位置直接读取根目录 `models/`，不依赖 AutoDL 的绝对路径。
`submission_env.sh` 仅用于命令行查看和复现时统一引用这些固定路径。

## 5. 可复现记录

每次正式复现应保存：

- Git commit；
- Python、PyTorch、CUDA、驱动和 CARLA 版本；
- `pip freeze` 或 Conda 环境文件；
- GPU/CPU/内存信息；
- 权重 SHA256；
- 数据清单 SHA256；
- 运行命令、配置和随机种子；
- 原始预测、时序日志、异常和汇总指标。

当前仓库没有统一锁定文件和一键全链路启动脚本，提交前仍需执行一次干净环境复现。
