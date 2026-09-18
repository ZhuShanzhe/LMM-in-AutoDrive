# 自动驾驶语音识别与指令处理（automatic_speech_recognition）

+ 车载语音指令处理链路：**语音识别 → 推理期优化 → 中英翻译 → 指令解析对接**，并提供轻量化、J6P 部署适配与统一评测。

## 1. 目录结构

```text
automatic_speech_recognition/
├── src/                      # 核心源码
│   ├── pipeline.py            # ASRPipeline：对外统一入口
│   ├── asr/                   # 语音识别（服务 + 优化 + 轻量化 + 部署）
│   ├── translator/            # 中英翻译（Qwen3-1.7B）
│   └── tts/                   # 语音合成（Qwen3-TTS）
├── training/                 # 微调：数据准备 / LoRA / 适配器 / 评估
├── tests/                    # 可运行测试 + 评测工具（utils/）
├── configs/                  # 全部配置（asr / training / translation / tts）
├── resources/dialect/        # 方言词典资源（JSON）
├── examples/                 # 最小可运行示例
├── recoder/                  # 麦克风录音（独立组件）
├── scripts/                  # 端到端任务链脚本
├── models/                   # 模型权重（不入 Git）
└── requirements.txt
```

## 2. 安装

```shell
pip install -r requirements.txt
# GPU 版 PyTorch（可选）
pip3 install torch torchvision torchaudio --index-url https://mirrors.nju.edu.cn/pytorch/whl/cu118
```

+ 注意：若环境中已有 `torch`，请确保版本与 `torchvision`、`torchaudio` 一致：
```shell
pip3 show torch torchvision torchaudio       # 确保一致
```

+ 注意：`onnxruntime` 版本需与 `torch` 版本一致。

## 3. 模型准备

| 模型 | 用途 | 默认路径 |
|:---|:---|:---|
| Qwen3-ASR-1.7B | 语音识别 | `models/Qwen3-ASR-1.7B` |
| Qwen3-1.7B | 中英翻译 | `models/Qwen3-1.7B` |
| Qwen3-TTS-12Hz-1.7B | 语音合成（造数据） | `models/Qwen3-TTS-12Hz-1.7B` |

## 4. 快速开始

```python
from automatic_speech_recognition import ASRPipeline

pipe = ASRPipeline(
    asr_model_path="models/Qwen3-ASR-1.7B",
    enable_optimization=True,          # 降噪 + 方言归一
    optimization_config="configs/asr/optimization.yaml",
    enable_translation=True,           # 中 -> 英
    output_language="english",         # chinese | english | both
    output_dir="outputs",
)
result = pipe.process("audio.wav", output_json="outputs/result.json")
print(result["text"], "|", result["translation"], "|", result["output_text"])
```

## 5. 完整流程

```shell
# 0) 构建数据集（可选，若已有 Chinese-commands.json 与语音集可跳过）
# Linux/MacOS: export DASHSCOPE_API_KEY=sk-ws-...      Windows: $env:DASHSCOPE_API_KEY="sk-ws-..."
python training/dataset/translation.py   --config configs/translation/qwen_mt.yaml
python training/dataset/build_dataset.py --kind standard --config configs/tts/standard.yaml
python training/dataset/build_dataset.py --kind dialect  --config configs/tts/dialect.yaml
python training/dataset/build_dataset.py --kind noise    --config configs/tts/noise.yaml

# 1) 基线评测（标准 / 噪声 / 方言）
python tests/asr_test.py     --dataset data/wav_files/standard/mapping.json       # --limit 20
python tests/noise_test.py   --dataset data/wav_files/noise/standard_noise/mapping.json # --limit 20
python tests/dialect_test.py --dataset data/wav_files/dialect/mapping.json # --dialect sichuan --limit 20

# 2) 轻量化（ONNX 导出 + INT8 量化）
python -m src.asr.compression.quantize --config configs/asr/compression.yaml

# 3) 微调（可选：LoRA / 结构适配器）
python -m training.dataset.augmentation --config configs/training/augmentation.yaml
python -m training.train    --config configs/training/finetune.yaml
python -m training.evaluate --config configs/training/finetune.yaml --checkpoint outputs/asr_finetune

# 4) 一键串起评测与轻量化
bash scripts/run_asr_optimization.sh
```

## 6. 地平线算法工具链

### 6.1 环境准备
+ 开发环境：

| 硬件/操作系统	| 要求 |
|:---|:---|
| CPU | CPU I3以上或者同级别E3/E5的处理器 |
| 内存 | 16G或以上级别 |
| GPU | CUDA 12.8、驱动版本 Linux: >= 550.163.01 |
| 系统 | 原生Ubuntu 22.04 |

+ Docker 容器准备：
    + Docker（20.10.10或更高版本，建议安装20.10.10版本）：
    ```shell
    sudo apt update
    sudo apt-get install -y docker.io
    ```
    + NVIDIA Container Toolkit（1.16.2或更高版本，建议安装1.17.8）
    + 将无 root 权限的用户添加到 Docker 用户组中：
    ```shell
    sudo groupadd docker
    sudo gpasswd -a ${USER} docker
    sudo service docker restart
    ```

+ PTQ 量化环境依赖：
    | 依赖项 | 版本 / 说明 |
    |:---|:---|
    | 操作系统 | Ubuntu 22.04 |
    | Python | 3.10 |
    | libpython3.10 | - |
    | python3-devel | - |
    | python3-pip | - |
    | gcc & g++ | 12.2.1 |
    | graphviz | - |

+ QAT 量化环境依赖：
    | 硬件/操作系统 | GPU | CPU |
    |:---|:---|:---|
    | OS | Ubuntu 22.04 | Ubuntu 22.04 |
    | CUDA | 12.8 | N/A |
    | Python | 3.10 | 3.10 |
    | torch | 2.8.0+cu128 | 2.8.0+cpu |
    | torchvision | 0.23.0+cu128 | 0.23.0+cpu |
    | 推荐显卡 | Titan V / 2080Ti / V100 / 3090 | N/A |

    + 完成 QAT 模型训练后，可在当前训练环境安装相关工具包，并直接通过接口调用的方式完成后续的模型转换工作。

### 6.2 环境部署
+ **环境部署方式选择（本项自推荐 Docker 路线）**：

    | 方式 | 适用场景 | 说明 |
    |:---|:---|:---|
    | Docker 容器 | 本项目采用 | 镜像已预置 CUDA / Python 3.10 / HBIR / HBDK，开箱即用 |
    | 本地手动安装 | 无法使用容器时 | 需自行处理 gcc 软链接、GLIBC 冲突等依赖问题 |

+ Docker 容器部署：
    1. 提前创建数据集目录，否则加载失败：
    ```shell
    mkdir -p data/
    ```
    2. 在 OE 包一级目录启动容器（本地无镜像时会自动从官方 Docker Hub 拉取）：
    ```shell
    bash run_docker.sh data/
    # 仅需 CPU 时追加 cpu 参数
    bash run_docker.sh data/ cpu
    ```
    3. 若使用离线镜像，需先加载：
    ```shell
    docker load -i docker_openexplorer_xxx.tar.gz
    ```
    4. 手动启动（`{version}` 为 OE 版本号，如 `3.8.1`；GPU 镜像为 `..._j6`，CPU 镜像为 `..._j6_cpu`）：
    ```shell
    # GPU Docker
    docker pull openexplorer/ai_toolchain_ubuntu_22_j6_gpu:{version}
    docker run -it --rm \ 
        --network host \ # 调整网络模式为host
        --gpus all \ # 在启动容器时，添加标记以启用GPU资源的访问
        --shm-size=15g \ # 修改共享内存大小
        -v {OE 包路径}:/open_explorer \ # 挂载 OE 包
        -v {数据集路径}:/data/horizon_j6/data \ # 挂载数据集
        openexplorer/ai_toolchain_ubuntu_22_j6_gpu:{version}
    ```
    + 注意事项：
        + 必须用推荐方式（`bash run_docker.sh` 或 `docker attach`）进入容器。直接 `docker exec` 可能因镜像构建时设置的 `PATH` / `LD_LIBRARY_PATH` 未加载，导致 CMake / GCC / CUDA 使用异常。
        + 去掉 `--rm` 可避免容器退出后被销毁；追加 `-d` 可后台常驻，再用 `docker exec -it {容器ID} /bin/bash` 进入。

+ 本地手动安装（不走 Docker 时）：
    ```shell
    cd package/host
    bash install.sh
    ```
    + 脚本会自动检查环境，缺依赖会中断并提示，补齐后重跑即可。
    + 安装成功后会在 `~/.bashrc` 追加 PATH 等变量，需执行 `source ~/.bashrc` 生效；建议顺带确认 `LD_LIBRARY_PATH` 是否符合预期。
    + 交叉编译工具链（仅当需要生成板端/X86/QNX 可执行程序时才用）：

        | 目标 | 工具链 |
        |:---|:---|
        | 板端（Linux）| `aarch64-none-linux-gnu-gcc` / `g++`，Arm GNU Toolchain 12.2.Rel1 |
        | X86 仿真 | X86 `gcc`；若提示版本不符，需重新建立 `gcc-12.2.0` / `g++-12.2.0` 软链接 |
        | QNX | `aarch64-unknown-nto-qnx8.0.0-gcc` / `g++`；因 LICENSE 限制不随 OE 包交付，需联系地平线技术支持 |

    + 编译报错处理：若出现 `xxx@GLIBC_xxx` 未定义符号，用 `-rpath-link` 指向 `aarch64-none-linux-gnu/lib`，并显式添加 `-lpthread` 等库；源文件变量 `SRCS` 需放在 `${LIBS}` 之前。

+ 运行环境部署（板端）：

    + 补充工具（部分工具不在系统镜像中，需从宿主机下发）：
    ```shell
    cd package/board
    # Linux 环境
    bash install_linux.sh ${board_ip}
    # QNX 环境
    bash install_qnx.sh ${board_ip}
    ```
    安装后重启开发板，执行 `hrt_model_exec --help` 验证是否成功。

    + DEB 部署包（UCP）：安装后可直接通过命令行调用相关工具，自动安装所需二进制文件与依赖库。详见 UCP 章节“总览-DEB 部署包”。
        + 注意：J6B 平台搭载 QNX 系统，**不支持 DEB 打包工具**；本项目目标平台 J6P 为 Linux，不受此限制影响。

+ 检查 onnx 模型是否符合要求：
```shell
conda create -n oe python=3.10 -y && conda activate oe
pip install hmct-2.8.4-cp310-cp310-linux_x86_64.whl
hb_compile --model outputs/compression/onnx/asr_encoder.onnx --march nash-p
```

## 7. 模块文档

| 模块 | 文档 |
|:---|:---|
| 统一入口 ASRPipeline | [src/README.md](src/README.md) |
| 语音识别 | [src/asr/README.md](src/asr/README.md) |
| 推理期优化 | [src/asr/optimization/README.md](src/asr/optimization/README.md) |
| 中英翻译 | [src/translator/README.md](src/translator/README.md) |
| 语音合成 | [src/tts/README.md](src/tts/README.md) |
| 微调 | [training/README.md](training/README.md) |
| 测试与评测 | [tests/README.md](tests/README.md) |
| 示例 | [examples/README.md](examples/README.md) |

## 8. 说明

+ 权重、数据集与生成语料不入 Git；`data/` 下音频需自行生成或替换。
+ 日志由 `src/utils.py` 统一提供（`setup_logging` / `log_and_print`），各任务不再自带副本；每次运行会**清空并重写**自己的日志文件，默认落在 `logs/` 下（如 `logs/tests/asr_test.log`、`logs/compression.log`、`logs/tts/build_standard.log`），可用 `--log-file ""` 关闭落盘。
+ 推理期优化免训练；模型级微调在 `training/`；评测代码统一在 `tests/`。
+ x86 仿真结果不等同于 J6P 板端性能；功耗与利用率仅在板端测量有效。

## 9. 参考资料

+ https://github.com/QwenLM/Qwen3-ASR
+ https://github.com/QwenLM/Qwen3-TTS
+ https://doc.oe.horizon.auto/3.8.1/guide/env_install/pre-installation_preparation.html