# 语言与决策模块 CPU 检查

本目录仅用于朱善哲负责的语言解析和轻量 VLA 的离线完整性、接口和输出一致性检查，不包含 ASR、感知、CARLA 服务或 J6P 工具链，也不是完整赛事提交镜像。

## 环境和权重

使用 Linux x86_64、Python 3.12.13。独立 CPU 环境安装：

```bash
python -m pip install -r lightweight_vla_adapter/deployment/requirements-cpu.txt
```

权重下载入口和固定版本见 [模型说明](../../models/README.md)。下载完成后可以关闭网络：

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python lightweight_vla_adapter/scripts/check_owned_cpu_runtime.py \
  --model-root /root/autodl-tmp/models/challenge-assets-pinned \
  --output /root/autodl-tmp/review_logs/owned_cpu_check
```

输出目录必须不存在，避免覆盖旧记录。输出包含 `report.json` 和三个地图的绝对路径配置。输入是冻结清单对应的本地模型目录；输出报告记录校验、实际权重加载、英文指令 JSON、40 km/h 目标速度、非法输入拒绝及卷积归一化融合前后对比。

## 容器入口

```bash
docker build -f lightweight_vla_adapter/deployment/Dockerfile.cpu -t autodrive-owned-cpu:0925 .
mkdir -p cpu-results
docker run --rm --network none \
  -v /root/autodl-tmp/models/challenge-assets-pinned:/models:ro \
  -v "$PWD/cpu-results":/results autodrive-owned-cpu:0925
```

镜像不内置权重或训练数据；模型只读挂载。重复执行需指定新的输出目录，例如追加 `--model-root /models --output /results/check2`。

当前无卡服务器未安装 Docker，未执行镜像构建或容器运行。已提供 Dockerfile 和依赖定义，不能据此宣称容器验收通过。CPU 推理记录也不代表驾驶准确率、端到端延时或 J6P 性能。默认权重和运行配置保持不变。
