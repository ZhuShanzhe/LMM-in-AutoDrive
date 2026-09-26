# 感知适配器 Conv-BN 融合：本地 GPU 优化对比

## 测试范围

- 设备：NVIDIA GeForce RTX 3050 Laptop GPU
- 精度：FP32
- 权重：`phase_recovery_v1.pt`
- 输入：固定合成输入
- 原始相机输入尺寸：224 × 224
- CUDA 计时：每次计时推理前后均执行同步
- TF32：关闭
- 覆盖范围：当前配置下的感知适配器前向推理

以下内容不在本次验证范围内：J6P 性能、端到端时延、CARLA 闭环效果、
ASR/ModernBERT、功耗、内存占用和异构算力利用率。

## 数值一致性校验

| 指标 | 结果 |
| --- | ---: |
| 融合的 Conv-BN 对数 | 34 |
| 最大绝对输出差异 | 1.1444091796875e-05 |
| 相对容差 | 0.003 |
| 绝对容差 | 0.001 |

融合前后输出已通过数值一致性校验。

## 本地 GPU 性能对比

| 指标 | 未融合 | Conv-BN 融合后 | 变化 |
| --- | ---: | ---: | ---: |
| 参数量 | 5,251,633 | 5,245,577 | -6,056（约 -0.12%） |
| 平均延迟 | 15.8715 ms | 13.1305 ms | 约 -17.27% |
| P50 延迟 | 15.5156 ms | 13.1293 ms | 约 -15.38% |
| P95 延迟 | 18.5267 ms | 15.5456 ms | 约 -16.09% |
| 最大延迟 | 28.6170 ms | 25.3302 ms | 改善 |

融合前后本地 GPU 的 P95 延迟均低于当前配置的 70 ms 适配器预算。

## CPU 补充说明

同一脚本在本地 CPU 上可通过数值一致性校验，但观测到的 P95 高于 70 ms 预算。
因此，CPU 数据仅作为本机诊断记录，不作为部署性能结论，也不能外推为 J6P 性能。

## 复现条件

1. 获取 `challenge-track` 分支代码。
2. 按 [模型下载说明](../models/README.md) 获取固定版本权重；公开下载入口为 [twlk666/lmm-autodrive-challenge-assets](https://huggingface.co/twlk666/lmm-autodrive-challenge-assets)，已验证可匿名下载。
3. 将权重按以下目录结构保存：

```text
<MODEL_ROOT>/
└─ challenge/
   └─ phase_recovery_v1.pt
```

4. 使用已安装 PyTorch 和 CUDA 的 Python 或 Conda 环境。

## Windows CMD 复现命令

```cmd
set "MODEL_ROOT=D:\models\lmm-autodrive-challenge-assets"
conda run -n pytorch python lightweight_vla_adapter\scripts\benchmark_adapter_convbn_compare.py --checkpoint "%MODEL_ROOT%\challenge\phase_recovery_v1.pt" --device cuda --warmup 30 --runs 200
```

如无可用 CUDA，可将 `--device cuda` 改为 `--device cpu`；CPU 结果只能用于本地诊断，不能与 GPU 或 J6P 性能直接比较。

## 合并核验说明

上述 GPU 数字为黄皓星提交的本地测试记录。当前提交包含复现脚本与汇总报告，未包含逐次 GPU 延时原始记录；本次合并核验不将它视作服务器独立复测结果。按表中 P95 数值计算，降幅约为 16.09%，统计口径限于适配器前向。

融合只用于对比模型副本，本提交不更换默认模型权重或自动开启公共链路融合。三场景同源 1000 帧数据的关键目标、灯色、追踪连续性及语义对齐指标不属于本报告已验证范围。

2026-09-26 服务器无卡复核使用冻结的 `phase_recovery_v1.pt`，CPU、2 线程、5 次预热、20 次计时：融合 34 对，最大绝对输出差为 `3.814697265625e-06`，数值一致性通过。CPU P95 为 292.3226 / 289.7327 ms（未融合 / 融合），仅为该主机诊断，不与上述 GPU 成绩混用。原始记录为服务器 `review_logs/20260926/hhx_convbn_cpu.json`（数据盘根目录下）。

首次合并回归中，已有解析模块时延断言测得 52.803 ms，超过 50 ms 阈值：1367 项通过、1 项失败、207 项子测试通过。此次提交未修改解析路径，保留 `hhx_merge_tests.log`，不修改阈值或删除失败记录。

保持代码和阈值不变再次运行全量回归：1368 项测试、207 项子测试通过，57.80 秒，记录为同目录的 `hhx_merge_tests_repeat.log`。重复通过不代表已经完成时延稳定性验收。
