# SparseDrive Reproduction

本目录保存刘旭负责的 SparseDrive baseline 复现实验精简材料。

## 实验状态

- 已完成 SparseDrive Stage2 官方权重加载。
- 已完成 nuScenes v1.0-mini 数据转换。
- 已完成 81 个验证样本的全量评测。
- 已完成 FPS、延时、显存与 GPU 遥测记录。
- 已完成 81 帧可视化及 3 个成功、3 个失败案例筛选。

## 文件说明

| 文件或目录 | 内容 |
| --- | --- |
| `metrics.csv` | 汇总指标 |
| `selected_cases.csv` | 典型案例输入、轨迹输出和原因 |
| `../../docs/baseline_research/assets/cases/` | 6 张典型案例可视化 |
| `environment.txt` | 硬件和软件环境 |
| `logs/evaluation.log` | 完整评测输出 |
| `logs/benchmark.log` | FPS 和显存测试输出 |
| `scripts/finalize_sparsedrive_experiment.py` | 指标、案例和 TensorBoard 汇总脚本 |
| `configs/sparsedrive_small_stage2_mini.py` | 本次 mini 评测配置 |

完整原始结果包含 `results.pkl`、81 张逐帧图、视频和 TensorBoard 事件文件，保存在本地归档：

```text
D:\LLM-AutoDrive\sparsedrive_final_20260709
```

这些大文件不提交到 Git，以避免仓库体积持续增长。

## 核心结果

| 指标 | 结果 |
| --- | ---: |
| 3D 感知 mAP / NDS | 0.4249 / 0.4811 |
| 跟踪 AMOTA / AMOTP | 0.5649 / 0.9584 |
| 在线地图 mAP | 0.7741 |
| 规划 L2 | 0.7612 m |
| 官方碰撞率 | 0.242% |
| FPS | 5.9 |
| 等效单帧延时 | 169.49 ms |
| PyTorch 峰值显存 | 3942 MiB |

完整分析见：

```text
docs/baseline_research/SparseDrive.md
```
