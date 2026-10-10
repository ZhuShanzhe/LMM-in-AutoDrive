# 挑战赛道答辩准备

版本：20261010 / V140。代码分支：`challenge-track`。本目录对应本次初审补充核验，历史测试不能替代本版结论。

## 材料入口

- [性能实测报告 PDF](PERFORMANCE_REPORT.pdf)
- [性能实测报告 Markdown](PERFORMANCE_REPORT.md)
- [场景二逐指令补充工程复核](scene2_full41_manual_review.json)
- [三场景路线结果](closed_route_results.json)
- [场景二运行源码指纹](source_files_sha256.json)
- [最终代码回归记录](v140_production_regression.log)
- [视频剪辑范围与校验](scene2_video_record.json)

权重、完整原始证据和视频见[初审材料共享目录](https://pan.baidu.com/s/1yYR908tHC06pbEB4hImVfA?pwd=1009)，提取码 `1009`。使用文件名 `面向智能驾驶的大模型应用场景-南京大学_初审材料.zip`；本索引不代表上传完成，云端文件以最终校验记录为准。

## 代码链路

```text
ASR 文本 -> 指令规范化与组合解析 -> DrivingIntent / ModernBERT
四视角 RGB / LiDAR / 雷达 / 自车状态 -> 观测与持续目标追踪
观测 + 指令 + 递归事件记忆 -> 多模态决策与短期纵向序列
指令图 + 行为状态机 + 风险约束 -> 路线 PID -> CARLA
```

- 指令模块：`structured_command_parser/src/`。
- 感知与执行接口：`scene_understanding/`。
- 决策、记忆、速度和风险契约：`lightweight_vla_adapter/src/`。
- 主控制器：`experiment/CARLA/universal_vla_controller.py`。
- 场景二地图及事件：`experiment/CARLA/scenarios/complex/town05_scene2.py`。
- 场景二配置：`experiment/CARLA/configs/scene_2_town05_runtime.json`。
- 评测与安全记录：`experiment/CARLA/benchmark/`；运行入口见 `docs/CHALLENGE_X86_TESTING.md`。

V140 保留约 8 km 路线、前 2260 m 事件几何、必需城市路口、公交站和五个事件；后段连续巡航。初始化取消三辆非必需随机背景公交，但保留 70 辆背景机动车、24 名行人和独立任务公交。碰撞、红灯、明确停车与运行时错误保护保留。

## 结果口径

| 场景 | 路线完成 | 原始登记结果 | 文本至控制 P95 |
| --- | --- | --- | --- |
| 一 | 4997.06 m | 15/15 | 148.857 ms |
| 二 | 8001.023 m | 11/15；另有四条补充工程复核通过 | 140.828 ms |
| 三 | 5999.96 m | 7/7 | 103.940 ms |

场景二原始结果为 `CHECK_FAILED`、退出码 2，三条 `TIMEOUT` 和一条 `SCENE_INVALID` 不改写。02 放宽交通受限条件下的速度保持时长；03 在严格窗口结束后、下一指令前合法完成礼让、变道和超越；11 等待真实交通解除后达到目标附近速度；04 经实际动作核对，右侧车道和目标速度连续保持 18.30 秒。合并工程口径为 15/15，不是独立金标准语义准确率或官方严格验收。

场景二同一次运行共有 48857 个连续帧，2442.85 仿真秒、7484.715 墙钟秒；安全封存记录为零碰撞、零实线/限制车道违规、零超速帧、零红灯违规。4387.422 m 阻挡自然解除，没有人工修车或重跑。

## 视频与性能边界

原始连续视频完整保留。展示版保留原视频 `[0,1568.9]` 与 `[1673.8,2443.1]` 秒，删除 104.9 秒，将停滞保留约 10.15 秒；展示版不能称未剪辑全程。测试时长和时延仍用完整原始记录。

FLOPs 压缩比 0.495118；同机组件 CPU P95 86.509 -> 51.856 ms；混合量化动作代理指标相对衰减 2.22%。这些是各自限定范围的实测，不等于整链路全部指标。编译器模型时延 9.067 ms 是估算，ASR 端到端、J6P 板端、官方异构利用率与功耗未取得对应实测结果。

答辩引用数字时同时说明测试范围、原始与补充判据以及实测/估算区别，具体证据以报告和 JSON 为准。
