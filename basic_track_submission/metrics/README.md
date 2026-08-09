# metrics.zip 内容说明

最终 `metrics.zip` 由服务器真实运行输出生成，不手工伪造指标。计划结构：

```text
metrics/
├── TEST_REPORT.md
├── environment.json
├── offline_model_metrics.json
├── scene1/
│   ├── summary.json
│   ├── commands.jsonl
│   ├── events.jsonl
│   ├── runtime_sampled.jsonl
│   └── vla_decision_statistics.json
├── scene2/
│   └── ...
└── scene3/
    └── ...
```

原始视频、逐帧图片、训练数据和中间 checkpoint 不进入 `metrics.zip`。高频日志可按固定步长采样，但碰撞、事件、命令和最终摘要必须完整保留。
