# Experiments

本目录用于存放 baseline 复现实验记录。

## 目录约定

```text
experiments/
|-- drivelm/
|   |-- README.md
|   |-- run_log.md
|   |-- external/        # 官方仓库或第三方代码，按需创建
|   |-- data/            # demo data 或小规模实验数据，按需创建
|   |-- outputs/         # 推理、评估、日志输出，按需创建
```

## 记录原则

- 复现实验相关结论必须基于实际运行结果。
- 每次运行命令、环境、错误和输出都记录到对应 baseline 的 `run_log.md`。
- 官方代码建议放到 baseline 子目录下的 `external/`，避免和本项目主代码混在一起。
- 大数据集、模型权重和临时输出不要直接提交到 Git。
