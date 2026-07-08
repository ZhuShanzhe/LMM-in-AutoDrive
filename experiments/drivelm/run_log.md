# DriveLM Run Log

当前状态：复现实验尚未开始。

本文件只记录实际执行过的命令、输出和问题。未运行的内容不要写成实验结果。

## 记录模板

~~~markdown
## YYYY-MM-DD HH:MM

### 目标

### 环境

- 主机：
- GPU：
- CUDA：
- Python：
- conda 环境：
- 当前目录：
- 代码版本：

### 执行命令

```bash

```

### 关键输出

```text

```

### 结果

- 是否成功：
- 生成文件：
- 耗时：
- 峰值显存：

### 问题与处理

### 下一步
~~~

## 2026-07-08

### 目标

根据官方文档创建 DriveLM / DriveLM-CARLA baseline 调研与复现实验记录框架。

### 已完成

- 阅读项目规划文档 `program/plan.pdf`。
- 阅读任务分配文档 `program/task_0607.pdf`。
- 确认朱善哲负责 DriveLM / DriveLM-CARLA baseline 调研。
- 根据官方 DriveLM 文档补充基础调研内容。

### 未执行

- 未克隆官方 DriveLM 仓库到实验目录。
- 未创建 conda 环境。
- 未下载数据或模型权重。
- 未运行数据转换、推理或评估脚本。

### 下一步

1. 检查 AutoDL 环境和 GPU 信息。
2. 克隆官方 DriveLM 仓库。
3. 优先尝试官方 demo data 流程。
4. 将实际命令、错误和结果继续写入本文件。
