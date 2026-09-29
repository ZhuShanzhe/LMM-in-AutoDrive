# 环境与复现

开发接口回归可在仓库根目录运行：

```bash
python -m pytest experiment/CARLA/tests lightweight_vla_adapter/tests scene_understanding/tests structured_command_parser/tests -q
```

该命令不是赛事全链路评测，也不启动 CARLA。真实仿真及权重配置按 [统一入口](../../../docs/CHALLENGE_X86_TESTING.md) 执行。

正式复现还须冻结 Git 提交、模型SHA256、Docker镜像digest、SDK/编译器/Python/框架版本、硬件信息、线程数、输入尺寸、随机种子、计时边界及逐条命令。运行环境清单与原始测试日志应同批保存。服务器绝对路径仅用于开发，不得作为外部评审机器的硬编码依赖。
