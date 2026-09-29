# 挑战赛道 x86 测试入口

本页只适用于当前挑战分支，不兼容基础赛道启动配置，不涉及 J6P。输入从文本开始；语音模块独立测试。x86 指主机架构，CARLA RGB 渲染仍需要可用图形设备；无 GPU 的模型客户端可连接另一台运行 CARLA 0.9.16 的 x86 主机。

## 当前结论

可以开始组员联合冒烟测试、真实同源数据采集和模块指标测量；不能直接作为三场景正式验收通过版本。

2026-09-26 已检查三个正式运行器可导入，23 个资产校验正确，Town04/Town05 新版配置均能生成。三个场景目录均 `config_valid=true`，但 `benchmark_ready=false`；该目录检查本身不证明路线几何、事件及独立真值采集有效。当前服务器没有运行 CARLA，连接 2000 端口失败，未进行本次物理闭环测试。不将 CPU 诊断结果当作 GPU 或目标平台性能。

## 配置环境

Linux x86_64、Python 3.12.13、CARLA 客户端和服务端均为 0.9.16。现有服务器可使用 `/root/autodl-tmp/conda_envs/command_parser/bin/python`。

```bash
# 在仓库根目录；按 models/README.md 下载冻结资产
export MODEL_ROOT=/root/autodl-tmp/models/challenge-assets-pinned
python lightweight_vla_adapter/scripts/download_challenge_assets.py --model-root "$MODEL_ROOT"
```

新 GPU 环境依赖见 `lightweight_vla_adapter/requirements-challenge.txt`。仅两个模型的 CPU 自检容器见该模块 `deployment/README.md`，它不是包含 CARLA 的整链路镜像。当前完整环境是在既有服务器验证，未声称全新安装或 Docker 构建通过。

CARLA 服务需在具备渲染能力的 Linux x86 主机以非 root 用户启动，例如在安装目录运行：

```bash
./CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Low -carla-rpc-port=2000
```

`RenderOffScreen` 不是禁用渲染；不要用无渲染模式生成空白 RGB 充当感知测试。远程服务须允许客户端访问对应端口，且每次只运行一个测试进程控制该世界。

## 统一运行

所有命令从仓库根目录执行，输出目录必须不存在。默认不执行仿真，仅校验、生成配置与实际运行命令：

```bash
python experiment/CARLA/tools/run_challenge_x86.py \
  --scene scene1 --model-root "$MODEL_ROOT" \
  --output /root/autodl-tmp/test_runs/scene1_prepare
```

在空闲 CARLA 服务上先检查连接，再执行 60 秒模型冒烟；`CARLA_HOST` 改为实际服务器地址：

```bash
export CARLA_HOST=127.0.0.1
python experiment/CARLA/tools/run_challenge_x86.py \
  --scene scene1 --model-root "$MODEL_ROOT" --host "$CARLA_HOST" \
  --device cpu --duration-s 60 --execute \
  --output /root/autodl-tmp/test_runs/scene1_smoke
```

分别将 `scene1` 改为 `scene2`、`scene3` 并使用不同输出目录。模型主机有 GPU 时用 `--device cuda`；仍采用 FP32，不静默降级。`--check-server` 只核对服务版本，不加载地图或驱动车辆。`--fuse-conv-bn` 用于优化对照，默认不启用。

原 `scripts/run_universal_vla.sh` 已替换为此挑战入口的包装，不再读取旧基础提交配置。它要求场景与新输出目录，默认执行仿真，不再接受旧输出基目录语义。

入口固定使用新版 `phase_recovery_v1` 记忆配置和当前 VLA/ModernBERT 组合，按场景生成灯头地图路径；没有切换为 Traffic Manager、规则基线或复合任务代驾。独立任务评测与实际 VLA 传感器记录默认启用。场景一使用 Town04，二三使用 Town05。其他地图不在此入口支持范围内。

输出中 `preflight.json` 记录环境版本、实际命令、场景检查及阻塞原因；`runtime.json` 是本次模型配置；`runner.log` 和 `run/` 保存运行器输出。`runner_finished` 只表示进程正常结束，不代表任务完成、安全或精度达标。默认 60 秒是联调冒烟，不是完整路线或 1000 帧采集完成保证。

## 开始指标测试的门槛

1. 刘旭在实际服务上验证路线与事件几何，完成各场景短测，确认四视角 RGB、LiDAR、车辆状态与独立真值真实写入，缺帧必须报告。检查碰撞、实线等已有安全记录；现有独立安全事件覆盖不等于覆盖全部交通违规。
2. 每场景采集至少 1000 个有效同源帧。不能用不同帧相机/点云拼齐，也不能把 1000 个仿真 tick 当成 1000 个有效模型输入。按采集器实际目录结构调用 `evaluation.sensor_replay`，校验数据哈希后再比较原版与优化版。
3. 黄皓星对同源数据统计关键目标、灯色、追踪连续性和语义对齐；王皓然检查指令顺序、相对目标锁定、风险阻断恢复和物理完成反馈。`ControlPlanState` 使用 1.1.0。
4. 朱善哲测模型与文本到控制链路的 P50/P95/P99、最大延时和超时率。固定设备、线程、精度、输入、预热与测量范围；仿真 tick/网络等待与模型计算分开报告，数据记录磁盘开销不能隐去。

以上门槛满足后才能汇总三场景正式结果。短测或单项测试可先开始，但未触发的任务必须标记未覆盖，模型反馈不能代替独立评测真值。HHX 的 16.09% 改善只属于其 RTX 3050 适配器对比，不是当前全链路指标。

## 已知边界

新增入口后的 CPU 回归为 1375 项测试、207 项子测试通过（57.33 秒）。原始记录位于服务器 `/root/autodl-tmp/review_logs/20260926/x86_launcher_tests.log`，三个准备报告在同目录 `x86_prepare_scene1/` 至 `x86_prepare_scene3/`；服务连接阻塞记录在 `x86_server_check/`。这些记录不包含物理驾驶结果。

- 当前无卡主机只能做模型 CPU 诊断或连接外部 CARLA；本轮没有宣称已跑通物理闭环。
- 场景目录仍有路线几何、事件准备、任务窗口及真值运行待核验提示；不删除提示、不降低判据换取 ready。
- 公共环境尚未提供本轮三场景 1000 帧验证结果，不能填入指标成绩。
- 本次不训练模型、不改变安全门、任务阈值、事件或其他组员算法。
# 0929 新地图整合说明

当前整理分支以 `lx-challenge` 新地图配置为准，三个场景均使用 Town05_Opt。场景一启动参数与静态信号灯配置同步使用 Town05，不可继续传入 Town04。旧地图测试记录不代表本轮结果。任务编号、速度目标、触发距离、依赖和评测来源指纹已重新绑定；38 项任务的完整闭环结果仍需在新地图实测。

本说明描述开发联调入口，不等于官方地平线 Docker 实测。初审正式材料及未完成项见 [材料索引](../submission/initial_review_20260929/README.md)。
