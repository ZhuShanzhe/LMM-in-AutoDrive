# 三场景路线断点与分段测试索引

## 已完成的划分

当前三个场景都使用 CARLA 0.9.16 的 `Town05_Opt` 官方地图，但路线和累计里程不同。本目录分别保存三条路线的断点，不修改原地图、任务语义、事件设置和模型权重。

| 场景 | 路线长度 | 任务主段 | 道路／事件细段 | 任务数 | 显式事件数 |
| --- | --- | --- | --- | --- | --- |
| 场景一 | 5000 米 | 15 | 34 | 15 | 0 |
| 场景二 | 8000 米 | 15 | 91 | 15 | 5 |
| 场景三 | 6000 米 | 9 | 38 | 8 | 7 |

场景一没有配置独立事件表，不代表没有交通参与者、信号灯或交通风险；它们仍由原交通配置与真实地图提供。场景三第一个主段是首条指令之前的巡航段。

详细列表：[场景一](scene_1/README.md)、[场景二](scene_2/README.md)、[场景三](scene_3/README.md)。

每条路线采用两级索引：`S2-P03` 是任务主段，`S2-012` 是更小的道路／事件细段。按问题定位细段，按任务衔接选择主段。各级区间分别连续覆盖全程，没有缺口；细段核心范围互不重叠，回放范围可以重叠。

## 断点依据

- 任务：指令宣布、激活、显式任务窗口结束、路线转向指令。
- 事件：进入、发生位置、配置的退出里程、施工锥桶与作业区域边界、参与者布置位置。
- 道路：路口进入／退出、实际弯道、坡度变化、窄车道、同向车道数量变化及信号灯停车线附近。
- 历史故障：关联 `test/10_02/whole_routes_001` 中的停滞位置、具体原因和原始日志路径。

道路特征来自当前官方地图与正式入口使用的路线构造函数，不是手工假设的固定地图坐标。弯道诊断阈值为曲率 0.012 /m，坡度阈值为 3%，窄车道阈值为 2.8 米；这些是分段阈值，不是驾驶安全判定阈值。道路特征周围保留约 40 米缓冲；临近变化保留为子断点，避免大量碎段；普通长区间最长约 400 米。

场景一使用当前配置重新进行任务—地图几何绑定。里程以本次生成的 `scene_1_bound_config.json` 为准，不能直接套用其他版本的任务位置。文件保存地图、场景配置及路线采样的指纹；原配置发生变化时，查询入口拒绝使用过期索引。

## 文件结构

```text
diagnostic_sections/
  README.md
  index.json
  scene_1_bound_config.json
  scene_1/
    README.md
    sections.json
    sections.csv
    geometry.json
  scene_2/
    README.md
    sections.json
    sections.csv
    geometry.json
  scene_3/
    README.md
    sections.json
    sections.csv
    geometry.json
```

各场景 `README.md` 给出主段、细段和任务原文；CSV 便于查看与筛选；JSON 保存断点、坐标、道路特征、任务、事件和历史故障；`geometry.json` 是沿路线的实际道路采样，不包含模型权重或训练数据。

## 查询测试段

在服务器仓库根目录运行，查询不需要 GPU 或启动 CARLA：

```bash
cd /root/autodl-tmp/worktrees/challenge-track
PYTHON=/root/autodl-tmp/conda_envs/command_parser/bin/python

# 查看场景二的任务主段
$PYTHON experiment/CARLA/tools/inspect_route_section.py --scene scene2

# 查看全部细段
$PYTHON experiment/CARLA/tools/inspect_route_section.py --scene scene2 --list-sections

# 按故障里程定位，并输出上下文、事件前置条件、坐标和检查项
$PYTHON experiment/CARLA/tools/inspect_route_section.py --scene scene3 --at-m 5349

# 按主段或细段编号查询
$PYTHON experiment/CARLA/tools/inspect_route_section.py --scene scene1 --phase S1-P04
$PYTHON experiment/CARLA/tools/inspect_route_section.py --scene scene3 --section S3-035
```

例如，约 5349 米的爬行停滞对应 `S3-035`，核心范围约为 5348.75–5420.75 米；保留任务／事件上下文后的回放范围约为 4849.28–5470.75 米。

## 启动与状态边界

核心区间用于定位与计量，不能直接等同于冷启动区间。索引保存建议回放起点、结束缓冲、起点坐标、事件前置状态、持续保留的事件演员，以及计划步骤、目标追踪、记忆、车速／车道和信号状态等需要保留的信息。建议传感器预热至少 5 秒，时序模型保留至少 30 秒有效历史；它们是测试配置要求，本目录不自动恢复这些状态。

| 原生场景入口 | 现有支持 | 使用边界 |
| --- | --- | --- |
| 场景一 | `--resume-route-progress-m`、`--stop-at-route-progress-m` | 可以设置路线起止；使用本目录绑定配置，不再次改变几何绑定；不会自动恢复完整模型和演员状态 |
| 场景二 | `--start-progress-m` | 原生冷启动会省略此前指令；有任务／事件依赖时必须从前置起点顺序回放或恢复状态；结束里程需要监控 |
| 场景三 | 原生入口没有按里程启动参数 | 断点坐标已提供，可用于日志回放及局部几何／接口／PID 测试；不声称已经支持任意位置的完整 VLA 与事件状态恢复 |

不把事件前置条件直接赋值为 `RESOLVED`，不把跳过的任务算作完成，不把到达区间终点当作任务完成。对于配置没有明确退出里程的事件，记录其已知空间范围，但事件结束仍需实际状态反馈。保留演员的事件也列入 `persistent_actor_event_ids`，不能仅因离开核心区间就删除演员。

历史故障的路线同源性统一标记为 `not_verified`，关联只用于定位，不保证旧记录中的同一里程必然对应本次路线上的同一物理位置。

## 生成与核验

生成需要已启动的 CARLA 服务，会读取／加载 Town05_Opt 并使用正式路线构造器；不要与正在运行的仿真测试共用服务。场景一绑定会校验当前配置与地图，已有绑定仍有效时复用。

```bash
export CARLA_ROOT=/root/autodl-tmp/CARLA_0.9.16
export PYTHONPATH=.:experiment/CARLA
$PYTHON experiment/CARLA/tools/build_route_sections.py \
  --output experiment/CARLA/configs/diagnostic_sections

$PYTHON -m pytest -q experiment/CARLA/tests/test_route_sections.py
```

分段单元测试通过 10 项，覆盖连续全程覆盖、任务／事件前置区间、嵌套事件位置、重复连接点、非法采样和参数检查。生成索引不代表这些主段、细段已完成闭环性能验证。

三份生成索引均通过全程覆盖、任务／事件覆盖和查询入口核验，共 12 次查询检查；最终模块回归为 1455 项测试、166 项子测试全部通过。
