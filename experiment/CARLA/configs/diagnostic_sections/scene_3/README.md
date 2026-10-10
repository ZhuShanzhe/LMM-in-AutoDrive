# scene_3 分段索引

地图：`Town05_Opt`；路线：6000 米；8 个任务主段，40 个道路/事件细段。

里程是沿本场景路线的累计里程，不是地图坐标。场景一使用当前配置重新绑定后的任务位置。

## 任务主段

| 主段 | 核心里程（米） | 包含细段 |
| --- | --- | --- |
| S3-P01 | 0.0–800.0 | S3-001, S3-002, S3-003, S3-004 |
| S3-P02 | 800.0–1250.0 | S3-005, S3-006, S3-007 |
| S3-P03 | 1250.0–2850.0 | S3-008, S3-009, S3-010, S3-011, S3-012, S3-013, S3-014, S3-015, S3-016, S3-017 |
| S3-P04 | 2850.0–2975.0 | S3-018 |
| S3-P05 | 2975.0–3150.0 | S3-019, S3-020, S3-021 |
| S3-P06 | 3150.0–3250.0 | S3-022 |
| S3-P07 | 3250.0–5100.0 | S3-023, S3-024, S3-025, S3-026, S3-027, S3-028, S3-029, S3-030, S3-031, S3-032, S3-033, S3-034 |
| S3-P08 | 5100.0–6000.0 | S3-035, S3-036, S3-037, S3-038, S3-039, S3-040 |

## 道路与事件细段

| 测试段 | 核心里程（米） | 带前置上下文的回放范围（米） | 任务 | 事件 | 道路特征 | 历史故障数 |
| --- | --- | --- | --- | --- | --- | --- |
| S3-001 | 0.0–145.8 | 0.0–195.8 | 巡航 | — | grade, traffic_signal | 0 |
| S3-002 | 145.8–410.5 | 0.0–460.5 | 巡航 | — | 普通路段 | 0 |
| S3-003 | 410.5–675.3 | 259.6–725.3 | 巡航 | — | 普通路段 | 0 |
| S3-004 | 675.3–800.0 | 524.7–850.0 | 巡航 | — | grade | 1 |
| S3-005 | 800.0–866.1 | 648.9–916.1 | scene3_general_hazard | — | grade, junction, traffic_signal | 0 |
| S3-006 | 866.1–1150.0 | 648.9–1200.0 | scene3_general_hazard | — | traffic_signal | 2 |
| S3-007 | 1150.0–1250.0 | 998.6–1300.0 | 巡航 | scene3_cut_in | 普通路段 | 0 |
| S3-008 | 1250.0–1432.0 | 1099.8–1482.0 | scene3_cut_in_decelerate | scene3_cut_in | 普通路段 | 2 |
| S3-009 | 1432.0–1550.0 | 1099.8–1600.0 | scene3_cut_in_decelerate | scene3_cut_in | grade, junction, traffic_signal | 0 |
| S3-010 | 1550.0–1672.3 | 1099.8–1722.3 | 巡航 | scene3_cut_in | grade | 0 |
| S3-011 | 1672.3–1937.0 | 1522.0–1987.0 | 巡航 | — | 普通路段 | 0 |
| S3-012 | 1937.0–2201.8 | 1786.1–2251.8 | 巡航 | — | 普通路段 | 0 |
| S3-013 | 2201.8–2295.8 | 2051.2–2345.8 | 巡航 | — | grade | 0 |
| S3-014 | 2295.8–2367.8 | 2144.9–2417.8 | 巡航 | — | grade, junction, traffic_signal | 1 |
| S3-015 | 2367.8–2439.6 | 2215.8–2489.6 | 巡航 | — | junction, traffic_signal | 0 |
| S3-016 | 2439.6–2644.8 | 2287.8–2694.8 | 巡航 | — | 普通路段 | 0 |
| S3-017 | 2644.8–2850.0 | 2494.6–2900.0 | 巡航 | scene3_advance_warning | 普通路段 | 0 |
| S3-018 | 2850.0–2975.0 | 2698.6–3025.0 | scene3_work_zone_warning | scene3_advance_warning, scene3_cone_taper | 普通路段 | 2 |
| S3-019 | 2975.0–3050.0 | 2698.6–3100.0 | scene3_work_zone_warning, scene3_right_lane_closure | scene3_advance_warning, scene3_cone_taper | junction, traffic_signal | 2 |
| S3-020 | 3050.0–3100.0 | 2698.6–3150.0 | scene3_right_lane_closure | scene3_advance_warning, scene3_cone_taper, scene3_work_zone | grade, traffic_signal | 0 |
| S3-021 | 3100.0–3150.0 | 2823.3–3200.0 | scene3_right_lane_closure | scene3_cone_taper, scene3_work_zone | grade | 2 |
| S3-022 | 3150.0–3250.0 | 2823.3–3300.0 | scene3_pass_work_zone | scene3_cone_taper, scene3_temporary_pedestrian, scene3_work_zone | grade | 6 |
| S3-023 | 3250.0–3425.0 | 2949.5–3475.0 | scene3_pass_work_zone, scene3_worker_crossing | scene3_temporary_pedestrian, scene3_work_zone | 普通路段 | 3 |
| S3-024 | 3425.0–3450.0 | 2949.5–3500.0 | scene3_pass_work_zone | scene3_temporary_pedestrian, scene3_work_zone | 普通路段 | 0 |
| S3-025 | 3450.0–3728.3 | 2949.5–3778.3 | 巡航 | scene3_work_zone | 普通路段 | 0 |
| S3-026 | 3728.3–3822.3 | 3577.7–3872.3 | 巡航 | — | grade | 0 |
| S3-027 | 3822.3–3894.3 | 3671.4–3944.3 | 巡航 | — | grade, junction, traffic_signal | 0 |
| S3-028 | 3894.3–3966.1 | 3742.3–4016.1 | 巡航 | — | junction, traffic_signal | 0 |
| S3-029 | 3966.1–4225.5 | 3814.3–4275.5 | 巡航 | — | 普通路段 | 0 |
| S3-030 | 4225.5–4484.9 | 4074.0–4534.9 | 巡航 | — | 普通路段 | 0 |
| S3-031 | 4484.9–4582.0 | 4332.9–4632.0 | 巡航 | — | junction, traffic_signal | 0 |
| S3-032 | 4582.0–4725.3 | 4431.8–4775.3 | 巡航 | — | grade, traffic_signal | 2 |
| S3-033 | 4725.3–5000.0 | 4575.0–5050.0 | 巡航 | scene3_work_zone_exit | 普通路段 | 6 |
| S3-034 | 5000.0–5100.0 | 4849.3–5150.0 | 巡航 | scene3_work_zone_exit | 普通路段 | 0 |
| S3-035 | 5100.0–5250.0 | 4849.3–5300.0 | scene3_resume_normal_driving | scene3_work_zone_exit | 普通路段 | 0 |
| S3-036 | 5250.0–5348.7 | 4849.3–5398.7 | scene3_resume_normal_driving | scene3_work_zone_exit | grade | 0 |
| S3-037 | 5348.7–5420.8 | 4849.3–5470.8 | scene3_resume_normal_driving | scene3_work_zone_exit | grade, junction, traffic_signal | 1 |
| S3-038 | 5420.8–5492.6 | 4849.3–5542.6 | scene3_resume_normal_driving | scene3_work_zone_exit | junction, traffic_signal | 1 |
| S3-039 | 5492.6–5746.3 | 4849.3–5796.3 | scene3_resume_normal_driving | scene3_work_zone_exit | 普通路段 | 0 |
| S3-040 | 5746.3–6000.0 | 4849.3–6000.0 | scene3_resume_normal_driving | scene3_work_zone_exit | 普通路段 | 0 |

## 任务原文

- `scene3_general_hazard`：前方路况危险，保持安全车速（触发 800.0 米）
- `scene3_cut_in_decelerate`：突发车辆加塞，保持安全车距，必要时减速让行。（触发 1250.0 米）
- `scene3_work_zone_warning`：前方进入施工区域，注意观察（触发 2850.0 米）
- `scene3_right_lane_closure`：施工路段，减速并道至左侧车道（触发 2975.0 米）
- `scene3_pass_work_zone`：前方路况危险，保持安全车速（触发 3150.0 米）
- `scene3_worker_crossing`：有人突然横穿，立即减速让行（触发 3250.0 米）
- `scene3_resume_normal_driving`：危险路段结束，在安全且不超速的前提下逐步恢复至30公里每小时，沿规划路线行驶至终点。（触发 5100.0 米）
