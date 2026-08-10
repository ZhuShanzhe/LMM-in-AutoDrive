# 正式提交检查清单

压缩包：`面向智能驾驶的大模型应用场景研究-南京大学.zip`

邮件主题：`【揭榜挂帅大赛作品】面向智能驾驶的大模型应用场景研究-南京大学`

## 必须项

- [ ] `image.tar` 可由 `docker load` 加载，标签为 `lmm-autodrive-basic:final`；
- [ ] 镜像包含官方 Bench2Drive 固定提交、CARLA 0.9.16、代码、环境、模型和标准 `AutonomousAgent`，可断网运行；
- [ ] 镜像静态校验输出 `RUNTIME_VERIFICATION_OK`；
- [ ] `README.md` 包含运行步骤、输入输出接口和预期输出；
- [ ] `weights/` 的 VLA、部署配置和命令解析模型 SHA256 通过；
- [ ] 技术方案 PDF 可打开且与最终代码、权重和日志一致；
- [ ] `metrics.zip` CRC 正常并包含 `scene1/`、`scene2/`、`scene3/` 和 `TEST_REPORT.md`；
- [ ] 场景日志来自同一 Stage‑8 checkpoint 和同一通用链路；
- [ ] 报告如实区分场景三 r32 全程与 r33 启动 smoke；
- [ ] 不包含 `demo.mp4`，不包含数据集、历史失败视频或中间训练缓存。

## 发件信息

- 收件：`tc-lixinzhu@dfmc.com.cn`、`tangyuh@dfmc.com.cn`；
- 抄送：`tc-linyao@dfmc.com.cn`、`panxinze@dfmc.com.cn`；
- 正文作品简介不超过 200 字；
- 大文件使用长期有效网盘链接，并附提取码；
- 每队最多提交两次，以最后一次为准。
