# 正式提交检查清单

压缩包名称：`面向智能驾驶的大模型应用场景研究-南京大学.zip`

邮件主题：`【揭榜挂帅大赛作品】面向智能驾驶的大模型应用场景研究-南京大学`

## 必须项

- [ ] `image.tar` 存在、可被 `docker load` 加载，镜像标签为 `lmm-autodrive-basic:final`。
- [ ] 镜像基于 Bench2Drive/CARLA 运行环境，包含代码，可在断网条件下独立运行。
- [ ] `README.md` 中三条命令已在最终镜像内逐条验证。
- [ ] `面向智能驾驶的大模型应用场景研究-南京大学_技术方案.pdf` 可正常打开，内容与最终代码/权重一致。
- [ ] 权重若未放入镜像，`weights/` 中 VLA 与命令解析模型齐全且 SHA256 校验通过。

## 评估材料

- [ ] `metrics.zip` 可解压，包含 `scene1/`、`scene2/`、`scene3/` 与总报告。
- [ ] 三个场景日志均来自同一最终 checkpoint 和同一通用链路。
- [ ] 报告如实记录碰撞、禁行线、任务、fallback 和延迟，不把 smoke test 写成完整验收。
- [ ] 不提交 `demo.mp4`；视频仅在服务器留作内部复核。

## 发件前

- [ ] 邮件正文作品简介不超过 200 字。
- [ ] 正文注明附件或百度网盘二选一；网盘链接不设置短时效。
- [ ] 收件人：`tc-lixinzhu@dfmc.com.cn`、`tangyuh@dfmc.com.cn`。
- [ ] 抄送：`tc-linyao@dfmc.com.cn`、`panxinze@dfmc.com.cn`。
- [ ] 运行 `python tools/build_submission.py` 生成 ZIP 与 SHA256，并在另一目录解压复核。
