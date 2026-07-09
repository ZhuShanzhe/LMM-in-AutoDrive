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

## 2026-07-09

### 目标

配置 DriveLM-CARLA / PDM-Lite 实验环境，适配 RTX 5090 (`sm_120`)，并确认 CARLA 数据下载路径和测试方法。

### 环境

- 主机：AutoDL `autodl-pro-7834aae1212f`
- GPU：NVIDIA GeForce RTX 5090
- Driver：580.105.08
- `nvidia-smi` CUDA：13.0
- 显存：32607 MiB
- 当前可用磁盘：约 65 GB

### 已完成

- 创建 conda 环境 `drivelm_carla`。
- 安装 Python 3.10.20。
- 安装适配 RTX 5090 的 PyTorch CUDA 12.8 wheel：
  - `torch 2.11.0+cu128`
  - `torchvision 0.26.0+cu128`
  - `torchaudio 2.11.0+cu128`
- 安装 DriveLM-CARLA / PDM-Lite 相关依赖：
  - `carla==0.9.15`
  - `laspy[lazrs,laszip]`
  - `opencv-python==4.6.0.66`
  - `pygame==2.6.0`
  - `py-trees==0.8.3`
  - `shapely==2.0.4`
  - `gym==0.26.2`
  - `scipy`, `h5py`, `tqdm`, `rdp`, `ujson` 等。
- 为官方 `DriveLM-CARLA` 分支创建独立 worktree：
  - `/root/autodl-tmp/LMM-in-AutoDrive/experiments/drivelm/external/DriveLM-CARLA`
- 完成 GPU 测试：
  - `torch.cuda.is_available() == True`
  - GPU 名称：`NVIDIA GeForce RTX 5090`
  - compute capability：`(12, 0)`
  - CUDA 矩阵乘法测试通过。
- 完成脚本入口测试：
  - `vqa_dataset/extract_keyframes.py --help`
  - `vqa_dataset/carla_vqa_generator_main.py --help`

### 重要调整

官方 DriveLM `environment.yml` 和 challenge requirements 使用 CUDA 11.7 / torch 2.0.x。该配置不适合 RTX 5090 (`sm_120`)，因此本环境没有照抄旧 torch，而是使用 CUDA 12.8 版 PyTorch。

### CARLA 数据情况

官方 DriveLM-CARLA README 提到两层数据：

1. DriveLM-CARLA GVQA labels：
   - `drivelm_carla_keyframes.txt`
   - `drivelm_carla_vqas.zip`
   - 位于 HuggingFace `OpenDriveLab/DriveLM`。
   - 当前访问返回 `GatedRepo`，需要 HuggingFace 账号申请访问并提供 token。

2. PDM-Lite CARLA raw data：
   - 位于 HuggingFace `autonomousvision/PDM_Lite_Carla_LB2`。
   - 官方说明解压后 330+ GB。
   - 当前 AutoDL 仅约 65 GB 可用，无法下载全量数据。
   - 单个 `Town01/data/ControlLoss.zip` 大约 415 MB，可作为最小样本测试。

### 未完成

- 未下载全量 PDM-Lite CARLA 数据：磁盘不足。
- 未下载 DriveLM-CARLA GVQA labels：HuggingFace gated repo，需要用户 token。
- 未下载最小 PDM-Lite 样本：下载操作被审批系统拦截，需要用户明确批准后再执行。

### 测试命令

```bash
cd /root/autodl-tmp/LMM-in-AutoDrive
bash experiments/drivelm/scripts/test_carla_env.sh
```

### 下一步

1. 用户申请 HuggingFace `OpenDriveLab/DriveLM` 数据访问权限。
2. 用户提供 HuggingFace token，或手动下载 `drivelm_carla_keyframes.txt` 和 `drivelm_carla_vqas.zip` 后上传到服务器。
3. 若要下载 PDM-Lite full raw data，需要扩容到至少 400 GB 可用空间。
4. 若只做 smoke test，可先明确批准下载 `Town01/ControlLoss` 小样本。

## 2026-07-09 20:14

### 目标

下载 PDM-Lite CARLA 小样本，并验证 DriveLM-CARLA keyframe 提取与 VQA graph 生成流程。

### 数据

- 数据源：`autonomousvision/PDM_Lite_Carla_LB2`
- 下载方式：`hf-mirror.com`
- 已下载：
  - `README.md`
  - `Town01/results.zip`
  - `Town01/data/ControlLoss.zip`
- 解压后路径：
  - `/root/autodl-tmp/LMM-in-AutoDrive/experiments/drivelm/data/PDM_Lite_Carla_LB2`
- 解压后大小：约 `460M`
- 磁盘状态：`374G` 总容量，约 `16G` 已用，约 `359G` 可用。

### 注意事项

`ControlLoss.zip` 解压后默认多一层 `ControlLoss/ControlLoss/Route*_Rep0`。DriveLM-CARLA 脚本期望的路径是 `Town01/data/ControlLoss/Route*_Rep0`，因此已经将 route 目录上移一层。

### Keyframe 提取

官方 `extract_keyframes.py` 中 `--filter-routes-for-DS` 参数为 `store_true`，但默认值也是 `True`，命令行无法关闭。对小样本 smoke test，直接调用 `main()` 并设置 `filter_routes_for_DS=False`。

结果：

```text
Found 273 keyframes out of 911 frames
```

输出文件：

```text
/root/autodl-tmp/LMM-in-AutoDrive/experiments/drivelm/outputs/carla_sample_keyframes_no_ds_filter.txt
```

`wc -l` 显示 `272`，原因是最后一行无换行符；实际脚本输出为 273 个 keyframes。

### VQA Graph 生成

执行小规模测试：

```bash
python vqa_dataset/carla_vqa_generator_main.py \
  --path-keyframes /root/autodl-tmp/LMM-in-AutoDrive/experiments/drivelm/outputs/carla_sample_keyframes_no_ds_filter.txt \
  --data-directory /root/autodl-tmp/LMM-in-AutoDrive/experiments/drivelm/data/PDM_Lite_Carla_LB2 \
  --output-graph-directory /root/autodl-tmp/LMM-in-AutoDrive/experiments/drivelm/outputs/carla_sample_vqa_graph_5 \
  --output-graph-examples-directory /root/autodl-tmp/LMM-in-AutoDrive/experiments/drivelm/outputs/carla_sample_vqa_examples_5 \
  --random-subset-count 5 \
  --sample-frame-mode keyframes
```

结果：

```text
Stats saved.
num_frames: 1
num_questions: 23
num_objects: 1
```

生成样例：

```text
/root/autodl-tmp/LMM-in-AutoDrive/experiments/drivelm/outputs/carla_sample_vqa_graph_5/Town01/ControlLoss/Route2_Rep0/0012.json
```

样例中包含 `key_object_infos` 和 `QA`，其中 QA 覆盖 perception、planning、prediction 等类型。

### 后续测试命令

下载小样本：

```bash
cd /root/autodl-tmp/LMM-in-AutoDrive
bash experiments/drivelm/scripts/download_carla_sample.sh
```

跑小样本 keyframe + VQA 测试：

```bash
cd /root/autodl-tmp/LMM-in-AutoDrive
bash experiments/drivelm/scripts/run_carla_sample_test.sh
```

复跑验证结果：

```text
Found 274 keyframes out of 911 frames
num_frames: 1
num_questions: 23
```

注：keyframe 提取中包含随机抽样，并且使用多线程处理 route，因此 273/274 这类 1 帧差异属于 smoke test 中可接受的波动。

### 下一步

- 若继续扩展样本，可按场景逐个下载 PDM-Lite zip，避免全量 zip 同时占用磁盘。
- 若要使用官方 DriveLM-CARLA GVQA labels，仍需要 HuggingFace gated repo 权限和 token。
