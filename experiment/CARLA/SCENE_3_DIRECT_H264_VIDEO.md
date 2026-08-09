# 场景三：第三人称追尾 RGB 直编码 H.264 视频

场景三现在复用控制实验的录像结构：

- `evaluation/camera.py` 挂载第三人称 `chase_rgb`（类型为 `sensor.camera.rgb`）。
- 每次 `world.tick()` 后，运行器按仿真帧号取出对应图像。
- HUD 在原始 BGRA 帧送入编码器之前叠加。
- `evaluation/video.py` 将帧直接写入 ffmpeg，编码为 H.264、`yuv420p` MP4。
- 不再依赖“先保存全部 PNG，再二次合成视频”。
- 晴天提交模式为场景三第三人称追尾相机锁定手动曝光，防止浅色道路和车道线过曝。

原有四路 RGB 相机模式仍保留，用于官方相机合同与离线回归。传入
`--video-output` 后会切换到提交视频所用的同步第三人称追尾录像链。

## 一次性检查

在仓库根目录执行：

```bash
python -m py_compile \
  experiment/CARLA/run_emergency_response_6km.py \
  experiment/CARLA/evaluation/camera.py \
  experiment/CARLA/evaluation/video.py

python experiment/CARLA/run_emergency_response_6km.py \
  --validate-config-only

command -v ffmpeg
```

最后一条必须输出 ffmpeg 的绝对路径。如果环境里没有 ffmpeg，需要先在
当前 Conda 环境安装，或在运行时通过 `--ffmpeg /绝对路径/ffmpeg` 指定。

## 完整 6 km 提交视频

在 CARLA 服务端已启动、2000/2001 端口正常监听后执行：

```bash
cd /mnt/beegfs/home/reco/projects/LMM-in-AutoDrive-main-upload

set -o pipefail

SCENE3_VIDEO_DIR="$PWD/experiment/CARLA/outputs/scene3_direct_h264_${SLURM_JOB_ID}"
FFMPEG_BIN="$(command -v ffmpeg)"

python \
  experiment/CARLA/run_emergency_response_6km.py \
  --duration 720 \
  --presentation-lighting clear-daylight \
  --camera-width 1920 \
  --camera-height 1080 \
  --video-fps 30 \
  --video-output "$SCENE3_VIDEO_DIR/scene3_emergency_response_6km.mp4" \
  --video-overlay \
  --terminal-hold-s 2 \
  --ffmpeg "$FFMPEG_BIN" \
  --output-dir "$SCENE3_VIDEO_DIR" \
  --require-complete-scene \
  2>&1 | tee /tmp/scene3_direct_h264.log

SCENE3_STATUS=${PIPESTATUS[0]}
echo "SCENE 3 DIRECT H264 STATUS: $SCENE3_STATUS"
```

不要添加 `--draw-presentation-lane-markings`。该开关只用于诊断；默认关闭可
避免人工 debug 车道线在视频里过亮。

## 输出验证

```bash
VIDEO_PATH="$SCENE3_VIDEO_DIR/scene3_emergency_response_6km.mp4"

ls -lh "$VIDEO_PATH"

ffprobe \
  -v error \
  -select_streams v:0 \
  -show_entries stream=codec_name,pix_fmt,width,height,nb_frames \
  -show_entries format=duration \
  -of default=noprint_wrappers=1 \
  "$VIDEO_PATH"

python -m json.tool \
  "$SCENE3_VIDEO_DIR/scene_summary.json"
```

视频应显示 `codec_name=h264`、`pix_fmt=yuv420p`、`1920x1080`。场景总结
应显示 `route_completed: true`、七个事件全部 `RESOLVED`、
`collision_count: 0` 和 `complete_scene_success: true`。

如需同时保留第三人称 PNG 证据，可在正式命令里额外加入 `--record-images`；
提交视频本身不需要该参数。
