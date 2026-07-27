# 音频录制模块 (Audio Recorder Module)

> 说明：本模块目前为独立组件，尚未集成到当前项目主流程中。

+ 模块概述：本模块提供了一个跨平台、稳定、易用的麦克风录音解决方案。它能自动探测系统音频输入设备，并在无可用麦克风时自动切换至模拟音频生成模式（用于测试）。

## 1. 项目结构

```text
recorder/
├── __init__.py
├── main.py          
├── config.py            
├── recorder.py          
├── utils.py  
├── requirements.txt 
└── README.md              
```

## 2. 安装与依赖

+ 环境要求：Python 3.8 及以上

+ 操作系统：Windows / Linux / macOS（需有音频硬件或使用模拟模式）
    
  + 若在 Linux 服务器或 WSL 上运行且无音频设备，`sounddevice` 可能无法加载 `PortAudio`，此时模块会自动降级为模拟模式，无需额外操作。

```shell
sudo apt-get install portaudio19-dev python3-pyaudio
pip install -r requirements.txt
```

## 3. Quick Start

```shell
# 录制 5 秒，指定输出文件名
python main.py --duration 5 --output my_speech.wav
```

+ 模拟模式说明：当系统检测不到音频输入设备（如服务器、无麦克风的 WSL 环境）或用户指定 `--simulate` 时，模块会生成白噪声代替真实录音。这样可以在无硬件的情况下测试整个录音流程，确保后续处理（如 ASR）能够正常运行。

    + 模拟音频同样会保存为标准的 WAV 文件，便于调试

+ 输出示例（正常录音且有麦克风）：
```text
Using input device: Microphone (Realtek High Definition Audio)
Recording will start in:
3...
2...
1...
Recording now...
Recording finished.
Audio saved to: recordings/my_speech.wav
Saved recording to: recordings/my_speech.wav
```
