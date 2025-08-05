# audio_to_srt

使用 OpenAI 的 Whisper 模型将音频文件转换为字幕文件（.srt）或文本文件（.txt）。

## 功能特点

- 支持 .wav、.mp3、mp4 等常见音视频格式
- 自动检测 GPU 可用性，支持 CUDA 加速
- 提供三种转换脚本：
  - `atb.py`: audio to both srt and txt, 同时生成 .srt 和 .txt 文件
  - `ats.py`: audio to srt, 仅生成 .srt 字幕文件
  - `att.py`: audio to txt, 仅生成 .txt 文本文件
- 支持多种 Whisper 模型尺寸，从小到大依次为：
  - tiny
  - base
  - small
  - medium
  - large
  - large-v2
  - large-v3

## 环境要求

- Python 3.x
- PyTorch
- Whisper
- CUDA 环境（可选，但强烈建议）

## 安装说明

1. 克隆此仓库（在cdn可以直接fork）：
```bash
git clone https://github.com/yourusername/audio_to_srt.git
cd audio_to_srt
```

2. 安装依赖（请按照顺序在终端运行）：
```bash
apt-get update #在云电脑上运行必要
apt-get install -y python3 python3-pip #在云电脑上运行必要
apt install python3.11-venv #在云电脑上运行必要
python3 -m venv venv #在云电脑上运行必要
source venv/bin/activate #在云电脑上运行必要

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -U openai-whisper
pip install git+https://github.com/openai/whisper.git
pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git
```
3. 安装ffmpeg：
```bash
choco install ffmpeg    #windows
apt install ffmpeg      #linux
```
4. 安装 setuptools-rust：
```bash
pip install setuptools-rust
```

## 使用方法

1. 将音频文件（如 audio.wav）放在脚本同一目录下

2. 选择需要的转换脚本运行：

- 同时生成 .srt 和 .txt：
```bash
python atb.py #windows
python3 atb.py #linux
```

- 仅生成 .srt 字幕文件：
```bash
python ats.py #windows
python3 ats.py #linux
```

- 仅生成 .txt 文本文件：
```bash
python att.py #windows
python3 att.py #linux
```

## 配置选项

在每个脚本文件的底部都有设置区域，您可以修改：

- `AUDIO_FILE`: 音频文件名称
- `MODEL_SIZE`: Whisper 模型大小，建议配置：
  - CPU 模式：使用 "tiny" 或 "base" 模型
  - GPU 模式：可以使用 "large-v3" 获得最佳效果

## 注意事项

1. 首次运行时会自动下载选定的 Whisper 模型
2. 建议使用 GPU 运行以获得更快的转录速度
3. 使用 large 系列模型时需要较大的 GPU 内存
4. 第一次使用可能会出现warning：CUDA不可用，不必担心，脚本已经设定好若你有GPU by nvedia，会自动下载适配的CUDA版本，若没有GPU，会自动使用CPU运行，速度会慢很多，但仍然可以正常使用

## 输出示例

运行脚本后将生成：

- `audio.srt`: 包含时间轴的字幕文件
- `audio.txt`: 纯文本转录结果
