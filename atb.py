import torch
import whisper
import datetime
import os

def check_gpu():
    """Checks if PyTorch can access the GPU."""
    print("--- GPU & CUDA 检查 ---")
    if not torch.cuda.is_available():
        print("警告：您的 CUDA 环境无法使用，PyTorch 将以 CPU 模式运行。")
        print("转录速度会非常慢。")
        print("请参考以下步骤来安装支持 CUDA 的 PyTorch 版本:")
        print("1. 卸载现有版本: pip uninstall torch torchvision torchaudio")
        print("2. 安装 CUDA 版本: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        return False
    
    device_count = torch.cuda.device_count()
    print(f"CUDA 环境正常！找到 {device_count} 个 GPU。")
    for i in range(device_count):
        print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
    print("--------------------------\n")
    return True

def generate_transcription_files(audio_path: str, model_name: str = "base"):
    """
    使用 Whisper 转录音频，并同时生成 SRT 和 TXT 两种格式的文件。

    Args:
        audio_path (str): 音频文件的路径 (例如 "audio.wav")。
        model_name (str): 要使用的 Whisper 模型名称。
    """
    # 首先执行 GPU 检查
    is_gpu_available = check_gpu()

    # 检查音频文件是否存在
    if not os.path.exists(audio_path):
        print(f"错误：找不到音频文件 '{audio_path}'")
        print("请确认音频文件与此脚本在同一个文件夹，或提供完整路径。")
        # 如果文件不存在，为了演示目的，建立一个假的音频文件
        print("正在建立一个用于演示的假 'audio.wav' 文件。")
        try:
            from pydub import AudioSegment
            AudioSegment.silent(duration=10000).export(audio_path, format="wav")
            print(f"已建立假文件 '{audio_path}'，转录内容将会是空的。")
        except ImportError:
            print("\n无法建立假文件，因为 'pydub' 套件未安装。")
            print("请手动建立一个 'audio.wav' 文件后再试一次。")
            return

    print(f"正在加载 Whisper 模型: '{model_name}'... (首次使用时可能会自动下载模型)")
    # 如果 GPU 可用，Whisper 会自动调用它
    model = whisper.load_model(model_name)

    print(f"正在转录音频文件: '{audio_path}'。")
    if is_gpu_available:
        print("以 GPU 模式运行，速度会快很多！")
    
    result = model.transcribe(audio_path, verbose=False)

    print("转录完成，正在生成输出文件...")

    # 取得不含副檔名的基本文件名
    base_filename = os.path.splitext(os.path.basename(audio_path))[0]

    # --- 1. 生成 SRT 文件 (包含时间轴) ---
    srt_filename = base_filename + ".srt"
    print(f"正在生成 SRT 文件: {srt_filename}")
    with open(srt_filename, "w", encoding="utf-8") as srt_file:
        for i, segment in enumerate(result["segments"]):
            def format_time(td_seconds):
                total_seconds = int(td_seconds)
                milliseconds = int((td_seconds - total_seconds) * 1000)
                hours, remainder = divmod(total_seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

            start_time = format_time(segment['start'])
            end_time = format_time(segment['end'])

            srt_file.write(f"{i + 1}\n")
            srt_file.write(f"{start_time} --> {end_time}\n")
            srt_file.write(f"{segment['text'].strip()}\n\n")
    print(f"成功生成 SRT 文件: '{srt_filename}'")

    # --- 2. 生成 TXT 文件 (仅纯文本) ---
    txt_filename = base_filename + ".txt"
    print(f"\n正在生成 TXT 文件: {txt_filename}")
    with open(txt_filename, "w", encoding="utf-8") as txt_file:
        # 将 'text' 键中的完整转录结果写入文件
        txt_file.write(result['text'])
    print(f"成功生成 TXT 文件: '{txt_filename}'")


    print(f"\n侦测到的语言: {result['language']}")
    print("\n完整转录文本内容:")
    print(result['text'])


if __name__ == '__main__':
    # --- 设置区 ---
    # 请将您的音频文件名称放在这里
    AUDIO_FILE = "audio.wav" 
    # 选择模型大小。RTX 4060 建议使用 'large-v3' 以获得最佳品质
    # 其他选项 (由小到大): tiny, base, small, medium, large, large-v2, large-v3
    MODEL_SIZE = "large-v3"

    # --- 执行转录 ---
    generate_transcription_files(AUDIO_FILE, MODEL_SIZE)
