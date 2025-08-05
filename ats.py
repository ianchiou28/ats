import torch
import whisper
import datetime
import os

def check_gpu():
    """Checks if PyTorch can access the GPU."""
    print("--- GPU & CUDA 检查 ---")
    if not torch.cuda.is_available():
        print("warning: CUDA 不可用。PyTorch 正在使用 CPU 运行。")
        print("转录速度会非常慢。")
        print("请按照以下步骤安装支持 CUDA 的 PyTorch 版本：")
        print("1. 卸载当前的 PyTorch: pip uninstall torch torchvision torchaudio")
        print("2. 安装支持 CUDA 的 PyTorch: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        return False
    
    device_count = torch.cuda.device_count()
    print(f"CUDA 可用！找到 {device_count} 个 GPU。")
    for i in range(device_count):
        print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
    print("--------------------------\n")
    return True

def generate_srt(audio_path: str, model_name: str = "base"):

    #Transcribes an audio file using Whisper and generates an SRT file.

    # Run the GPU check first
    is_gpu_available = check_gpu()

    # Check if the audio file exists
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at '{audio_path}'")
        return

    print(f"Loading Whisper model: '{model_name}'... (This may download the model on first use)")
    # If GPU is available, Whisper will automatically use it.
    # A powerful GPU like the RTX 4060 can easily handle the large models.
    model = whisper.load_model(model_name)

    print(f"Transcribing audio from '{audio_path}'.")
    if is_gpu_available:
        print("Running on GPU, this will be much faster!")
    
    result = model.transcribe(audio_path, verbose=False)

    print("Transcription complete. Generating SRT file...")

    # Define the output file path
    srt_filename = os.path.splitext(os.path.basename(audio_path))[0] + ".srt"

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

    print(f"已生成 SRT 檔案: {srt_filename}")

    print(f"\n侦测到的语言: {result['language']}")
    print("\n完整转录文字內容:")
    print(result['text'])


if __name__ == '__main__':
    # --- 设定区 ---
    AUDIO_FILE = "audio.wav"  # 音频文件路径 (可以是 .wav, .mp3 等格式)
    # 模型选项 (由小到大): tiny, base, small, medium, large, large-v2, large-v3
    MODEL_SIZE = "large-v3"
    
    generate_srt(AUDIO_FILE, MODEL_SIZE)
