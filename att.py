import torch
import whisper
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

def generate_txt(audio_path: str, model_name: str = "base"):
    #使用 Whisper 转录音频文件并生成 TXT 文件。
    
    is_gpu_available = check_gpu()
    if not os.path.exists(audio_path):
        print(f"Error: 找不到档案 '{audio_path}'")
        print("请确保音频文件与脚本在同一目录下，或是提供完整路径。")

    print(f"正在加载 Whisper 模型: '{model_name}'... (第一次使用可能需要下载模型)")
    model = whisper.load_model(model_name)

    print(f"正在转录音频 '{audio_path}'。")
    if is_gpu_available:
        print("正在使用 GPU 运行，这将大大加快速度！")
    result = model.transcribe(audio_path, verbose=False)
    print("转录完成！生成 TXT 文件中...")
    base_filename = os.path.splitext(os.path.basename(audio_path))[0]

    # 生成 TXT 文件
    txt_filename = base_filename + ".txt"
    with open(txt_filename, "w", encoding="utf-8") as txt_file:
        txt_file.write(result['text'])
    print(f"已生成 TXT 檔案: {txt_filename}")

    print(f"\n侦测到的语言: {result['language']}")
    print("\n完整转录文字內容:")
    print(result['text'])


if __name__ == '__main__':
    # --- 设定区 ---
    AUDIO_FILE = "moon.mp3"  # 音频文件路径 (可以是 .wav, .mp3 等格式)
    # 模型选项 (由小到大): tiny, base, small, medium, large, large-v2, large-v3
    MODEL_SIZE = "large-v3"
    
    generate_txt(AUDIO_FILE, MODEL_SIZE)
