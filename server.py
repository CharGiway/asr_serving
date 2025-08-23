# server.py
import tempfile
import os
import shutil
import whisper
import subprocess
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles

# 创建 FastAPI 应用实例
app = FastAPI()

# 定义文件保存路径，并确保目录存在
AUDIO_DIR = "./audio_files"
TRANSCRIPT_DIR = "./transcripts"
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)

# 自动选择设备：如果有可用的GPU，则使用GPU，否则使用CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"检测到设备: {device.upper()}")

# 加载 Whisper 模型，只在应用启动时加载一次
print("正在加载 Whisper 模型，这可能需要一些时间...")
try:
    # 使用用户脚本中的 "base" 模型，并指定加载设备
    model = whisper.load_model("base", device=device)
    print("✅ 模型加载完成")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    # 退出应用，因为无法加载模型
    raise RuntimeError("无法加载 Whisper 模型，请检查网络或配置")

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    接收音频文件，进行转录，并保存音频和结果。
    """
    # 1. 将上传的文件保存到临时文件
    temp_webm_path = os.path.join(tempfile.gettempdir(), file.filename)
    try:
        with open(temp_webm_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        return {"transcription": f"保存临时音频文件时出错：{e}"}

    # 2. 将临时 .webm 文件转换为 .wav 并保存到指定位置
    # 注意：每次请求都会覆盖之前的文件
    output_wav_path = os.path.join(AUDIO_DIR, "output.wav")
    try:
        # 使用 ffmpeg 将音频流转换为 wav 格式
        subprocess.run(["ffmpeg", "-i", temp_webm_path, "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-y", output_wav_path], check=True, capture_output=True)
        print(f"✅ 音频文件已转换为 WAV 并保存到: {output_wav_path}")
    except subprocess.CalledProcessError as e:
        return {"transcription": f"音频转换失败：{e.stderr.decode()}"}
    finally:
        os.remove(temp_webm_path)

    # 3. 使用 Whisper 模型进行转录
    try:
        # 用户脚本指定了中文语言
        result = model.transcribe(output_wav_path, language="zh")
        transcribed_text = result["text"].strip()
        print(f"📝 识别结果: {transcribed_text}")
    except Exception as e:
        return {"transcription": f"转录音频时出错：{str(e)}"}

    # 4. 将转录结果保存为 result.txt
    # 注意：每次请求都会覆盖之前的文件
    output_txt_path = os.path.join(TRANSCRIPT_DIR, "result.txt")
    try:
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(transcribed_text)
        print(f"💾 已保存识别结果到: {output_txt_path}")
    except Exception as e:
        return {"transcription": f"保存转录文件时出错：{e}"}

    # 5. 返回转录文本给前端
    return {"transcription": transcribed_text}

# 挂载静态文件目录，用于提供前端页面
app.mount("/", StaticFiles(directory=".", html=True), name="static")
