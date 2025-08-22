import sounddevice as sd
import numpy as np
import whisper
import tempfile
import os
import wave

# 录音参数
DURATION = 20  # 秒
SAMPLE_RATE = 16000
OUTPUT_WAV = "output.wav"
OUTPUT_TXT = "result.txt"

# 选择设备
device_info = sd.query_devices(kind="input")
print("使用设备:", device_info["name"])

# 录音
print("🎤 录音中...（20 秒）")
audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="int16")
sd.wait()
print("✅ 录音结束")

# 保存到 wav 文件
with wave.open(OUTPUT_WAV, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)  # int16 = 2 bytes
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(audio.tobytes())
print(f"🎵 已保存音频到 {OUTPUT_WAV}")

# Whisper 转文字
model = whisper.load_model("base")
result = model.transcribe(OUTPUT_WAV, language="zh")
text = result["text"].strip()

# 保存文字结果
with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
    f.write(text)

print(f"📝 识别结果: {text}")
print(f"💾 已保存识别结果到 {OUTPUT_TXT}")
