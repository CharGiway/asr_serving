import time
import whisper

model = whisper.load_model("base")
start = time.time()
result = model.transcribe("output.wav", language="zh")
print("识别结果:", result["text"])
print("耗时:", time.time() - start, "秒")
