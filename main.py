from funasr import AutoModel
import sounddevice as sd
import numpy as np
import librosa

chunk_size = [0, 100, 5]   # 600ms
encoder_chunk_look_back = 4
decoder_chunk_look_back = 1

model_dir = "models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"
model = AutoModel(model=model_dir)

mic_sample_rate = 44100
asr_sample_rate = 16000
chunk_stride = chunk_size[1] * 960
frames_per_buffer = chunk_stride

cache = {}

print("🎤 麦克风实时识别启动 (Ctrl+C 结束)")

with sd.InputStream(
        samplerate=mic_sample_rate,
        channels=1,
        dtype="float32",
        blocksize=frames_per_buffer) as stream:

    while True:
        audio_chunk, overflowed = stream.read(frames_per_buffer)

        speech_chunk = audio_chunk[:, 0]
        
        speech_chunk = librosa.resample(
            speech_chunk,
            orig_sr=mic_sample_rate,
            target_sr=asr_sample_rate
        )

        res = model.generate(
            input=speech_chunk,
            cache=cache,
            is_final=False,
            chunk_size=chunk_size,
            encoder_chunk_look_back=encoder_chunk_look_back,
            decoder_chunk_look_back=decoder_chunk_look_back
        )

        if len(res) > 0:
            print(res[0]["text"], end="", flush=True)