from funasr import AutoModel
import sounddevice as sd
import soundfile
import numpy as np
import librosa
from scipy.signal import resample_poly

import sys
from utils.logger import Logger

sys.stdout = Logger()

asr_chunk_size = [0, 10, 5]
vad_chunk_size = 200
encoder_chunk_look_back = 4
decoder_chunk_look_back = 1


# asr_model_dir = "models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"
asr_model_dir = "models/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
vad_model_dir = "models/speech_fsmn_vad_zh-cn-16k-common-pytorch"

# model = AutoModel(
#     model=asr_model_dir,
#     vad_model=vad_model_dir
# )

asr_model = AutoModel(
    model=asr_model_dir
)
vad_model = AutoModel(
    model=vad_model_dir
)

mic_sample_rate = 44100
asr_sample_rate = 16000
chunk_stride = int(asr_chunk_size[1] * mic_sample_rate * 600 / 1000)
frames_per_buffer = chunk_stride
# exp_wav = 'models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online/example/asr_example.wav'

buffer = np.zeros(0, dtype=np.float32)

silence_count = 0
silence_threshold = 8  # 0.8秒静音认为一句结束

cache = {}

with sd.InputStream(
        samplerate=mic_sample_rate,
        channels=1,
        dtype="float32",
        blocksize=frames_per_buffer) as stream:

    while True:
        audio, overflowed = stream.read(frames_per_buffer)        
        
        speech_chunk = audio[:, 0]
        
        # speech_chunk = librosa.resample(
        #     speech_chunk,
        #     orig_sr=mic_sample_rate,
        #     target_sr=asr_sample_rate
        # )
        
        speech_chunk = resample_poly(speech_chunk, asr_sample_rate, mic_sample_rate)

        # 加入buffer
        buffer = np.concatenate([buffer, speech_chunk])

        # 不够一个ASR chunk就继续采集
        if len(buffer) < chunk_stride:
            continue

        speech_chunk = buffer[:chunk_stride]
        buffer = buffer[chunk_stride:]

        res = asr_model.generate(
            input=speech_chunk,
            cache=cache,
            is_final=False,
            chunk_size=asr_chunk_size,
            # chunk_size=vad_chunk_size,
            # encoder_chunk_look_back=encoder_chunk_look_back,
            # decoder_chunk_look_back=decoder_chunk_look_back
        )
        
        
        if len(res) > 0:
            print(res[0]["text"], end="", flush=True)
        # if len(res[0]["value"]):
        #     print(res)
        