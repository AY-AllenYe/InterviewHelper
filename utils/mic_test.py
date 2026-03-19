import sounddevice as sd
import numpy as np

def callback(indata, frames, time, status):
    volume = np.linalg.norm(indata)
    print(volume)

with sd.InputStream(callback=callback):
    while True:
        pass