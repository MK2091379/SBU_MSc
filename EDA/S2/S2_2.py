import soundfile as sf
import numpy as np
import IPython.display as ipd
import librosa
y, sr = librosa.load("zack_hemsey-vengeance_www.myfreesongs.cc_.mp3")
print("Audio shape:", y.shape)
print("Sampling rate:", sr)
ipd.Audio(y, rate=sr)
import matplotlib.pyplot as plt
plt.plot(y)
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")
plt.title("Audio Waveform")
plt.show()
