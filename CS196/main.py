from __future__ import print_function
from scipy.io import wavfile

import background_mask_librosa
import spectral_gating
import wav_to_csv

import pydub
import librosa
import IPython


input_dir = "UtkarshSpeaking"

# convert m4a file to .wav
sound = pydub.AudioSegment.from_file("input/" + input_dir + ".m4a")
sound.export("input/" + input_dir + ".wav", format="wav")

# audio is a series of floating points
audio, sampling_rate = librosa.load("input/" + input_dir + ".wav", duration=120)

background_mask_librosa.backgroundOutput(audio, sampling_rate, input_dir)

# input original speaking audio, retrieve data and rate of recording
local_wave = "input/" + input_dir + ".wav"
rate, data = wavfile.read(local_wave)
data = data / 32768

# input solely background noise from background_mask_librosa.py, retrieve data and rate of recording
noise = "output/" + input_dir + "Background.wav"
rateN, dataN = wavfile.read(noise)
dataN = dataN / 32768

# delegate noise clip as extracted background noise and original audio as audio_clip_band_limited
noise_clip = dataN
audio_clip_band_limited = data

output = spectral_gating.removeNoise(audio_clip=audio_clip_band_limited, noise_clip=noise_clip)

cleaned_audio = IPython.display.Audio(data=output, rate=44100)

with open("output/" + input_dir + "BackgroundRemoved.wav", 'wb') as f:
    f.write(cleaned_audio.data)

wav_to_csv.wav2csv("output/" + input_dir + "BackgroundRemoved.wav")
