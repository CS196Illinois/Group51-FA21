from __future__ import print_function
from scipy.io import wavfile

import background_mask_librosa
import spectral_gating
import ravdesscsv

import pydub
import librosa
import IPython


input_direc = "OjuSpeaking"

# convert m4a file to .wav
sound = pydub.AudioSegment.from_file("input/" + input_direc + ".m4a")
sound.export("input/" + input_direc + ".wav", format="wav")

# audio is a series of floating points
audio, sampling_rate = librosa.load("input/" + input_direc + ".wav", duration=120)

# input original speaking audio, retrieve data and rate of recording
local_wave = "input/" + input_direc + ".wav"
rate, data = wavfile.read(local_wave)
data = data / 32768

# input solely background noise from background_mask_librosa.py, retrieve data and rate of recording
noise = "output/" + input_direc + "Background.wav"
rateN, dataN = wavfile.read(noise)
dataN = dataN / 32768

# delegate noise clip as extracted background noise and original audio as audio_clip_band_limited
noise_clip = dataN
audio_clip_band_limited = data

background_mask_librosa.backgroundOutput(audio, sampling_rate, input_direc)

output = spectral_gating.removeNoise(audio_clip=audio_clip_band_limited, noise_clip=noise_clip)

cleaned_audio = IPython.display.Audio(data=output, rate=44100)

with open("output/" + input_direc + "BackgroundRemoved.wav", 'wb') as f:
    f.write(cleaned_audio.data)
