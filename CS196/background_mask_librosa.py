from __future__ import print_function
import numpy as np
import soundfile as sf

import librosa.display


def backgroundOutput(audio, sampling_rate, input_direc):
    # stft turns the audio into a spectogram, and magphase outputs the magnitude and phase of the spectogram
    full_spectogram, phase = librosa.magphase(librosa.stft(audio))

    # each spectogram column is replaced by a combination of its nearest neighbors, which reduces noise
    spectogram_filter = librosa.decompose.nn_filter(full_spectogram,
                                                    aggregate=np.median,
                                                    metric='cosine',
                                                    width=int(librosa.time_to_frames(2, sr=sampling_rate)))

    # to ensure the output isn't greater than the input, so taking the minimum of the reduced noise spectogram and regular spectogram ensures that
    spectogram_filter = np.minimum(full_spectogram, spectogram_filter)

    # constants for the softmasks
    margin_back, margin_fore = 2, 10
    power = 2

    # acts as a mask over the spectogram that permits only a certain level of transparency
    mask_back = librosa.util.softmask(spectogram_filter,
                                   margin_back * (full_spectogram - spectogram_filter),
                                   power=power)

    mask_fore = librosa.util.softmask(full_spectogram - spectogram_filter,
                                   margin_fore * spectogram_filter,
                                   power=power)

    # creates spectograms of the foreground and background
    foreground_spectogram = mask_fore * full_spectogram
    background_spectogram = mask_back * full_spectogram

    # creates a digital spectogram variable of the foreground
    digital_foreground = foreground_spectogram * phase
    digital_background = background_spectogram * phase

    # transforms the digital spectogram to audio file
    audio_foreground = librosa.istft(digital_foreground)
    audio_background = librosa.istft(digital_background)

    # writes to the given file path with the audio file at the given sampling rate
    sf.write("output/" + input_direc + "Foreground.wav", audio_foreground, sampling_rate)
    sf.write("output/" + input_direc + "Background.wav", audio_background, sampling_rate)
