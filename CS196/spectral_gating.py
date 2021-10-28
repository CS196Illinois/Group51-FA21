import IPython
from scipy.io import wavfile
import scipy.signal
import numpy as np
import librosa


# from: https://timsainburg.com/noise-reduction-python.html
# fourier transformation given necessary parameters, finds frequency and phase over specified intervals in a spectrogram
def _stft(y, n_fft, hop_length, win_length):
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


# inverse fourier transformation given necessary parameters, converts given parameters to a synthesized signal
def _istft(y, hop_length, win_length):
    return librosa.istft(y, hop_length, win_length)


# amplitude to decibel function with necessary constants
def _amp_to_db(x):
    return librosa.core.amplitude_to_db(x, ref=1.0, amin=1e-20, top_db=80.0)


# decibel to amplitude function with delegated constants
def _db_to_amp(x,):
    return librosa.core.db_to_amplitude(x, ref=1.0)


# core function that removes noise from audio given a clip of noise, extrapolates to the entire audio
def removeNoise(audio_clip, noise_clip, n_grad_freq=2, n_grad_time=4, n_fft=2048, win_length=2048, hop_length=512,
                n_std_thresh=1.5, prop_decrease=1.0):

    global start

    # STFT over noise
    noise_stft = _stft(noise_clip, n_fft, hop_length, win_length)

    # amplitude of waveform converted to decibel
    noise_stft_db = _amp_to_db(np.abs(noise_stft))

    # find mean, std, and noise threshold
    mean_freq_noise = np.mean(noise_stft_db, axis=1)
    std_freq_noise = np.std(noise_stft_db, axis=1)
    noise_thresh = mean_freq_noise + std_freq_noise * n_std_thresh

    # fourier transformation on audio
    sig_stft = _stft(audio_clip, n_fft, hop_length, win_length)

    # computed amplitude from transformed spectrogram into decibels
    sig_stft_db = _amp_to_db(np.abs(sig_stft))

    # computed mask by converted signal to all positive values, turning those amplitude values from the spectrogram
    mask_gain_db = np.min(_amp_to_db(np.abs(sig_stft)))

    # Create a smoothing filter for the mask in time and frequency
    smoothing_filter = np.outer(
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_freq + 1, endpoint=False),
                np.linspace(1, 0, n_grad_freq + 2),
            ]
        )[1:-1],
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_time + 1, endpoint=False),
                np.linspace(1, 0, n_grad_time + 2),
            ]
        )[1:-1],
    )
    smoothing_filter = smoothing_filter / np.sum(smoothing_filter)

    # calculate the threshold for each frequency/time bin
    db_thresh = np.repeat(
        np.reshape(noise_thresh, [1, len(mean_freq_noise)]),
        np.shape(sig_stft_db)[1],
        axis=0,
    ).T

    # mask if the signal is above the threshold
    sig_mask = sig_stft_db < db_thresh

    # convolve the mask with a smoothing filter
    sig_mask = scipy.signal.fftconvolve(sig_mask, smoothing_filter, mode="same")
    sig_mask = sig_mask * prop_decrease

    # mask the signal
    sig_stft_db_masked = (
        sig_stft_db * (1 - sig_mask)
        + np.ones(np.shape(mask_gain_db)) * mask_gain_db * sig_mask
    )

    # mask real
    sig_imag_masked = np.imag(sig_stft) * (1 - sig_mask)
    sig_stft_amp = (_db_to_amp(sig_stft_db_masked) * np.sign(sig_stft)) + (
        1j * sig_imag_masked
    )

    # returns the spectrogram modified with the mask as an audio signal
    recovered_signal = _istft(sig_stft_amp, hop_length, win_length)
    recovered_spec = _amp_to_db(
        np.abs(_stft(recovered_signal, n_fft, hop_length, win_length))
    )

    return recovered_signal
