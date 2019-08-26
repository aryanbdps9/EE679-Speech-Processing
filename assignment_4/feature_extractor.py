import numpy as np 
import librosa
import params
import scipy 

frame_size = params.frame_size
frame_stride = params.frame_stride
num_ceps = params.num_ceps
def framing(input_signal, sample_rate):
    frame_length = frame_size * sample_rate
    frame_step = frame_stride * sample_rate  
    signal_length = len(input_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(input_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    frames = frames * np.hamming(frame_length)

    return frames


def frame_wise_fft(frames):
    fft_length = params.fft_length
    mag_frames = np.abs(np.fft.rfft(frames, fft_length))
    pow_frames = ((1.0 / fft_length) * ((mag_frames) ** 2))   # frame wise power spectrum

    return pow_frames

def filter_banks(frames, sample_rate):
    '''
    We can convert between Hertz (f) and Mel (m) using the following equation
    m = 2595* log10(1 + f * 700)
    '''
    fft_length = params.fft_length
    num_filters = params.num_filters


    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, num_filters + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((fft_length + 1) * hz_points / sample_rate)

    fbank = np.zeros((num_filters, int(np.floor(fft_length / 2 + 1))))
    for m in range(1, num_filters + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB
    return filter_banks


def mfcc(filter_banks, num_ceps):
    cep_lifter = 22 
    mfcc = scipy.fftpack.dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13


    (nframes, ncoeff) = mfcc.shape
    n = np.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mfcc *= lift

    filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)


    mfcc -= (np.mean(mfcc, axis=0) + 1e-8)

    return mfcc


def main(filepath):
	input_signal, sample_rate = librosa.load(filepath, sr=None)
	frames = framing(input_signal, sample_rate)
	pow_frames = frame_wise_fft(frames)
	filter_banks_frames = filter_banks(pow_frames, sample_rate)
	mfcc_coeffs = mfcc(filter_banks_frames, num_ceps)

	return np.transpose(mfcc_coeffs)



