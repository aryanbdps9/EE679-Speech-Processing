import librosa
from scipy import signal
import numpy as np
from math import pi
import matplotlib.pyplot as plt
import hparams

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[int(result.size/2):]


files = hparams.files
file_index = hparams.file_index


y, samp_freq = librosa.load(files[file_index],sr=None)
y = y/32768
win_size = hparams.win_size/1000.0
num_samples = int(samp_freq*win_size)

window = y[int((len(y)-num_samples)/2):int((len(y)+num_samples)/2)]*np.hamming(num_samples)

if file_index in [0, 1, 2]:
    for i in range(1, len(window)):
        window[i] = window[i] - 15.0/16.0 * window[i-1]


order = 10

R = autocorr(window)
error = np.zeros(order+1)
error[0] = R[0]
G = np.zeros(order + 1)

coeffs = np.zeros(order+1)
dummy_coeffs = np.zeros(order + 1)


for i in range(1, order +1):
    reflec_coeffs = 0
    dummy_coeffs[1:len(coeffs)] = coeffs[1:len(coeffs)]
    
    for j in range(1, i):
        reflec_coeffs = reflec_coeffs + dummy_coeffs[j]*R[i-j]
    reflec_coeffs = (R[i] - reflec_coeffs)/error[i - 1]

    coeffs[i] = reflec_coeffs

    for j in range(1, i):
        coeffs[j] = dummy_coeffs[j] - reflec_coeffs*dummy_coeffs[i-j]

    error[i] = (1-np.square(reflec_coeffs))*error[i-1]
    G[i] = np.sqrt(error[i])

coeffs[0] = 1.0
coeffs[1:len(coeffs)] = -coeffs[1:len(coeffs)]
num_coeffs = np.zeros(coeffs.shape)
num_coeffs[0] = 1

window_filter = signal.lfilter(coeffs, num_coeffs, window)
fig1 = plt.figure()
dft = np.fft.fft(window_filter, 1024)
freq = np.fft.fftfreq(dft.shape[-1], 1/float(samp_freq))
plt.title('Residual Signal for ' + files[file_index][-5:-4])
plt.ylabel('DFT Amplitude', color='b')
plt.xlabel('Frequency')
plt.grid()
plt.plot(freq[:len(freq)/2], 20*np.log10(np.abs(dft[:len(dft)/2])), 'b')

fig2 = plt.figure()
plt.title('Residual autocorrelation for ' + files[file_index][-5:-4])
plt.ylabel('Amplitude', color='b')
plt.xlabel('Sample')
plt.grid()
plt.plot(autocorr(window_filter))

plt.show()

