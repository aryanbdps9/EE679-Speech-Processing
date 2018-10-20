from scipy import signal
import numpy as np
from math import pi
import matplotlib.pyplot as plt
from scipy.io import wavfile
import params

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[int(result.size/2):]


[sample_rate, y] = wavfile.read('a.wav');
y = y/32768.0
window_duration = params.window_duration 
num_samples = sample_rate*window_duration
fft_length = params.fft_length
fig = plt.figure()

window = y[int((len(y) - num_samples)/2):int((len(y) + num_samples)/2)]*np.hamming(num_samples)


log_fft = np.log10(np.abs(np.fft.fft(window, fft_length)))
cepstrum = np.real(np.fft.ifft(log_fft))


N_cep = params.N_cep
cepstrum[N_cep:(cepstrum.shape[-1]-N_cep)] = 0

# Take the DFT and plot it
cepstrum_dft = np.abs(np.fft.fft(cepstrum, fft_length))
freq = np.fft.fftfreq(cepstrum_dft.shape[-1], 1/float(sample_rate))

plt.plot(freq[:len(freq)/2], 20*np.abs(log_fft[:len(log_fft)/2]), 'r')
plt.plot(freq[:len(freq)/2], 20*np.abs(cepstrum_dft[:len(cepstrum_dft)/2]), 'b')



order = params.order

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
    reflec_coeffs = (R[i] - reflec_coeffs)/error[i-1]

    coeffs[i] = reflec_coeffs

    for j in range(1, i):
        coeffs[j] = dummy_coeffs[j] - reflec_coeffs*dummy_coeffs[i-j]

    error[i] = (1-np.square(reflec_coeffs))*error[i-1]

coeffs[0] = 1.0
coeffs[1:len(coeffs)] = -coeffs[1:len(coeffs)]
num_coeffs = np.zeros(coeffs.shape)
num_coeffs[0] = 1
G[i] = np.sqrt(error[i])
w, h = signal.freqz(num_coeffs, coeffs)

plt.plot(sample_rate*w/(2*pi), 20 * np.log10(abs(h)), 'c')
plt.ylabel('DFT Amplitude')
plt.xlabel('Frequency')
plt.legend(['Original Signal', 'Cepstral', 'LP'])
plt.grid()
fig.savefig('a_winlen=' + str(window_duration*1000) +  ".pdf", bbox_inches='tight')

plt.show()