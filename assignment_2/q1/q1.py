import librosa
import numpy as np 
from matplotlib import pyplot as plt
import pylab
from scipy import signal
from scipy.io.wavfile import write
import hparams
import sys
from math import pi


def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[int(result.size/2):]


sig_freq = hparams.sig_freq

y, samp_freq = librosa.load( str(sig_freq) + '.wav',sr=None)
win_size = hparams.win_size/1000.0 
num_samples = int(samp_freq*win_size)
window = y[:num_samples]*np.hamming(num_samples)

fig1 = plt.figure()
plt.title('LPC Filters: Frequency Response for /a/ at '+str(sig_freq)+' Hz')
plt.ylabel('Amplitude [dB]')
plt.xlabel('Frequency [rad/sample]')

colors = {2: 'r', 4: 'g', 6: 'b', 8: 'y', 10: 'm'}
orders = hparams.filter_order

for order in orders:
    
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

    plt.plot(samp_freq*w/(2*np.pi), 10*order + 20 * np.log10(abs(h)), colors[order])

formant_freq = hparams.formant_freq
formant_bw = hparams.formant_bw

r = np.exp(np.multiply(-np.pi, formant_bw)/samp_freq)
theta = 2*np.multiply(np.pi, formant_freq)/samp_freq

poles = np.concatenate([r * np.exp(1j*theta), r * np.exp(-1j*theta)])
zeros = np.zeros(poles.shape, poles.dtype)

b, a = signal.zpk2tf(zeros, poles, 1)

wf, hf = signal.freqz(b, a)

plt.plot(samp_freq*wf/(2*pi), 120 + 20 * np.log10(abs(hf)), 'c')
plt.legend(['Order 2', 'Order 4', 'Order 6', 'Order 8', 'Order 10', 'Orig'])
plt.grid()

plt.show()