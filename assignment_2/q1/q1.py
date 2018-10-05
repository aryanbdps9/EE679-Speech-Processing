import librosa
import numpy as np 
from matplotlib import pyplot as plt
import pylab
from scipy import signal
from scipy.io.wavfile import write
import hparams
import sys
from math import pi


def gen_vowel():
	F1 = np.array(hparams.formant_freq)
	B1 = np.array(hparams.formant_bw)
	Fs = hparams.samp_freq
	T = 1.0/Fs
	t0 = hparams.time_length
	num_samples = Fs*t0
	F0 = hparams.sig_freq
	t = np.linspace(0, t0, num_samples)

	# sig = signal.sawtooth(2 * np.pi * F0 * t, width=1)
	sig = signal.square(2 * np.pi * F0 * t, duty=0.01)

	# Calculate pole angles and radii
	R = np.exp(-pi*B1/Fs)
	theta = 2*pi*F1/Fs


	poles = np.concatenate([R * np.exp(1j*theta), R * np.exp(-1j*theta)])
	zeros = np.zeros(poles.shape, poles.dtype)
	b, a = signal.zpk2tf(zeros, poles, 1)

	y = np.zeros(sig.shape, sig.dtype)
	# time = np.linspace(0, num_samples/Fs, num_samples, endpoint=False)

	for i in range(len(sig)):
	    y[i] = y[i] + a[0]*sig[i]
	    for j in range(1, len(a)):
	        if i-j >= 0:
	            y[i] = y[i] - a[j]*y[i-j]

	write('120.wav', int(Fs), y)


def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size/2:]	

y, sr = librosa.load('120.wav',sr=None) 
filter_order = hparams.filter_order

for order in filter_order:

	
	size = order + 1
	coeffs = np.zeros(size)
	dummy_coeffs = np.zeros(size)
	

	r = autocorr(y)
	sig_eng = r[0]
	
	 

	for i in range(1, order + 1): 
		
		reflec_coeff = 0
		dummy_coeffs[1:len(coeffs)] = coeffs[1:len(coeffs)]
		
		for j in range(1, i): 
			reflec_coeff = reflec_coeff + dummy_coeffs[j]*r[i-j]

		coeffs[m] = reflec_coeff	
		for j in range(1, i):
			coeffs[j] = dummy_coeffs[j] - reflec_coeff*dummy_coeffs[i - j]  


		sig_eng = (1 - np.square(reflec_coeff))	


	coeffs[0] = 1.0 
	coeffs[1:size] = -coeffs[1:size] 	
	zeros = np.zeros(size)
	zeros[0] = 1
	w, h = signal.freqz(zeros, coeffs)
	plt.plot(hparams.samp_freq*w/(2*pi), 10*order + 20 * np.log10(abs(h)), {2: 'r', 4: 'g', 6: 'b', 8: 'y', 10: 'm'})

formant_freq = hparams.formant_freq
formant_bw = hparams.formant_bw

r = np.exp(-pi*formant_bw/hparams.samp_freq)
theta = 2*pi*formant_freq/hparams.samp_freq

poles = np.concatenate([R * np.exp(1j*theta), R * np.exp(-1j*theta)])
zeros = np.zeros(poles.shape, poles.dtype)

b, a = signal.zpk2tf(zeros, poles, 1)

wf, hf = signal.freqz(b, a)

plt.plot(hparams.samp_freq*wf/(2*pi), 120 + 20 * np.log10(abs(hf)), 'c')
plt.legend(['Order 2', 'Order 4', 'Order 6', 'Order 8', 'Order 10', 'Orig'])
plt.grid()


plt.show()






