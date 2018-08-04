import numpy as np 
import librosa
from matplotlib.pyplot import figure
from matplotlib import pyplot as plt
import pylab
def resonator(z, F1, B1, Fs): 
	k = 10.0
	num = k 
	T = 1.0/Fs
	r = np.exp(-B1*np.pi*T)
	theta = 2*np.pi*F1*T
	den = 1 - 2*r*np.cos(theta)*np.power(z,-1) + np.square(r)*np.power(z, -2)
	response = num/den

	return response

F1 = 900
B1 = 200
Fs = 16000
T = 1.0/Fs
t = np.linspace(0,5000,num=10000)
freq = np.fft.fftfreq(10000)*Fs
#omega = np.linspace(0,5000,num=10000)
#omega = omega/(2*np.pi)
z = np.exp(1j*2*np.pi*freq*T) 

freq_response = resonator(z, F1, B1, Fs)
mag_freq_response = np.abs(freq_response)
mag_freq_response = np.log(mag_freq_response)
# plt.plot(omega, mag_freq_response)
# pylab.savefig('./results/' + 'q1' + '_' + 'mag_response' + 'png')
# plt.show()


impulse_response = np.fft.ifft(freq_response)
import pdb; pdb.set_trace()

