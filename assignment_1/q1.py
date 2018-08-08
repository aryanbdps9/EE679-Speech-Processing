import numpy as np 
import librosa
from matplotlib.pyplot import figure
from matplotlib import pyplot as plt
import pylab
import utils

F1 = 900
B1 = 200
Fs = 16000
T = 1.0/Fs
t0 = 1
num_samples = Fs*t0
t = np.linspace(0, t0, num_samples)
freq = np.fft.fftfreq(len(t), d=T)
z = np.exp(1j*2*np.pi*freq*T) 


def get_freq_response(save_flag=False, plot_flag=True):
	freq_response = utils.resonator(z, F1, B1, Fs)
	mag_freq_response = np.abs(freq_response)
	mag_freq_response = np.log(mag_freq_response)
	plt.plot(freq[:-len(freq)/2], mag_freq_response[:-len(mag_freq_response)/2])
	if save_flag:
		pylab.savefig('./results/' + 'q1' + '_' + 'mag_response' + 'png')
	if plot_flag:
		plt.show()

def get_impulse_response(save_flag=False, plot_flag=True):
	freq_response = utils.resonator(z, F1, B1, Fs)
	# import pdb; pdb.set_trace()
	impulse_response = np.fft.ifft(freq_response)

	fig = figure()
	ax = fig.add_subplot(111, autoscale_on=True)
	ax.set_title('impulse response of a first formant resonator')
	plt.plot(impulse_response[0:300])

	# sections = np.split(impulse_response, 20)
	# variances = []
	# for x in sections:
	# 	variances.append(np.var(x))
	# figure()
	# plt.plot(variances)	
	# import pdb; pdb.set_trace()

	if save_flag:
		pylab.savefig('./results/' + 'q1' + '_' + 'mag_response' + 'png')
	zp = utils.ZoomPan()
	
	scale = 1.1
	figZoom = zp.zoom_factory(ax, base_scale = scale)
	figPan = zp.pan_factory(ax)
	if plot_flag:
		plt.show()
	return impulse_response		

impulse_response = get_impulse_response(save_flag=False, plot_flag=True)