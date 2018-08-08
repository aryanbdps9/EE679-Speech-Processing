import numpy as np 
import librosa
from matplotlib.pyplot import figure
from matplotlib import pyplot as plt
import pylab
from scipy import signal
from scipy.io.wavfile import write
from matplotlib.pyplot import figure
import hparams
import utils

# Create a signal of frequency F0Hz. 
# Signal time length = t0 seconds
# Signal sampling frequency is Fs
def create_periodic_train(F0, Fs, t0, save_flag=False, plot_flag=True):
	num_samples = Fs*t0
	t = np.linspace(0, 0.5, num_samples)
	periodic_train = signal.sawtooth(2 * np.pi * F0 * t)
	plt.plot(periodic_train)
	if save_flag: 
		pylab.savefig('./results/' + 'q2' + '_' + 'periodic_train' + '.png')
	if plot_flag:
		plt.show()
	write('signal.wav', Fs, periodic_train)	
	return periodic_train	


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

def time_conv_output(signal, impulse_response, save_flag, plot_flag):
	output = np.convolve(signal, impulse_response, mode='same') 

	write('time_conv_output.wav', Fs, output.real)

	if save_flag: 
		fig = figure()
		ax = fig.add_subplot(111, autoscale_on=True)
		ax.set_title('Time domain convolution output')
		plt.plot(output)
		zp = utils.ZoomPan()
		scale = 1.1
		figZoom = zp.zoom_factory(ax, base_scale = scale)
		figPan = zp.pan_factory(ax)		
		pylab.savefig('./results/' + 'q2' + '_' + 'output_time_conv' + '.png')

	if plot_flag: 
		fig = figure()
		ax = fig.add_subplot(111, autoscale_on=True)
		ax.set_title('Time domain convolution output')
		plt.plot(output)
		zp = utils.ZoomPan()
		scale = 1.1
		figZoom = zp.zoom_factory(ax, base_scale = scale)
		figPan = zp.pan_factory(ax)		
		plt.show()
		
def freq_mult_output(signal, freq_response, save_flag, plot_flag):
	signal_fft = np.fft.fft(signal)
	output = np.multiply(freq_response, signal)
	output = np.fft.ifft(fft_output)
	fig = figure()
	ax = fig.add_subplot(111, autoscale_on=True)
	ax.set_title('frequency domain multiplication output')
	plt.plot(output[0:500])
	zp = ZoomPan()
	scale = 1.1
	figZoom = zp.zoom_factory(ax, base_scale = scale)
	figPan = zp.pan_factory(ax)

	write('freq_mult_output.wav', Fs, output.real)
	plt.show()




F1 = 900
B1 = 200
Fs = 16000
T = 1.0/Fs
t0 = hparams.time_length
num_samples = Fs*t0
t = np.linspace(0, t0, num_samples)
freq = np.fft.fftfreq(len(t), d=T)
z = np.exp(1j*2*np.pi*freq*T) 

F0 = hparams.sig_freq

impulse_response = get_impulse_response(save_flag=False, plot_flag=False)

signal = create_periodic_train(F0, Fs, t0, save_flag=False, plot_flag=False)
output = time_conv_output(signal, impulse_response, save_flag=False, plot_flag=True)
