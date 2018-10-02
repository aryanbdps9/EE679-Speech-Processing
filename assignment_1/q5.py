import numpy as np 
from matplotlib import pyplot as plt
import pylab
from scipy import signal
from scipy.io.wavfile import write
import hparams
import sys
from math import pi
from scipy.io.wavfile import write, read

Fs = hparams.samp_freq
win_size = hparams.win_size/1000.0;
window_samples = int(win_size*Fs)
t0 = hparams.time_length
num_samples = Fs*t0

output = read('./results/q4_ 220_e.wav')
output = output[1]


window = output[:window_samples]*np.hamming(window_samples)
dft = np.fft.fft(window, n=hparams.dft_len)
freq = np.fft.fftfreq(dft.shape[-1], 1/Fs)


f,a = plt.subplots()
plt.subplot(211)
plt.title('DFT for /e/ at 220Hz')
plt.plot(freq, 20*np.log10(np.abs(dft)))
plt.ylabel('Hamming window', color='b')
plt.xlabel('Frequency')
plt.grid()
plt.axis('tight')

def onclick(event):
    print [event.xdata,event.ydata]
f.canvas.mpl_connect('button_press_event', onclick)

window = output[:window_samples]
dft = np.fft.fft(window, n=hparams.dft_len)
freq = np.fft.fftfreq(dft.shape[-1], 1/Fs)

plt.subplot(212)
plt.plot(freq, 20*np.log10(np.abs(dft)))
plt.ylabel('Rect window', color='b')
plt.xlabel('Frequency')
plt.grid()
plt.axis('tight')

def onclick(event):
    print [event.xdata,event.ydata]
f.canvas.mpl_connect('button_press_event', onclick)
plt.show()