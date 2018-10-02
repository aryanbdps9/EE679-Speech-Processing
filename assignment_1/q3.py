import numpy as np 
from matplotlib import pyplot as plt
import pylab
from scipy import signal
from scipy.io.wavfile import write
import hparams
import sys
from math import pi
from scipy.io.wavfile import write


F1 = np.array(hparams.formant_freq)
B1 = np.array(hparams.formant_bw)
Fs = hparams.samp_freq
T = 1.0/Fs
t0 = hparams.time_length
num_samples = Fs*t0
F0 = hparams.sig_freq
t = np.linspace(0, t0, num_samples)
sig = signal.square(2 * np.pi * F0 * t, duty=0.01)



# Calculate pole angles and radii
R = np.exp(-pi*B1/Fs)
theta = 2*pi*F1/Fs


poles = np.array([R * np.exp(1j*theta), R * np.exp(-1j*theta)])
zeros = np.zeros(poles.shape, poles.dtype)

b, a = signal.zpk2tf(zeros, poles, 1)
# import pdb; pdb.set_trace()
y = np.zeros(sig.shape, sig.dtype)
time = np.linspace(0, num_samples/float(Fs), num_samples, endpoint=False)

for i in range(len(sig)):
    y[i] = y[i] + a[0]*sig[i]
    for j in range(1, len(a)):
        if i-j >= 0:
            y[i] = y[i] - a[j]*y[i-j]
fig3 = plt.figure()
plt.title('Filter Output')
# ax3 = fig2.add_subplot(111)
plt.plot(time[0:1000], y[0:1000], 'b')
plt.ylabel('Filter output', color='b')
plt.xlabel('Time [seconds]')
pylab.savefig('./results/' + 'q3' + '_' + 'trial1' + '.png')
write('./results/' + 'q3' + '_' + ' trial1.wav', Fs, y)
plt.show()



