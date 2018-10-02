from scipy import signal
import numpy as np
from math import pi
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import pylab

F1 = 900
B1 = 200
Fs = 16000
T = 1.0/Fs


# Calculate pole angles and radii
R = np.exp(-pi*B1/Fs)
theta = 2*pi*F1/Fs

# Get poles and an equal number of zeros
poles = np.array([R * np.exp(1j*theta), R * np.exp(-1j*theta)])
zeros = np.zeros(poles.shape, poles.dtype)

b, a = signal.zpk2tf(zeros, poles, 1)

w, h = signal.freqz(b, a)
fig1 = plt.figure()
plt.title('Single Formant Resonator - Frequency Response')
plt.plot(Fs*w/(2*pi), 20 * np.log10(abs(h)), 'b')
plt.ylabel('Amplitude [dB]', color='b')
plt.xlabel('Frequency')
pylab.savefig('./results/' + 'q1' + '_' + 'freq_response' + '.png')

fig2 = plt.figure()
angles = np.unwrap(np.angle(h))
plt.plot(Fs*w/(2*pi), angles, 'g')
plt.ylabel('Angle (in radians)', color='g')
plt.grid()
plt.axis('tight')
pylab.savefig('./results/' + 'q1' + '_' + 'phase_response' + '.png')



pulse = np.zeros([200], 'float64')
pulse[0] = 1
y = np.zeros(pulse.shape, pulse.dtype)
time = np.linspace(0, len(pulse)/float(Fs), 200, endpoint=False)

for i in range(len(pulse)):
    y[i] = y[i] + a[0]*pulse[i]
    for j in range(1, len(a)):
        if i-j >= 0:
            y[i] = y[i] - a[j]*y[i-j]
fig3 = plt.figure()
plt.title('Single Formant Resonator - Impulse Response')
ax3 = fig2.add_subplot(111)
plt.plot(time, y, 'b')
plt.ylabel('Impulse Response', color='b')
plt.xlabel('Time [in seconds]')
pylab.savefig('./results/' + 'q1' + '_' + 'impulse_response' + '.png')

