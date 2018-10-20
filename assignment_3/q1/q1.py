from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import params

F0 = params.F0
sounds = params.sounds

a_den = params.a_den
n_den = params.n_den
i_den = params.i_den
s_den = params.s_den
den = [a_den, n_den, i_den, s_den]

a_num = params.a_num
n_num = params.n_num
i_num = params.i_num
s_num = params.s_num
num = [a_num, n_num, i_num, s_num]

for i in range(len(sounds)):
    
    duration = params.duration
    samp_freq = params.samp_freq
    t = np.linspace(0, duration, duration*samp_freq, endpoint=False)
    sig = (1 + signal.square(2 * np.pi * F0 * t, duty=0.01))/2
    if i == 3:
        samp_freq = samp_freq*2
        sig = np.random.normal(0, 1, int(duration*samp_freq))

    
    result = signal.lfilter(num[i], den[i], sig)
    
    result = signal.lfilter(np.asarray([1.0, 0.0]), np.asarray([1, -15.0/16.0]), result)
    fig = plt.figure()
    plt.plot(result[0:1000])
    plt.title("Sound: " + sounds[i])
    plt.xlabel('time')
    plt.ylabel('signal')
    fig.savefig("Sound: " + sounds[i] + ".pdf", bbox_inches='tight')
    wavfile.write(sounds[i] + '.wav', samp_freq, np.int16(result/np.max(result)*32767))

