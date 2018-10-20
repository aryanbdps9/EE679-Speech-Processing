from scipy import signal
import numpy as np
from math import pi
import matplotlib.pyplot as plt
from scipy.io import wavfile
import params

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[int(result.size/2):]


filenames = params.filenames
    
for filename in filenames:
    samp_rate, y = wavfile.read(filename)
    y = y/32768
    duration = params.duration
    num_samples = samp_rate*duration 
    fft_length = params.fft_length

    window = y[int((len(y) - num_samples)/2):int((len(y) + num_samples)/2)]*np.hamming(num_samples)

    log_fft = np.log10(np.abs(np.fft.fft(window, fft_length)))
    cepstrum = np.real(np.fft.ifft(log_fft))


    cep_N =  params.cep_N
    cepstrum[cep_N:(cepstrum.shape[-1]-cep_N)] = 0


    cepstrum_fft = np.abs(np.fft.fft(cepstrum, fft_length))
    freq = np.fft.fftfreq(cepstrum_fft.shape[-1], 1/float(samp_rate))
    fig = plt.figure()
    plt.plot(freq[:len(freq)/2], 20*np.abs(log_fft[:len(log_fft)/2]), 'r')
    plt.plot(freq[:len(freq)/2], 20*np.abs(cepstrum_fft[:len(cepstrum_fft)/2]), 'b')


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
    plt.plot(samp_rate*w/(2*pi), 20 * np.log10(abs(h)), 'c')
    plt.ylabel('DFT Amplitude for sound ' + filename)
    plt.xlabel('Frequency')
    plt.legend(['Original', 'Cepstral Envelope', 'LP Envelope'])
    plt.grid()

    fig.savefig('q3'+ filename.split('.')[0] + ".pdf", bbox_inches='tight')

plt.show()    