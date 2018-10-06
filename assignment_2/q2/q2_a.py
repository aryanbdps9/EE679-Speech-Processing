import librosa
from scipy import signal
import numpy as np
from math import pi
import matplotlib.pyplot as plt
import hparams

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[int(result.size/2):]


files = hparams.files
file_index = hparams.file_index
colors = hparams.colors
displacements = hparams.displacements
orders = hparams.orders


y, samp_freq = librosa.load(files[file_index],sr=None)
y = y/32768
win_size = hparams.win_size/1000.0
num_samples = int(samp_freq*win_size)

window = y[int((len(y)-num_samples)/2):int((len(y) + num_samples)/2)]*np.hamming(num_samples)

#----------------------------------------------------------------------------------
# dft = np.fft.fft(window, hparams.dft_length)
# freq = np.fft.fftfreq(dft.shape[-1], 1/float(samp_freq))
# fig1 = plt.figure()
# plt.plot(freq[:int(len(freq)/2)], 20*np.log10(np.abs(dft[:int(len(dft)/2)])))
# plt.ylabel('DFT Amplitude')
# plt.xlabel('Frequency')
# plt.title('Spectra for '+files[file_index][-5:-4])
#----------------------------------------------------------------------------------


# if file_index in [0, 1, 2]:
#     for i in range(1, len(window)):
#         window[i] = window[i] - 15.0/16.0 * window[i-1]

dft = np.fft.fft(window, hparams.dft_length)
freq = np.fft.fftfreq(dft.shape[-1], 1/float(samp_freq))
# plt.plot(freq[:int(len(freq)/2)], 20*np.log10(np.abs(dft[:int(len(dft)/2)])), 'g')


# # fig2 = plt.figure()
# plt.title('LPC Filters: Frequency Response for '+files[file_index][-5:-4])
# plt.ylabel('Amplitude [dB]')
# plt.xlabel('Frequency')


# #----------------------------------------------------------------------------------
orders = np.multiply(orders,(samp_freq/8000))
for order in orders:
    order = int(order)

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
        reflec_coeffs = (R[i] - reflec_coeffs)/error[i - 1]

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


    # plt.plot(samp_freq*w/(2*pi), displacements[int(order*8000/samp_freq)] + 20 * np.log10(abs(h)), colors[order*8000/samp_freq])
#----------------------------------------------------------------------------------
    if file_index == 3:

        if order == 12 or order == 20:
            z, p, k = signal.tf2zpk(num_coeffs, coeffs)

            fig3 = plt.figure()
            
            plt.plot(np.real(z), np.imag(z), 'xb')
            plt.plot(np.real(p), np.imag(p), 'or')
            plt.legend(['Zeros', 'Poles'], loc=2)
            plt.grid()
            plt.title('Pole / Zero Plot | order = ' + str(order) )
            plt.ylabel('Real')
            plt.xlabel('Imaginary')
            
            plt.savefig( files[file_index][-5:-4] + '_pole-zero_'+str(order)+'_.png')


    else:
        if order == 6 or order == 10:
            z, p, k = signal.tf2zpk(num_coeffs, coeffs)

            fig3 = plt.figure()
            
            plt.plot(np.real(z), np.imag(z), 'xb')
            plt.plot(np.real(p), np.imag(p), 'or')
            plt.legend(['Zeros', 'Poles'], loc=2)
            plt.grid()
            plt.title('Pole / Zero Plot | order = ' + str(order) )
            plt.ylabel('Real')
            plt.xlabel('Imaginary')
            
    
            plt.savefig( files[file_index][-5:-4] + '_pole-zero_'+str(order)+'_.png')            

# plt.legend(['Orig', 'Order 4', 'Order 6', 'Order 8', 'Order 10', 'Order 12', 'Order 20'])
# plt.grid()
# plt.show()