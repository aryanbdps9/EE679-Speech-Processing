import os
import scipy
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import librosa
import params
from scipy.io.wavfile import write,read



data_dir = params.data_dir
male_dir = data_dir + '/' + params.male_dir
female_dir = data_dir + '/' + params.female_dir

male_files = os.listdir(male_dir)
male_names =[] 
for male_file in male_files:
    male_names.append(male_file.split('.')[0][13:])
if (not os.path.isdir("male_segmented")):
    os.mkdir('male_segmented')

female_files = os.listdir(female_dir)
female_names =[] 
for female_file in female_files:
    female_names.append(female_file.split('.')[0][13:])
if (not os.path.isdir("female_segmented")):
    os.mkdir('female_segmented')

def pre_emphasis(input_signal):
    '''
    A pre-emphasis filter is useful in several ways: 
    - balance the frequency spectrum since high frequencies usually have smaller magnitudes compared to lower frequencies,
    - avoid numerical problems during the Fourier transform operation and 
    - may also improve the Signal-to-Noise Ratio (SNR).
    Equation: y[t] = x[t] - alpha * x[t-1] 
    '''
    pre_emphasis_alpha = params.pre_emphasis_alpha 
    pre_emphasized_signal = np.append(input_signal[0], input_signal[1:] - pre_emphasis_alpha * input_signal[:-1]) 
    return pre_emphasized_signal


def get_limiting_indices(y):
    y = y/np.max(y)
    energy_threshold = params.energy_threshold
    window_len = 3500

    window = np.hamming(window_len)
    sig_energy = np.convolve(y**2,window**2,'same')
    
    
    sig_energy = sig_energy/max(sig_energy)     #Normalize energy
    sig_energy_thresh = (sig_energy > energy_threshold).astype('float')
    
    #convert the bar graph to impulses by subtracting signal from it's shifted version 
    indices = np.nonzero(abs(sig_energy_thresh[1:] - sig_energy_thresh[0:-1]))[0]         
    
    start_indices = [indices[2*i] for i in range(len(indices)/2)]
    end_indices   = [indices[2*i+1] for i in range(len(indices)/2)]

    return start_indices, end_indices





digits = params.digits
male_files = np.sort(male_files)
male_names = np.sort(male_names)
female_files = np.sort(female_files)
female_names = np.sort(female_names)

for i in range(len(male_names)):
    print('Segmenting audio files of ' + male_names[i])
    y, sr = librosa.load( male_dir + '/' + male_files[i], sr=None)   
    start_indices, end_indices = get_limiting_indices(y)
    
    if (not os.path.isdir("male_segmented/" + male_names[i])):
        os.mkdir("male_segmented/" + male_names[i])
    
    for p in range(len(end_indices)):
        sig = y[start_indices[p] : end_indices[p]]
        digit = pre_emphasis(sig)
        
        write("male_segmented/" + male_names[i] + '/' + digits[int(np.floor(p/2))] +'_' + str(p%2 + 1) +'.wav', sr, digit)

for i in range(len(female_names)):
    print('Segmenting audio files of ' + female_names[i])
    y, sr = librosa.load( female_dir + '/' + female_files[i], sr=None)   
    start_indices, end_indices = get_limiting_indices(y)
    
    if (not os.path.isdir("female_segmented/" + female_names[i])):
        os.mkdir("female_segmented/" + female_names[i])
    
    for p in range(len(end_indices)):
        sig = y[start_indices[p] : end_indices[p]]
        digit = pre_emphasis(sig)
        
        write("female_segmented/" + female_names[i] + '/' + digits[int(np.floor(p/2))] +'_' + str(p%2 + 1) +'.wav', sr, digit)





