import os
import scipy
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy import spatial
from scipy.cluster.vq import vq,kmeans
import params
import feature_extractor

sample_rate = params.sample_rate

digits = ['zero','one','two','three','four','five','six','seven','eight','nine']

seg_male_dir = params.seg_male_dir
seg_female_dir = params.seg_female_dir

male_speakers = os.listdir(seg_male_dir)
female_speakers = os.listdir(seg_female_dir)

all_speakers = male_speakers + female_speakers
all_speakers.remove('.DS_Store')
codeBook = {}
n_frame_dict = {}

for digit in digits:
    print("Preparing codebook for digit", digit)
    codeBook[digit] = {}
    n_frame_dict[digit] = {}

    for speaker in all_speakers:
        print("speaker = ", speaker)
        parent_dir = ''
        if speaker in male_speakers:
            parent_dir = seg_male_dir

        if speaker in female_speakers:
            parent_dir = seg_female_dir
        codeBook[digit][speaker] = []
        n_frame_dict[digit][speaker] = []
        for iteration in range(1,3):        
            file = parent_dir + '/' + speaker + '/' + str(digit) + '_' + str(iteration) + '.wav'
            feature_mat = feature_extractor.main(file)
            n_frame_dict[digit][speaker].append(feature_mat.shape[1])
            for i in range(feature_mat.shape[1]):
                codeBook[digit][speaker].append(feature_mat[:,i])

print('Codebook created')
confusion_matrix = np.zeros([10,10])

for test_speaker in all_speakers:
    train_speakers = all_speakers
    train_speakers.remove(test_speaker)

    for test_digit in digits:
        for utterance in range(1,3):

            n_frames = n_frame_dict[test_digit][test_speaker]
            test_mat = codeBook[test_digit][test_speaker][sum(n_frames[0:(utterance-1)]):sum(n_frames[0:utterance])]

            sum_dist = np.zeros(10)
            for digit in digits:
                for l in range(len(test_mat)):
                    test_vec = np.asarray(test_mat)[l,:]    
                    min_dist = float('Inf')
                    for speaker in train_speakers:
                        temp_list = codeBook[digit][speaker]
                        curr_dist,index = spatial.KDTree(temp_list).query(test_vec)
                        if(curr_dist < min_dist):
                            min_dist = curr_dist

                    sum_dist[digits.index(digit)]+=min_dist

            pred_digit = np.argmin(sum_dist)
            print( "For ", test_speaker, " predicted digit = ", pred_digit, " ground truth = ", test_digit)
            confusion_matrix[digits.index(test_digit),pred_digit] += 1

    np.save('BOF_confusion_matrix',confusion_matrix)


wer = 1 - np.trace(confusion_matrix)/640
print wer