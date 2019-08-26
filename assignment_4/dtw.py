import os
import scipy
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.cluster.vq import vq,kmeans
import params
import feature_extractor


def find_dtw_distance(test_pattern,ref_pattern):
	n = test_pattern.shape[1]
	m = ref_pattern.shape[1]
	
	distMat = np.zeros([n, m])

	for i in range(n):
		for j in range(m):

			distMat[i, j] = np.linalg.norm(np.subtract(test_pattern[:, i], ref_pattern[:, j]))

	DTW = np.zeros([n+1, m+1])

	for i in range(1,n+1):
		DTW[i, 0] = float('Inf')

	for i in range(1,m+1):
		DTW[0, i] = float('Inf')

	DTW[0,0] = 0

	for i in range(1,n+1):
		for j in range(1,m+1):
			cost = distMat[i-1, j-1]
			DTW[i, j] = cost + np.min([DTW[i-1, j], np.min([DTW[i-1, j-1], DTW[i, j-1]])])

	return DTW[n, m]


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
			min_dist = float('Inf')
			for digit in digits:

				for speaker in train_speakers:
					for ref_utterance in range(1,3):
						n_ref_frames = n_frame_dict[digit][speaker]
						ref_mat = codeBook[digit][speaker][sum(n_ref_frames[0:(ref_utterance-1)]):sum(n_ref_frames[0:ref_utterance])]

						test_pattern = np.transpose(np.asarray(test_mat))
						ref_pattern = np.transpose(np.asarray(ref_mat))
						if np.sum(ref_pattern.shape) == 0:
							continue 
						curr_dist = find_dtw_distance(test_pattern,ref_pattern)
						if(curr_dist < min_dist):
							min_dist = curr_dist
							pred_digit = digits.index(digit)

			print( "For ", test_speaker, " predicted digit = ", pred_digit, " ground truth = ", test_digit)
			confusion_matrix[digits.index(test_digit),pred_digit] += 1

	np.save('DTW_confusion_matrix',confusion_matrix)

wer = 1 - np.trace(confusion_matrix)/640

confusion_matrix = confusion_matrix/64
d = np.around(confusion_matrix,decimals = 3)
np.savetxt("../report/DTW_confusion_matrix.csv", d, fmt = '%s')

