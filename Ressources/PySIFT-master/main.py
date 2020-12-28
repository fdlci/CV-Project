from skimage.io import imread
from sift import SIFT

import argparse
import pickle
import os
from os.path import isdir

import matplotlib.pyplot as plt

if __name__ == '__main__':
	# parser = argparse.ArgumentParser(description='PySIFT')
	# parser.add_argument('all_souls_000002.jpg', type=str, dest='input_fname')
	# parser.add_argument('jpg', type=str, dest='output_prefix', help='The prefix for the kp_pyr and feat_pyr files generated')
	# args = parser.parse_args()

	im = imread('all_souls_000002.jpg')

	sift_detector = SIFT(im)
	_ = sift_detector.get_features()
	kp_pyr = sift_detector.kp_pyr

	if not isdir('results'):
		os.mkdir('results')

	pickle.dump(sift_detector.kp_pyr, open('results/output_kp_pyr.pkl', 'wb'))
	pickle.dump(sift_detector.feats, open('results/output_feat_pyr.pkl', 'wb'))

	_, ax = plt.subplots(1, sift_detector.num_octave)
	
	for i in range(sift_detector.num_octave):
		ax[i].imshow(im)

		scaled_kps = kp_pyr[i] * (2**i)
		ax[i].scatter(scaled_kps[:,0], scaled_kps[:,1], c='r', s=2.5)

	plt.show()
