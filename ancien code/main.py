from skimage.io import imread
import matplotlib.pyplot as plt
from sift import SIFT
import numpy as np

if __name__ == '__main__':

    im = imread('all_souls_000002.jpg')
    sift_detector = SIFT(im)
    feat = sift_detector.get_features()
    kp_pyr = sift_detector.kp_pyr

    _, ax = plt.subplots(1, sift_detector.num_octave)

    for i in range(sift_detector.num_octave):
        ax[i].imshow(im)

        scaled_kps = kp_pyr[i] * (2**i)
        
        ax[i].scatter(scaled_kps[:,0], scaled_kps[:,1], c='r', s=2.5)

    plt.show()
