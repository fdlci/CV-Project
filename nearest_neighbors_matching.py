from sift import SIFT
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt


def NN_im1_im2(frames1, descrs1, descrs2):
    N_frames1 = frames1.shape[0]
    matches=np.zeros((N_frames1,2),dtype=np.int)
    matches[:,0]=range(N_frames1)
    descrs1=descrs1.astype(np.float64)
    descrs2=descrs2.astype(np.float64)
    for i in range(N_frames1):
        mat = np.argmin(np.sqrt(np.sum((descrs2-descrs1[i])**2,axis=1)))
        matches[i,1]=mat
    return matches

def plot_NN(matches, frames1, frames2, im1, im2):
    N_frames1 = frames1.shape[0]
    plt.figure()
    plt.imshow(np.concatenate((im1,im2),axis=1))
    for i in range(N_frames1):
        j=matches[i,1]
        plt.gca().scatter([frames1[i,0],im1.shape[1]+frames2[j,0]], [frames1[i,1],frames2[j,1]], s=5, c='green')
        plt.plot([frames1[i,0],im1.shape[1]+frames2[j,0]],[frames1[i,1],frames2[j,1]],linewidth=0.5)
    plt.show()

def main_NN():

    # loading images
    im1 = imread('all_souls_000002.jpg')
    im2 = imread('all_souls_000015.jpg')

    sift_detector_1 = SIFT(im1)
    descrs1 = sift_detector_1.get_features()[0]
    frames1 = sift_detector_1.kp_pyr[0]

    sift_detector_2 = SIFT(im2)
    descrs2 = sift_detector_2.get_features()[0]
    frames2 = sift_detector_2.kp_pyr[0]

    # matches
    matches = NN_im1_im2(frames1, descrs1, descrs2)

    # plot
    plot_NN(matches, frames1, frames2, im1, im2)

# print(main_NN())