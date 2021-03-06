from skimage.io import imread
import numpy as np
from nearest_neighbors_matching import NN_im1_im2, keypoints_to_frames
import cv2 as cv
import matplotlib.pyplot as plt

def Lowe_NN(frames1, descrs1, descrs2, matches):
    N_frames1 = frames1.shape[0]
    NN_threshold = 0.8
    ratio=np.zeros((N_frames1,1))
    for i in range(N_frames1):
        NN1_arg=np.argmin(np.sqrt(np.sum((descrs2-descrs1[i])**2,axis=1)))
        NN1 = np.min(np.sqrt(np.sum((descrs2-descrs1[i])**2,axis=1)))
        descrs2_copy=np.delete(descrs2,NN1_arg,axis=0)
        NN2 = np.min(np.sqrt(np.sum((descrs2_copy-descrs1[i])**2,axis=1)))
        ratio[i]=NN1/NN2
    filtered_indices = np.flatnonzero(ratio<NN_threshold)
    filtered_matches = matches[filtered_indices,:]
    return filtered_matches

def plot_NN_Lowe(filtered_matches, im1, im2, frames1, frames2):
    plt.figure()
    plt.imshow(np.concatenate((im1,im2),axis=1))
    for idx in range(filtered_matches.shape[0]):
        i=filtered_matches[idx,0]
        j=filtered_matches[idx,1]
        plt.gca().scatter([frames1[i,0],im1.shape[1]+frames2[j,0]], [frames1[i,1],frames2[j,1]], s=5, c='green') 
        plt.plot([frames1[i,0],im1.shape[1]+frames2[j,0]],[frames1[i,1],frames2[j,1]],linewidth=0.5)
    plt.axis('off')
    plt.show()   

if __name__ == "__main__":

    # loading images
    img1 = imread('all_souls_000002.jpg')
    img2 = imread('all_souls_000015.jpg')

    sift = cv.SIFT_create()
    keypoints1, descrs1 = sift.detectAndCompute(img1,None)
    keypoints2, descrs2 = sift.detectAndCompute(img2,None)

    frames1 = keypoints_to_frames(keypoints1)
    frames2 = keypoints_to_frames(keypoints2)

    # matches
    matches = NN_im1_im2(frames1, descrs1, descrs2)

    # filtered matches
    filtered_matches = Lowe_NN(frames1, descrs1, descrs2, matches)

    # plot
    plot_NN_Lowe(filtered_matches, img1, img2, frames1, frames2)
