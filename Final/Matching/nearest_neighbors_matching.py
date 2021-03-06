from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv



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
    plt.axis('off')
    plt.show()

def keypoints_to_frames(keypoints):
    frames = np.zeros((len(keypoints),2))
    for i, keyp in enumerate(keypoints):
        # print(keyp.pt)
        frames[i,0], frames[i,1] = keyp.pt[0], keyp.pt[1]
    return frames

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

    # plot
    plot_NN(matches, frames1, frames2, img1, img2)