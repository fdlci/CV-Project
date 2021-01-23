from skimage.io import imread
import numpy as np
from nearest_neighbors_matching import NN_im1_im2
import cv2 as cv
import matplotlib.pyplot as plt

def ransac(frames1,frames2,matches,N_iters=10000,dist_thresh=15):
    # initialize
    max_inliers=0
    tnf=None
    # run random sampling
    for it in range(N_iters):
        # pick a random sample
        i = np.random.randint(0,frames1.shape[0])
        x_1,y_1,s_1,theta_1=frames1[i,:]
        j = matches[i,1]
        x_2,y_2,s_2,theta_2=frames2[j,:]

        # estimate transformation
        theta = (theta_1-theta_2)
        s = s_2/s_1
        t_x = x_2 - s*(x_1*np.cos(theta)-y_1*np.sin(theta))
        t_y = y_2 - s*(x_1*np.sin(theta)+y_1*np.cos(theta))

        # evaluate estimated transformation
        X_1 = frames1[:,0]
        Y_1 = frames1[:,1]
        X_2 = frames2[matches[:,1],0]
        Y_2 = frames2[matches[:,1],1]

        X_1_prime = s*(X_1*np.cos(theta)-Y_1*np.sin(theta))+t_x
        Y_1_prime = s*(X_1*np.sin(theta)+Y_1*np.cos(theta))+t_y
      
        dist = np.sqrt((X_1_prime-X_2)**2+(Y_1_prime-Y_2)**2)
        inliers_indices = np.flatnonzero(dist<dist_thresh)
        num_of_inliers = len(inliers_indices)

        # keep if best
        if num_of_inliers>max_inliers:
            max_inliers=num_of_inliers
            best_inliers_indices = inliers_indices
            tnf = [t_x,t_y,s,theta]

    return (tnf,best_inliers_indices)

def keypoints_to_frames(keypoints):
    frames = np.zeros((len(keypoints),4))
    for i, keyp in enumerate(keypoints):
        # print(keyp.pt)
        frames[i,0], frames[i,1] = keyp.pt[0], keyp.pt[1]
        frames[i,2] = keyp.size
        frames[i,3] = keyp.angle
    return frames

def plot_geom(filtered_matches, im1, im2, frames1, frames2):
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
    _,inliers_indices=ransac(frames1,frames2,matches)
    filtered_matches = matches[inliers_indices,:]

    # plot
    plot_geom(filtered_matches, img1, img2, frames1, frames2)