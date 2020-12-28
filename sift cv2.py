import cv2


# img1 = cv2.imread('all_souls_000002.jpg')
# gray= cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

# sift = cv2.xfeatures2d.SIFT_create()
# kp = sift.detect(gray,None)

# img1=cv2.drawKeypoints(gray,kp,img1)

# cv2.imwrite('sift_keypoints.jpg',img1)
# cv2.imshow('image',img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
img1 = cv.imread('all_souls_000002.jpg',cv.IMREAD_GRAYSCALE)          # queryImage
img2 = cv.imread('all_souls_000015.jpg',cv.IMREAD_GRAYSCALE) # trainImage
# Initiate SIFT detector
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()