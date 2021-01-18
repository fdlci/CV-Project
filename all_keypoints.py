import numpy as np
from cv2 import imread
import matplotlib.pyplot as plt
from functools import cmp_to_key
from builduing_DoG_octaves import building_DOG_octaves
from finding_keypoints import is_extremum, localize_extremum
from orientation import compute_orientations

def all_keypoints(gaussian_octaves, dog_octaves, num_sets, sigma, width, thresh=0.04):


    threshold = np.floor(0.5 * thresh / num_sets * 255) #from opencv implementation
    all_keypoints = []

    for oct_ind, dog_oct in enumerate(dog_octaves):
        for img_ind, (im1, im2, im3) in enumerate(zip(dog_oct, dog_oct[1:], dog_oct[2:])):
            n , p = im1.shape[0], im1.shape[1]
            for i in range(width, n-width):
                for j in range(width, p-width):
                    if is_extremum(im1[i-1:i+2, j-1:j+2], im2[i-1:i+2, j-1:j+2], im3[i-1:i+2, j-1:j+2], threshold):
                        loc = localize_extremum(i, j, img_ind + 1, oct_ind, num_sets, dog_oct, sigma, thresh, width)
                        if loc is not None:
                            keypoint, localization = loc
                            keypoints_oriented = compute_orientations(keypoint, oct_ind, gaussian_octaves[oct_ind][localization])

                            for kp in keypoints_oriented:
                                all_keypoints.append(kp)
    return all_keypoints

def compareKeypoints(keypoint1, keypoint2):
    """Return True if keypoint1 is less than keypoint2
    """
    if keypoint1.pt[0] != keypoint2.pt[0]:
        return keypoint1.pt[0] - keypoint2.pt[0]
    if keypoint1.pt[1] != keypoint2.pt[1]:
        return keypoint1.pt[1] - keypoint2.pt[1]
    if keypoint1.size != keypoint2.size:
        return keypoint2.size - keypoint1.size
    if keypoint1.angle != keypoint2.angle:
        return keypoint1.angle - keypoint2.angle
    if keypoint1.response != keypoint2.response:
        return keypoint2.response - keypoint1.response
    if keypoint1.octave != keypoint2.octave:
        return keypoint2.octave - keypoint1.octave
    return keypoint2.class_id - keypoint1.class_id

def removeDuplicateKeypoints(keypoints):
    """Sort keypoints and remove duplicate keypoints
    """
    if len(keypoints) < 2:
        return keypoints

    keypoints.sort(key=cmp_to_key(compareKeypoints))
    unique_keypoints = [keypoints[0]]

    for next_keypoint in keypoints[1:]:
        last_unique_keypoint = unique_keypoints[-1]
        if last_unique_keypoint.pt[0] != next_keypoint.pt[0] or \
           last_unique_keypoint.pt[1] != next_keypoint.pt[1] or \
           last_unique_keypoint.size != next_keypoint.size or \
           last_unique_keypoint.angle != next_keypoint.angle:
            unique_keypoints.append(next_keypoint)
    return unique_keypoints

def convertKeypointsToInputImageSize(keypoints):
    """Convert keypoint point, size, and octave to input image size
    """
    converted_keypoints = []
    for keypoint in keypoints:
        keypoint.pt = tuple(0.5 * np.array(keypoint.pt))
        keypoint.size *= 0.5
        keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1) & 255)
        converted_keypoints.append(keypoint)
    return converted_keypoints

def plotting_keypoints(keyp, img):
    x_list = []
    y_list = []
    for kp in keyp:
        kx, ky = kp.pt[0], kp.pt[1]
        x_list.append(kx)
        y_list.append(ky)
    plt.imshow(img)
    plt.plot(x_list,y_list, 'o')
    plt.show()

if __name__ == "__main__":
    image = imread('all_souls_000002.jpg', 0)
    sigma, assumed_blur, num_sets, width = 1.6, 0.5, 3, 5
    DoG_octaves, gaussian_octaves = building_DOG_octaves(image, sigma, assumed_blur, num_sets)
    keypoints = all_keypoints(gaussian_octaves, DoG_octaves, num_sets, sigma, width, thresh=0.04)
    keypoints = removeDuplicateKeypoints(keypoints)
    keypoints = convertKeypointsToInputImageSize(keypoints)
    plotting_keypoints(keypoints, plt.imread('all_souls_000002.jpg'))
