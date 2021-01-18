from cv2 import KeyPoint
import numpy as np

# def compute_orientation(keypoint, oct_ind, gaussian_image, radius_factor=3, num_bins=36, peak_ratio=0.8, scale_factor=1.5):

#     kp_oriented = []
#     img_shape = gaussian_image.shape
#     hist = np.zeros(num_bins)

#     # Defining the parameters of the given keypoint
#     # The scale_factor comes from the original SIFT paper
#     scale = scale_factor*keypoint.size/np.float32(2**(oct_ind+1))
#     radius = int(round(radius_factor * scale))
#     weight_factor = -0.5 / (scale ** 2)

#     # Updating the keypoint's coordinates according to the size of the gaussian
#     x = int(round(keypoint.pt[0]/2 ** oct_ind))
#     y = int(round(keypoint.pt[1]/2 ** oct_ind))

#     # Defining a region around the keypoint (2*radius x 2*radius)
#     for i in range(-radius, radius+1):
#         for j in range(-radius, radius+1):
#             # Checking it is inside the image
#             y_region = y + i
#             x_region = x + i
#             if y_region > 0 and y_region < img_shape[0] - 1 and x_region > 0 and x_region < img_shape[1] - 1:
#                 # Compute the gradient
#                 dx = gaussian_image[y_region, x_region + 1] - gaussian_image[y_region, x_region - 1]
#                 dy = gaussian_image[y_region - 1, x_region] - gaussian_image[y_region + 1, x_region]
#                 m = np.sqrt(dx * dx + dy * dy)
#                 theta = np.rad2deg(np.arctan2(dy, dx))    

#                 # Updating the histogram
#                 weight = np.exp(weight_factor * (i ** 2 + j ** 2))
#                 hist_ind = int(round(theta * num_bins / 360.))
#                 hist[hist_ind % num_bins] += weight * m
#     orientation = np.argmax(hist)*(360. / num_bins)
#     print(orientation)
#     new_keypoint = KeyPoint(*keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
#     kp_oriented.append(new_keypoint)

#     return kp_oriented


def compute_orientations(keypoint, octave_index, gaussian_image, radius_factor=3, num_bins=36, peak_ratio=0.8, scale_factor=1.5):
    """Compute orientations for each keypoint
    """
    keypoints_with_orientations = []
    image_shape = gaussian_image.shape

    scale = scale_factor * keypoint.size / np.float32(2 ** (octave_index + 1))  # compare with keypoint.size computation in localizeExtremumViaQuadraticFit()
    radius = int(round(radius_factor * scale))
    weight_factor = -0.5 / (scale ** 2)
    raw_histogram = np.zeros(num_bins)
    smooth_histogram = np.zeros(num_bins)

    for i in range(-radius, radius + 1):
        region_y = int(round(keypoint.pt[1] / np.float32(2 ** octave_index))) + i# size of the redduced image
        if region_y > 0 and region_y < image_shape[0] - 1:
            for j in range(-radius, radius + 1):
                region_x = int(round(keypoint.pt[0] / np.float32(2 ** octave_index))) + j
                if region_x > 0 and region_x < image_shape[1] - 1:
                    dx = gaussian_image[int(region_y), int(region_x) + 1] - gaussian_image[int(region_y), int(region_x) - 1]
                    dy = gaussian_image[int(region_y) - 1, int(region_x)] - gaussian_image[int(region_y) + 1, int(region_x)]
                    gradient_magnitude = np.sqrt(dx * dx + dy * dy)
                    gradient_orientation = np.rad2deg(np.arctan2(dy, dx))
                    weight = np.exp(weight_factor * (i ** 2 + j ** 2))  # constant in front of exponential can be dropped because we will find peaks later
                    histogram_index = int(round(gradient_orientation * num_bins / 360.))
                    raw_histogram[histogram_index % num_bins] += weight * gradient_magnitude

    for n in range(num_bins):
        smooth_histogram[n] = (6 * raw_histogram[n] + 4 * (raw_histogram[n - 1] + raw_histogram[(n + 1) % num_bins]) + raw_histogram[n - 2] + raw_histogram[(n + 2) % num_bins]) / 16.
    orientation_max = max(smooth_histogram)
    orientation_peaks = np.where(np.logical_and(smooth_histogram > np.roll(smooth_histogram, 1), smooth_histogram > np.roll(smooth_histogram, -1)))[0]
    for peak_index in orientation_peaks:
        peak_value = smooth_histogram[peak_index]
        if peak_value >= peak_ratio * orientation_max:
            left_value = smooth_histogram[(peak_index - 1) % num_bins]
            right_value = smooth_histogram[(peak_index + 1) % num_bins]
            interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) / (left_value - 2 * peak_value + right_value)) % num_bins
            orientation = 360. - interpolated_peak_index * 360. / num_bins
            if abs(orientation - 360.) < 1e-7:
                orientation = 0
            new_keypoint = KeyPoint(*keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
            keypoints_with_orientations.append(new_keypoint)
    return keypoints_with_orientations