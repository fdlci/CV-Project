from numpy import all, any, array, arctan2, cos, sin, exp, dot, log, logical_and, roll, sqrt, stack, trace, unravel_index, pi, deg2rad, rad2deg, where, zeros, floor, full, nan, isnan, round, float32
from numpy.linalg import det, lstsq, norm
from cv2 import resize, GaussianBlur, subtract, KeyPoint, INTER_LINEAR, INTER_NEAREST, imread
from functools import cmp_to_key
import numpy as np
import matplotlib.pyplot as plt

def number_of_octaves(image):
    return int(np.log(min(image.shape))/np.log(2) -1)

def initial_blur(image, sigma, assumed_blur):

    # Multiply the size of the image by two
    image = resize(image, (0, 0), fx=2, fy=2, interpolation=INTER_LINEAR)

    # set a value of initial blur
    sig = np.sqrt(max((sigma ** 2) - ((2 * assumed_blur) ** 2), 0.01))

    # set initial blur to sigma
    return GaussianBlur(image, (0, 0), sigmaX=sig, sigmaY=sig)

def gaussian_kernel(sigma, num_sets):

    # Initializing the blur_scale by sigma
    num_img = num_sets + 3
    k = 2**(1/num_sets)
    blur_scale = np.zeros(num_img)
    blur_scale[0] = sigma

    # multiplying the blur by 2**(1/s) to achieve twice the blur
    for i in range(1, num_img):
        sigma_previous = k**(i-1) * sigma
        sigma_after = k * sigma_previous
        blur_scale[i] = np.sqrt(sigma_after**2 - sigma_previous**2)

    return blur_scale

def generate_octaves(image, num_octaves, blur_scale):

    octaves = []

    # Computing num_ocatves, each with num_sets images
    for i in range(num_octaves):
        octave =  []
        octave.append(image)

        for blur in blur_scale[1:]:
            image = GaussianBlur(image, (0, 0), sigmaX=blur, sigmaY=blur)
            octave.append(image)
        octaves.append(octave)

        # Resizing the image (division by 2) to use it as base for next octave
        new_base = octave[-3]
        image = resize(new_base, (int(new_base.shape[1] / 2), int(new_base.shape[0] / 2)), interpolation=INTER_NEAREST)

    return np.array(octaves, dtype=object)

def generate_DOG_octaves(gaussian_octaves):

    dog_octaves = []

    for octave in gaussian_octaves:
        dog_oct = []
        for i in range(1,len(octave)):
            dog_oct.append(subtract(octave[i], octave[i-1]))
        dog_octaves.append(dog_oct)

    return array(dog_octaves, dtype=object)

def grad_and_hessian(im):

    #computing the gradient
    dx = (im[1, 1, 2] - im[1, 1, 0])/2
    dy = (im[1, 2, 1] - im[1, 0, 1])/2
    ds = (im[2, 1, 1] - im[0, 1, 1])/2

    grad = np.array([dx, dy, ds])

    #computing the Hessian
    cpixel = im[1, 1, 1]
    dxx = im[1, 1, 2] - 2 * cpixel + im[1, 1, 0]
    dyy = im[1, 2, 1] - 2 * cpixel + im[1, 0, 1]
    dss = im[2, 1, 1] - 2 * cpixel + im[0, 1, 1]
    dxy = (im[1, 2, 2] - im[1, 2, 0] - im[1, 0, 2] + im[1, 0, 0])/4
    dxs = (im[2, 1, 2] - im[2, 1, 0] - im[0, 1, 2] + im[0, 1, 0])/4
    dys = (im[2, 2, 1] - im[2, 0, 1] - im[0, 2, 1] + im[0, 0, 1])/4

    hess =  np.array([[dxx, dxy, dxs], 
                  [dxy, dyy, dys],
                  [dxs, dys, dss]])

    return grad, hess

def is_extremum(im1, im2, im3, thresh):

    # checks if the middle pixel is an extremum
    n, p = im1.shape
    new_sub_im = np.zeros((n,p,3))
    new_sub_im[:,:,0] = im1
    new_sub_im[:,:,1] = im2
    new_sub_im[:,:,2] = im3
    # take the pixel in the middle of the 3x3x3 subimage
    interest_pixel = im2[1,1]

    if abs(interest_pixel) > thresh:
        if interest_pixel == np.amax(new_sub_im):
            return True
        elif interest_pixel == np.amin(new_sub_im):
            return True
    return False

def outside_image(i, j, width, image_shape0, image_shape1, img_ind, num_sets):

    if i < width or i >= image_shape0 - width or j < width or j >= image_shape1 - width or img_ind < 1 or img_ind > num_sets:
        return True
    return False

def localize_extremum(i, j, img_ind, oct_ind, num_sets, dog_oct, sigma, thresh, width, r=10, max_iter=5):

    # starting with the keypoint inside the image
    loc_out = False

    image_shape = dog_oct[0].shape

    for cpt in range(max_iter):

        # Takes three consecutive images of the considered DoG octave
        im1, im2, im3 = dog_oct[img_ind-1:img_ind+2]

        #convert pixel values to values between (0,1) to apply Lowe's thresh
        interest_area = np.stack([im1[i-1:i+2, j-1:j+2], im2[i-1:i+2, j-1:j+2], im3[i-1:i+2, j-1:j+2]]).astype('float32') / 255

        # Compute gradient and Hessian of the pixel cube
        grad, hess = grad_and_hessian(interest_area)

        # Computing the subpixel offset
        subpixel_offset = -lstsq(hess, grad, rcond=None)[0]

        # If one of the absolute values of the coordinates is below 0.5, get rid of kp
        max_abs = np.amax(abs(subpixel_offset))
        if max_abs < 0.5:
            break
        
        # Compute position of new extrema and image_ind
        j += int(round(subpixel_offset[0]))
        i += int(round(subpixel_offset[1]))
        img_ind += int(round(subpixel_offset[2])) 

        # Check if new extrema is in the considered frame of the image
        if outside_image(i, j, width, image_shape[0], image_shape[1], img_ind, num_sets):
            loc_out = True
            break
    
    # Keypoint is outside the frame considered
    if loc_out:
        return None
    
    # Too many attempts without convergence
    if cpt >= max_iter - 1:
        return None

    # Computing value of new extrema
    new_extrema_value = interest_area[1,1,1] + np.dot(grad, subpixel_offset)/2

    # Eliminating Low contrast
    if abs(new_extrema_value)*num_sets >= thresh:

        # Eliminating Edge responses
        trace_hess = np.trace(hess[:2, :2])
        determinant = det(hess[:2, :2])

        if determinant > 0 and r * trace_hess < (r+1)*determinant:
            k = 2**oct_ind
            x = (j + subpixel_offset[0])*k
            y = (i + subpixel_offset[1])*k
            s = subpixel_offset[2]

            # creating a keypoint class
            keypoint = KeyPoint()
            # keypoint coordinates
            keypoint.pt = (x,y)
            # from which octave the keypoint has been extracted
            keypoint.octave = oct_ind + img_ind * (2 ** 8) + int(round((s + 0.5) * 255)) * (2 ** 16)
            # diameter of the meaningful leypoint neighborhood
            keypoint.size = sigma * (2 ** ((img_ind + s) / np.float32(num_sets))) * (2 ** (oct_ind + 1))
            # Value of the intensity of the keypoint
            keypoint.response = abs(new_extrema_value)

            return keypoint, img_ind

    return None

def compute_orientations(keypoint, octave_index, gaussian_image, radius_factor=3, num_bins=36, peak_ratio=0.8, scale_factor=1.5):
    """Compute orientations for each keypoint
    """
    keypoints_with_orientations = []
    image_shape = gaussian_image.shape

    scale = scale_factor * keypoint.size / float32(2 ** (octave_index + 1))  # compare with keypoint.size computation in localizeExtremumViaQuadraticFit()
    radius = int(round(radius_factor * scale))
    weight_factor = -0.5 / (scale ** 2)
    raw_histogram = zeros(num_bins)
    smooth_histogram = zeros(num_bins)

    for i in range(-radius, radius + 1):
        region_y = int(round(keypoint.pt[1] / float32(2 ** octave_index))) + i
        if region_y > 0 and region_y < image_shape[0] - 1:
            for j in range(-radius, radius + 1):
                region_x = int(round(keypoint.pt[0] / float32(2 ** octave_index))) + j
                if region_x > 0 and region_x < image_shape[1] - 1:
                    dx = gaussian_image[region_y, region_x + 1] - gaussian_image[region_y, region_x - 1]
                    dy = gaussian_image[region_y - 1, region_x] - gaussian_image[region_y + 1, region_x]
                    gradient_magnitude = sqrt(dx * dx + dy * dy)
                    gradient_orientation = rad2deg(arctan2(dy, dx))
                    weight = exp(weight_factor * (i ** 2 + j ** 2))  # constant in front of exponential can be dropped because we will find peaks later
                    histogram_index = int(round(gradient_orientation * num_bins / 360.))
                    raw_histogram[histogram_index % num_bins] += weight * gradient_magnitude

    for n in range(num_bins):
        smooth_histogram[n] = (6 * raw_histogram[n] + 4 * (raw_histogram[n - 1] + raw_histogram[(n + 1) % num_bins]) + raw_histogram[n - 2] + raw_histogram[(n + 2) % num_bins]) / 16.
    orientation_max = max(smooth_histogram)
    orientation_peaks = where(logical_and(smooth_histogram > roll(smooth_histogram, 1), smooth_histogram > roll(smooth_histogram, -1)))[0]
    for peak_index in orientation_peaks:
        peak_value = smooth_histogram[peak_index]
        if peak_value >= peak_ratio * orientation_max:
            left_value = smooth_histogram[(peak_index - 1) % num_bins]
            right_value = smooth_histogram[(peak_index + 1) % num_bins]
            interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) / (left_value - 2 * peak_value + right_value)) % num_bins
            orientation = 360. - interpolated_peak_index * 360. / num_bins
            if abs(orientation - 360.) < float_tolerance:
                orientation = 0
            new_keypoint = KeyPoint(*keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
            keypoints_with_orientations.append(new_keypoint)
    return keypoints_with_orientations

def all_keypoints(gaussian_octaves, dog_octaves, num_sets, sigma, width, thresh=0.04):


    threshold = floor(0.5 * thresh / num_sets * 255) #from opencv implementation
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
        keypoint.pt = tuple(0.5 * array(keypoint.pt))
        keypoint.size *= 0.5
        keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1) & 255)
        converted_keypoints.append(keypoint)
    return converted_keypoints

# def unpackOctave(keypoint):
#     """Compute octave, layer, and scale from a keypoint
#     """
#     octave = keypoint.octave & 255
#     layer = (keypoint.octave >> 8) & 255
#     if octave >= 128:
#         octave = octave | -128
#     scale = 1 / float32(1 << octave) if octave >= 0 else float32(1 << -octave)
#     return octave, layer, scale

# def generateDescriptors(keypoints, gaussian_images, window_width=4, num_bins=8, scale_multiplier=3, descriptor_max_value=0.2):
#     """Generate descriptors for each keypoint
#     """
#     descriptors = []

#     for keypoint in keypoints:
#         octave, layer, scale = unpackOctave(keypoint)
#         gaussian_image = gaussian_images[octave + 1, layer]
#         num_rows, num_cols = gaussian_image.shape
#         point = round(scale * array(keypoint.pt)).astype('int')
#         bins_per_degree = num_bins / 360.
#         angle = 360. - keypoint.angle
#         cos_angle = cos(deg2rad(angle))
#         sin_angle = sin(deg2rad(angle))
#         weight_multiplier = -0.5 / ((0.5 * window_width) ** 2)
#         row_bin_list = []
#         col_bin_list = []
#         magnitude_list = []
#         orientation_bin_list = []
#         histogram_tensor = zeros((window_width + 2, window_width + 2, num_bins))   # first two dimensions are increased by 2 to account for border effects

#         # Descriptor window size (described by half_width) follows OpenCV convention
#         hist_width = scale_multiplier * 0.5 * scale * keypoint.size
#         half_width = int(round(hist_width * sqrt(2) * (window_width + 1) * 0.5))   # sqrt(2) corresponds to diagonal length of a pixel
#         half_width = int(min(half_width, sqrt(num_rows ** 2 + num_cols ** 2)))     # ensure half_width lies within image

#         for row in range(-half_width, half_width + 1):
#             for col in range(-half_width, half_width + 1):
#                 row_rot = col * sin_angle + row * cos_angle
#                 col_rot = col * cos_angle - row * sin_angle
#                 row_bin = (row_rot / hist_width) + 0.5 * window_width - 0.5
#                 col_bin = (col_rot / hist_width) + 0.5 * window_width - 0.5
#                 if row_bin > -1 and row_bin < window_width and col_bin > -1 and col_bin < window_width:
#                     window_row = int(round(point[1] + row))
#                     window_col = int(round(point[0] + col))
#                     if window_row > 0 and window_row < num_rows - 1 and window_col > 0 and window_col < num_cols - 1:
#                         dx = gaussian_image[window_row, window_col + 1] - gaussian_image[window_row, window_col - 1]
#                         dy = gaussian_image[window_row - 1, window_col] - gaussian_image[window_row + 1, window_col]
#                         gradient_magnitude = sqrt(dx * dx + dy * dy)
#                         gradient_orientation = rad2deg(arctan2(dy, dx)) % 360
#                         weight = exp(weight_multiplier * ((row_rot / hist_width) ** 2 + (col_rot / hist_width) ** 2))
#                         row_bin_list.append(row_bin)
#                         col_bin_list.append(col_bin)
#                         magnitude_list.append(weight * gradient_magnitude)
#                         orientation_bin_list.append((gradient_orientation - angle) * bins_per_degree)

#         for row_bin, col_bin, magnitude, orientation_bin in zip(row_bin_list, col_bin_list, magnitude_list, orientation_bin_list):
#             # Smoothing via trilinear interpolation
#             # Notations follows https://en.wikipedia.org/wiki/Trilinear_interpolation
#             # Note that we are really doing the inverse of trilinear interpolation here (we take the center value of the cube and distribute it among its eight neighbors)
#             row_bin_floor, col_bin_floor, orientation_bin_floor = floor([row_bin, col_bin, orientation_bin]).astype(int)
#             row_fraction, col_fraction, orientation_fraction = row_bin - row_bin_floor, col_bin - col_bin_floor, orientation_bin - orientation_bin_floor
#             if orientation_bin_floor < 0:
#                 orientation_bin_floor += num_bins
#             if orientation_bin_floor >= num_bins:
#                 orientation_bin_floor -= num_bins

#             c1 = magnitude * row_fraction
#             c0 = magnitude * (1 - row_fraction)
#             c11 = c1 * col_fraction
#             c10 = c1 * (1 - col_fraction)
#             c01 = c0 * col_fraction
#             c00 = c0 * (1 - col_fraction)
#             c111 = c11 * orientation_fraction
#             c110 = c11 * (1 - orientation_fraction)
#             c101 = c10 * orientation_fraction
#             c100 = c10 * (1 - orientation_fraction)
#             c011 = c01 * orientation_fraction
#             c010 = c01 * (1 - orientation_fraction)
#             c001 = c00 * orientation_fraction
#             c000 = c00 * (1 - orientation_fraction)

#             histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, orientation_bin_floor] += c000
#             histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c001
#             histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, orientation_bin_floor] += c010
#             histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c011
#             histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, orientation_bin_floor] += c100
#             histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c101
#             histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, orientation_bin_floor] += c110
#             histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c111

#         descriptor_vector = histogram_tensor[1:-1, 1:-1, :].flatten()  # Remove histogram borders
#         # Threshold and normalize descriptor_vector
#         threshold = norm(descriptor_vector) * descriptor_max_value
#         descriptor_vector[descriptor_vector > threshold] = threshold
#         descriptor_vector /= max(norm(descriptor_vector), float_tolerance)
#         # Multiply by 512, round, and saturate between 0 and 255 to convert from float32 to unsigned char (OpenCV convention)
#         descriptor_vector = round(512 * descriptor_vector)
#         descriptor_vector[descriptor_vector < 0] = 0
#         descriptor_vector[descriptor_vector > 255] = 255
#         descriptors.append(descriptor_vector)
#     return array(descriptors, dtype='float32')

def main(image, sigma=1.6, num_intervals=3, assumed_blur=0.5, image_border_width=5):
    """Compute SIFT keypoints and descriptors for an input image
    """
    image = image.astype('float32')
    base_image = initial_blur(image, sigma, assumed_blur)
    num_octaves = number_of_octaves(base_image)
    gaussian_kernels = gaussian_kernel(sigma, num_intervals)
    gaussian_images = generate_octaves(base_image, num_octaves, gaussian_kernels)
    dog_images = generate_DOG_octaves(gaussian_images)
    keypoints = all_keypoints(gaussian_images, dog_images, num_intervals, sigma, image_border_width)
    keypoints = removeDuplicateKeypoints(keypoints)
    keypoints = convertKeypointsToInputImageSize(keypoints)
    # descriptors = generateDescriptors(keypoints, gaussian_images)
    return keypoints#, descriptors

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
    float_tolerance = 1e-7
    keypoints = main(image, sigma=1.6, num_intervals=3, assumed_blur=0.5, image_border_width=5)
    plotting_keypoints(keypoints, plt.imread('all_souls_000002.jpg'))